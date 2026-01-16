import json
import os
import random
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Protocol, Tuple


class DotsBoxesState(Protocol):
    size: int
    horizontal: List[List[int]]
    vertical: List[List[int]]
    boxes: List[List[int]]
    scores: List[int]
    current_player: int
    done: bool


Move = Dict[str, object]

_API_KEYS = ("OPENAI_API_KEY", "CHATGPT_API_KEY")
_MODEL_KEYS = ("OPENAI_MODEL", "CHATGPT_MODEL")
_DEFAULT_MODEL = "gpt-4o-mini"
_dotenv_cache: Optional[Dict[str, str]] = None


def _available_moves(state: DotsBoxesState) -> List[Move]:
    moves: List[Move] = []
    for y in range(state.size + 1):
        for x in range(state.size):
            if state.horizontal[y][x] == 0:
                moves.append({"x": x, "y": y, "direction": "r"})
    for y in range(state.size):
        for x in range(state.size + 1):
            if state.vertical[y][x] == 0:
                moves.append({"x": x, "y": y, "direction": "d"})
    return moves


def _load_dotenv() -> Dict[str, str]:
    global _dotenv_cache
    if _dotenv_cache is not None:
        return _dotenv_cache
    values: Dict[str, str] = {}
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    values[key] = value
    except OSError:
        values = {}
    _dotenv_cache = values
    return values


def _get_env_value(keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    values = _load_dotenv()
    for key in keys:
        value = values.get(key)
        if value:
            return value
    return ""


def _normalize_move(obj: object) -> Optional[Move]:
    if not isinstance(obj, dict):
        return None
    if {"x", "y", "direction"} <= obj.keys():
        try:
            x = int(obj["x"])
            y = int(obj["y"])
            direction = str(obj["direction"]).lower()
        except (TypeError, ValueError):
            return None
        if direction not in ("r", "d"):
            return None
        return {"x": x, "y": y, "direction": direction}
    return None


def _parse_move_text(text: str) -> Optional[Move]:
    try:
        return _normalize_move(json.loads(text))
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return _normalize_move(json.loads(text[start : end + 1]))
    except json.JSONDecodeError:
        return None


def _openai_chat(api_key: str, model: str, messages: List[Dict[str, str]]) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"ChatGPT API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"ChatGPT API unavailable: {exc}") from exc
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("ChatGPT API response missing choices.") from exc


def choose_move(state: DotsBoxesState) -> Optional[Move]:
    moves = _available_moves(state)
    if not moves:
        return None

    api_key = _get_env_value(_API_KEYS)
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY (or CHATGPT_API_KEY) in .env or environment.")
    model = _get_env_value(_MODEL_KEYS) or _DEFAULT_MODEL

    state_payload = {
        "size": state.size,
        "currentPlayer": state.current_player,
        "scores": {"1": state.scores[1], "2": state.scores[2]},
        "horizontal": state.horizontal,
        "vertical": state.vertical,
        "boxes": state.boxes,
    }
    user_payload = {
        "state": state_payload,
        "moves": moves,
        "rules": "Pick exactly one move from moves. Respond with JSON only.",
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Dots and Boxes bot. Try to win by maximizing your score. "
                "Return ONLY a JSON object with keys x, y, direction "
                "using one of the provided moves. No extra text."
            ),
        },
        {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
    ]

    content = _openai_chat(api_key, model, messages)
    move = _parse_move_text(content)
    if not move:
        return random.choice(moves)
    move_key = (move["x"], move["y"], move["direction"])
    valid_moves = {(m["x"], m["y"], m["direction"]) for m in moves}
    if move_key not in valid_moves:
        return random.choice(moves)
    return move
