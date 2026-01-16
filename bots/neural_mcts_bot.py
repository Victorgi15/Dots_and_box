import os
from typing import Dict, List, Optional, Protocol, Tuple

from neural.mcts import NeuralMCTS


class DotsBoxesState(Protocol):
    size: int
    horizontal: List[List[int]]
    vertical: List[List[int]]
    boxes: List[List[int]]
    scores: List[int]
    current_player: int
    done: bool


Move = Dict[str, object]

_MODEL_KEYS = ("NEURAL_MODEL_PATH", "DOTSBOXES_MODEL_PATH")
_SIM_KEYS = ("NEURAL_MCTS_SIMULATIONS",)
_C_PUCT_KEYS = ("NEURAL_MCTS_C_PUCT",)
_dotenv_cache: Optional[Dict[str, str]] = None


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


def _parse_int(value: str, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _parse_float(value: str, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _resolve_model_path(size: int) -> str:
    env_value = _get_env_value(_MODEL_KEYS)
    if env_value:
        return env_value
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    size_dir = os.path.join(root, "models", f"size_{size}")
    latest = os.path.join(size_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    legacy = os.path.join(root, "models", f"size_{size}.pt")
    if os.path.exists(legacy):
        return legacy
    fallback = os.path.join(root, "models", "latest.pt")
    if os.path.exists(fallback):
        return fallback
    return latest


class NeuralMCTSBot:
    def __init__(
        self,
        model_path: str,
        size: int,
        n_simulations: int = 200,
        c_puct: float = 1.4,
        device: Optional[str] = None,
    ) -> None:
        if not os.path.exists(model_path):
            raise ValueError(f"Neural model not found at {model_path}")
        try:
            from neural.network import NeuralPolicy
        except ImportError as exc:
            raise ValueError(
                "Neural bot requires PyTorch. Install torch to use this bot."
            ) from exc
        self.policy = NeuralPolicy(model_path, board_size=size, device=device)
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def choose_move(self, state: DotsBoxesState) -> Optional[Move]:
        if state.done:
            return None
        mcts = NeuralMCTS(
            self.policy.predict,
            n_simulations=self.n_simulations,
            c_puct=self.c_puct,
        )
        distribution = mcts.get_move_distribution(state, temperature=0.0)
        if not distribution:
            return None
        best = max(distribution, key=lambda item: item["prob"])
        return {"x": best["x"], "y": best["y"], "direction": best["direction"]}


_BOT_CACHE: Dict[int, NeuralMCTSBot] = {}


def choose_move(state: DotsBoxesState) -> Optional[Move]:
    if state.size not in _BOT_CACHE:
        model_path = _resolve_model_path(state.size)
        n_sim = _parse_int(_get_env_value(_SIM_KEYS), 200)
        c_puct = _parse_float(_get_env_value(_C_PUCT_KEYS), 1.4)
        _BOT_CACHE[state.size] = NeuralMCTSBot(
            model_path=model_path,
            size=state.size,
            n_simulations=n_sim,
            c_puct=c_puct,
        )
    return _BOT_CACHE[state.size].choose_move(state)
