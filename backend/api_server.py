import json
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from game_logic import GameState


GAMES: Dict[str, GameState] = {}
GAME_RE = re.compile(r"^/api/game/([a-f0-9]+)$")
MOVE_RE = re.compile(r"^/api/game/([a-f0-9]+)/move$")


class DotsAndBoxesHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send_json(200, {"ok": True})
            return

        match = GAME_RE.match(self.path)
        if match:
            game_id = match.group(1)
            game = GAMES.get(game_id)
            if not game:
                self._send_json(404, {"error": "Game not found."})
                return
            self._send_json(200, {"gameId": game_id, "state": game.serialize()})
            return

        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if self.path == "/api/game":
            data = self._read_json()
            if data is None:
                self._send_json(400, {"error": "Invalid JSON."})
                return
            size_raw = data.get("size", 5) if isinstance(data, dict) else 5
            try:
                size = int(size_raw)
            except (TypeError, ValueError):
                size = 5
            size = max(2, min(size, 10))
            game_id = uuid.uuid4().hex
            GAMES[game_id] = GameState.new(size)
            self._send_json(200, {"gameId": game_id, "state": GAMES[game_id].serialize()})
            return

        match = MOVE_RE.match(self.path)
        if match:
            game_id = match.group(1)
            game = GAMES.get(game_id)
            if not game:
                self._send_json(404, {"error": "Game not found."})
                return
            data = self._read_json()
            if data is None:
                self._send_json(400, {"error": "Invalid JSON.", "state": game.serialize()})
                return
            if not isinstance(data, dict):
                self._send_json(400, {"error": "Request body must be a JSON object.", "state": game.serialize()})
                return
            try:
                x = int(data.get("x"))
                y = int(data.get("y"))
            except (TypeError, ValueError):
                self._send_json(400, {"error": "x and y must be integers.", "state": game.serialize()})
                return
            direction = data.get("direction", "")
            result = game.play_move(x, y, str(direction))
            if not result["valid"]:
                self._send_json(400, {"error": result["message"], "state": game.serialize(), "result": result})
                return
            self._send_json(200, {"state": game.serialize(), "result": result})
            return

        self._send_json(404, {"error": "Not found."})


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), DotsAndBoxesHandler)
    print(f"API server running on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
