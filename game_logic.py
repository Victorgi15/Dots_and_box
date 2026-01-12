from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class GameState:
    size: int
    horizontal: List[List[int]]
    vertical: List[List[int]]
    boxes: List[List[int]]
    current_player: int = 1
    scores: List[int] = field(default_factory=lambda: [0, 0, 0])
    done: bool = False

    @classmethod
    def new(cls, size: int) -> "GameState":
        horizontal = [[0 for _ in range(size)] for _ in range(size + 1)]
        vertical = [[0 for _ in range(size + 1)] for _ in range(size)]
        boxes = [[0 for _ in range(size)] for _ in range(size)]
        return cls(size=size, horizontal=horizontal, vertical=vertical, boxes=boxes)

    def play_move(self, x: int, y: int, direction: str) -> Dict[str, Any]:
        if self.done:
            return {"valid": False, "message": "Game is over."}

        direction = direction.lower().strip()
        if direction not in ("r", "d"):
            return {"valid": False, "message": "Direction must be 'r' or 'd'."}

        if direction == "r":
            if not (0 <= x < self.size and 0 <= y <= self.size):
                return {"valid": False, "message": "Out of bounds for a horizontal edge."}
            if self.horizontal[y][x] != 0:
                return {"valid": False, "message": "Edge already taken."}
            self.horizontal[y][x] = self.current_player
            completed = self._complete_boxes_for_horizontal(x, y)
        else:
            if not (0 <= x <= self.size and 0 <= y < self.size):
                return {"valid": False, "message": "Out of bounds for a vertical edge."}
            if self.vertical[y][x] != 0:
                return {"valid": False, "message": "Edge already taken."}
            self.vertical[y][x] = self.current_player
            completed = self._complete_boxes_for_vertical(x, y)

        if completed:
            self.scores[self.current_player] += completed
        else:
            self.current_player = 2 if self.current_player == 1 else 1

        self.done = self._check_done()
        return {
            "valid": True,
            "completed": completed,
            "currentPlayer": self.current_player,
            "done": self.done,
        }

    def _complete_boxes_for_horizontal(self, x: int, y: int) -> int:
        completed = 0
        if y > 0:
            completed += self._claim_box(y - 1, x)
        if y < self.size:
            completed += self._claim_box(y, x)
        return completed

    def _complete_boxes_for_vertical(self, x: int, y: int) -> int:
        completed = 0
        if x > 0:
            completed += self._claim_box(y, x - 1)
        if x < self.size:
            completed += self._claim_box(y, x)
        return completed

    def _claim_box(self, row: int, col: int) -> int:
        if self.boxes[row][col] != 0:
            return 0
        if (
            self.horizontal[row][col] != 0
            and self.horizontal[row + 1][col] != 0
            and self.vertical[row][col] != 0
            and self.vertical[row][col + 1] != 0
        ):
            self.boxes[row][col] = self.current_player
            return 1
        return 0

    def _check_done(self) -> bool:
        for row in self.horizontal:
            for edge in row:
                if edge == 0:
                    return False
        for row in self.vertical:
            for edge in row:
                if edge == 0:
                    return False
        return True

    def render_ascii(self) -> str:
        lines = []
        header = "   " + "".join(f"{i:3}" for i in range(self.size + 1))
        lines.append(header)
        for y in range(self.size + 1):
            line = f"{y:2} "
            for x in range(self.size):
                line += "+"
                line += "--" if self.horizontal[y][x] else "  "
            line += "+"
            lines.append(line)
            if y < self.size:
                line = "   "
                for x in range(self.size + 1):
                    line += "|" if self.vertical[y][x] else " "
                    line += "  "
                lines.append(line)
        return "\n".join(lines)

    def serialize(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "horizontal": self.horizontal,
            "vertical": self.vertical,
            "boxes": self.boxes,
            "currentPlayer": self.current_player,
            "scores": {"1": self.scores[1], "2": self.scores[2]},
            "done": self.done,
        }
