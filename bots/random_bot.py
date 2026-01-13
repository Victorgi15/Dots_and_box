import random
from typing import Dict, List, Optional, Protocol


class DotsBoxesState(Protocol):
    size: int
    horizontal: List[List[int]]
    vertical: List[List[int]]


Move = Dict[str, object]


def available_moves(state: DotsBoxesState) -> List[Move]:
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


def choose_move(state: DotsBoxesState) -> Optional[Move]:
    moves = available_moves(state)
    if not moves:
        return None
    return random.choice(moves)
