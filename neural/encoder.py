from typing import List, Tuple

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    torch = None
    _torch_import_error = exc


Move = Tuple[int, int, str]


def policy_size(size: int) -> int:
    return 2 * size * (size + 1)


def move_to_index(move: Move, size: int) -> int:
    x, y, direction = move
    if direction == "r":
        return y * size + x
    return (size + 1) * size + y * (size + 1) + x


def index_to_move(index: int, size: int) -> Move:
    horiz = (size + 1) * size
    if index < horiz:
        y, x = divmod(index, size)
        return x, y, "r"
    offset = index - horiz
    y, x = divmod(offset, size + 1)
    return x, y, "d"


def state_planes(state) -> List[List[List[float]]]:
    size = state.size
    grid = size + 1
    zeros = [[0.0 for _ in range(grid)] for _ in range(grid)]
    h_self = [row[:] for row in zeros]
    h_opp = [row[:] for row in zeros]
    v_self = [row[:] for row in zeros]
    v_opp = [row[:] for row in zeros]
    b_self = [row[:] for row in zeros]
    b_opp = [row[:] for row in zeros]

    current = state.current_player
    other = 2 if current == 1 else 1

    for y in range(size + 1):
        for x in range(size):
            owner = state.horizontal[y][x]
            if owner == current:
                h_self[y][x] = 1.0
            elif owner == other:
                h_opp[y][x] = 1.0

    for y in range(size):
        for x in range(size + 1):
            owner = state.vertical[y][x]
            if owner == current:
                v_self[y][x] = 1.0
            elif owner == other:
                v_opp[y][x] = 1.0

    for y in range(size):
        for x in range(size):
            owner = state.boxes[y][x]
            if owner == current:
                b_self[y][x] = 1.0
            elif owner == other:
                b_opp[y][x] = 1.0

    turn = [[1.0 for _ in range(grid)] for _ in range(grid)]
    return [h_self, h_opp, v_self, v_opp, b_self, b_opp, turn]


def encode_state(state, device=None):
    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for encode_state") from _torch_import_error
    planes = state_planes(state)
    tensor = torch.tensor(planes, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0)
