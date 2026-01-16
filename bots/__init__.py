"""Bots for dots and boxes."""

from .chatgpt_bot import choose_move as choose_chatgpt_move
from .mcts_puct import MCTS, choose_move as choose_mcts_move
from .random_bot import choose_move as choose_random_move

try:
    from .neural_mcts_bot import choose_move as choose_neural_move
except Exception:  # pragma: no cover - optional dependency
    choose_neural_move = None

__all__ = ["MCTS", "choose_chatgpt_move", "choose_mcts_move", "choose_random_move"]
if choose_neural_move is not None:
    __all__.append("choose_neural_move")
