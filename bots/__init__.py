"""Bots for dots and boxes."""

from .chatgpt_bot import choose_move as choose_chatgpt_move
from .mcts_puct import MCTS, choose_move as choose_mcts_move
from .neural_mcts_bot import choose_move as choose_neural_move
from .random_bot import choose_move as choose_random_move

__all__ = [
    "MCTS",
    "choose_chatgpt_move",
    "choose_mcts_move",
    "choose_neural_move",
    "choose_random_move",
]
