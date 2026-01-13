"""Bots for dots and boxes."""

from .mcts_puct import MCTS, choose_move as choose_mcts_move
from .random_bot import choose_move as choose_random_move

__all__ = ["MCTS", "choose_mcts_move", "choose_random_move"]
