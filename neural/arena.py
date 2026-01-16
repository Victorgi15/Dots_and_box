from typing import Callable, Tuple

from backend.game_logic import GameState

from .mcts import NeuralMCTS, legal_moves


def _policy_value_fn(policy):
    def _fn(state):
        return policy.predict(state)

    return _fn


def play_match(
    policy_a,
    policy_b,
    size: int,
    n_simulations: int = 200,
    c_puct: float = 1.4,
) -> Tuple[int, int]:
    game = GameState.new(size)
    policies = {1: policy_a, 2: policy_b}

    while not game.done:
        if not legal_moves(game):
            break
        policy = policies[game.current_player]
        mcts = NeuralMCTS(_policy_value_fn(policy), n_simulations=n_simulations, c_puct=c_puct)
        distribution = mcts.get_move_distribution(game, temperature=0.0)
        move = max(distribution, key=lambda item: item["prob"])
        game.play_move(move["x"], move["y"], move["direction"])

    return game.scores[1], game.scores[2]
