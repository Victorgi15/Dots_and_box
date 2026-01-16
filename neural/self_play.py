import random
from typing import List, Tuple

from backend.game_logic import GameState

from .encoder import move_to_index, policy_size, state_planes
from .mcts import NeuralMCTS, legal_moves


Sample = Tuple[List[List[List[float]]], List[float], float]


def _select_move(distribution):
    choices = [(item["prob"], (item["x"], item["y"], item["direction"])) for item in distribution]
    total = sum(prob for prob, _ in choices)
    if total <= 0:
        return random.choice([move for _, move in choices])
    pick = random.random() * total
    cumulative = 0.0
    for prob, move in choices:
        cumulative += prob
        if pick <= cumulative:
            return move
    return choices[-1][1]


def _policy_value_fn(policy):
    def _fn(state):
        return policy.predict(state)

    return _fn


def self_play_game(
    policy,
    size: int,
    n_simulations: int = 200,
    c_puct: float = 1.4,
    temperature: float = 1.0,
    temperature_cutoff: int = 12,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> List[Sample]:
    game = GameState.new(size)
    policy_fn = _policy_value_fn(policy)
    samples: List[Tuple[List[List[List[float]]], List[float], int]] = []
    move_count = 0

    while not game.done:
        if not legal_moves(game):
            break
        temp = temperature if move_count < temperature_cutoff else 0.0
        mcts = NeuralMCTS(
            policy_fn,
            n_simulations=n_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )
        distribution = mcts.get_move_distribution(game, temperature=temp)
        pi = [0.0 for _ in range(policy_size(size))]
        for item in distribution:
            idx = move_to_index((item["x"], item["y"], item["direction"]), size)
            pi[idx] = item["prob"]
        samples.append((state_planes(game), pi, game.current_player))
        move = _select_move(distribution)
        game.play_move(move[0], move[1], move[2])
        move_count += 1

    max_score = size * size
    if max_score == 0:
        max_score = 1
    labeled: List[Sample] = []
    for planes, pi, player in samples:
        other = 2 if player == 1 else 1
        value = (game.scores[player] - game.scores[other]) / max_score
        labeled.append((planes, pi, value))
    return labeled
