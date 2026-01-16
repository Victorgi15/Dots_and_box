from typing import Dict, Tuple

from backend.game_logic import GameState
from bots.mcts_puct import choose_move_mcts

from .mcts import NeuralMCTS, legal_moves


def _policy_value_fn(policy):
    def _fn(state):
        return policy.predict(state)

    return _fn


def _neural_best_move(policy, state, n_simulations: int, c_puct: float):
    mcts = NeuralMCTS(_policy_value_fn(policy), n_simulations=n_simulations, c_puct=c_puct)
    distribution = mcts.get_move_distribution(state, temperature=0.0)
    if not distribution:
        return None
    return max(distribution, key=lambda item: item["prob"])


def play_match(
    policy_a,
    policy_b,
    size: int,
    n_simulations: int = 200,
    c_puct: float = 1.4,
    policy_a_player: int = 1,
) -> Tuple[int, int]:
    game = GameState.new(size)
    other_player = 2 if policy_a_player == 1 else 1
    policies = {policy_a_player: policy_a, other_player: policy_b}

    while not game.done:
        if not legal_moves(game):
            break
        policy = policies[game.current_player]
        move = _neural_best_move(policy, game, n_simulations=n_simulations, c_puct=c_puct)
        if not move:
            break
        game.play_move(move["x"], move["y"], move["direction"])

    return game.scores[1], game.scores[2]


def evaluate_neural_policies(
    policy_a,
    policy_b,
    size: int,
    games: int = 20,
    n_simulations: int = 200,
    c_puct: float = 1.4,
) -> Dict[str, float]:
    wins_a = 0
    wins_b = 0
    draws = 0
    for idx in range(games):
        policy_a_player = 1 if idx % 2 == 0 else 2
        s1, s2 = play_match(
            policy_a,
            policy_b,
            size=size,
            n_simulations=n_simulations,
            c_puct=c_puct,
            policy_a_player=policy_a_player,
        )
        if s1 == s2:
            draws += 1
        else:
            winner = 1 if s1 > s2 else 2
            if winner == policy_a_player:
                wins_a += 1
            else:
                wins_b += 1
    total = max(1, games)
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": wins_a / total,
        "win_rate_b": wins_b / total,
        "draw_rate": draws / total,
    }


def evaluate_neural_vs_mcts(
    policy,
    size: int,
    games: int = 20,
    neural_simulations: int = 200,
    neural_c_puct: float = 1.4,
    mcts_simulations: int = 200,
    mcts_c_puct: float = 1.4,
) -> Dict[str, float]:
    wins = 0
    losses = 0
    draws = 0
    for idx in range(games):
        neural_player = 1 if idx % 2 == 0 else 2
        game = GameState.new(size)
        while not game.done:
            if not legal_moves(game):
                break
            if game.current_player == neural_player:
                move = _neural_best_move(
                    policy, game, n_simulations=neural_simulations, c_puct=neural_c_puct
                )
            else:
                move = choose_move_mcts(
                    game, n_simulations=mcts_simulations, c_puct=mcts_c_puct, temperature=0.0
                )
            if not move:
                break
            game.play_move(move["x"], move["y"], move["direction"])
        s1, s2 = game.scores[1], game.scores[2]
        if s1 == s2:
            draws += 1
        else:
            winner = 1 if s1 > s2 else 2
            if winner == neural_player:
                wins += 1
            else:
                losses += 1
    total = max(1, games)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
    }
