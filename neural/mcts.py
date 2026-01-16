import copy
import math
import random
from typing import Callable, Dict, List, Optional, Tuple

from .encoder import move_to_index, policy_size


Move = Tuple[int, int, str]
PolicyValueFn = Callable[[object], Tuple[List[float], float]]


def legal_moves(state) -> List[Move]:
    moves: List[Move] = []
    for y in range(state.size + 1):
        for x in range(state.size):
            if state.horizontal[y][x] == 0:
                moves.append((x, y, "r"))
    for y in range(state.size):
        for x in range(state.size + 1):
            if state.vertical[y][x] == 0:
                moves.append((x, y, "d"))
    return moves


def terminal_value(state) -> Optional[float]:
    if not state.done:
        return None
    max_score = state.size * state.size
    if max_score == 0:
        return 0.0
    current = state.current_player
    other = 2 if current == 1 else 1
    value = (state.scores[current] - state.scores[other]) / max_score
    return max(-1.0, min(1.0, value))


def _policy_priors(
    logits: List[float], moves: List[Move], size: int
) -> Dict[Move, float]:
    if len(logits) != policy_size(size):
        raise ValueError("Policy logits size does not match board size.")
    indices = [(move, move_to_index(move, size)) for move in moves]
    if not indices:
        return {}
    max_logit = max(logits[index] for _, index in indices)
    exp_values: Dict[Move, float] = {}
    total = 0.0
    for move, index in indices:
        value = math.exp(logits[index] - max_logit)
        exp_values[move] = value
        total += value
    if total <= 0:
        uniform = 1.0 / len(indices)
        return {move: uniform for move, _ in indices}
    return {move: value / total for move, value in exp_values.items()}


class TreeNode:
    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.children: Dict[Move, "TreeNode"] = {}
        self.n_visits = 0
        self.value_sum = 0.0

    def q_value(self) -> float:
        if self.n_visits == 0:
            return 0.0
        return self.value_sum / self.n_visits

    def expand(self, priors: Dict[Move, float]) -> None:
        for move, prior in priors.items():
            if move not in self.children:
                self.children[move] = TreeNode(prior)

    def select(self, c_puct: float) -> Tuple[Move, "TreeNode"]:
        best_move = None
        best_child = None
        best_score = -float("inf")
        parent_visits = max(1, self.n_visits)
        sqrt_visits = math.sqrt(parent_visits)
        for move, child in self.children.items():
            u_score = c_puct * child.prior * sqrt_visits / (1 + child.n_visits)
            score = child.q_value() + u_score
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        if best_move is None or best_child is None:
            raise ValueError("No child nodes available for selection.")
        return best_move, best_child


class NeuralMCTS:
    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        n_simulations: int = 200,
        c_puct: float = 1.4,
        dirichlet_alpha: Optional[float] = None,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        self.policy_value_fn = policy_value_fn
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root = TreeNode(1.0)

    def _apply_dirichlet_noise(self, priors: Dict[Move, float]) -> Dict[Move, float]:
        if not priors or self.dirichlet_alpha is None:
            return priors
        moves = list(priors.keys())
        alpha = self.dirichlet_alpha
        noise = [random.gammavariate(alpha, 1.0) for _ in moves]
        total = sum(noise)
        if total <= 0:
            return priors
        mixed: Dict[Move, float] = {}
        for move, n in zip(moves, noise):
            mixed_prior = (1 - self.dirichlet_epsilon) * priors[move] + self.dirichlet_epsilon * (
                n / total
            )
            mixed[move] = mixed_prior
        return mixed

    def _simulate(self, root_state) -> None:
        state = copy.deepcopy(root_state)
        node = self.root
        path: List[TreeNode] = [node]
        players: List[int] = [state.current_player]

        while node.children:
            move, node = node.select(self.c_puct)
            result = state.play_move(move[0], move[1], move[2])
            if not result.get("valid", False):
                break
            path.append(node)
            players.append(state.current_player)

        value = terminal_value(state)
        if value is None:
            logits, value = self.policy_value_fn(state)
            moves = legal_moves(state)
            priors = _policy_priors(logits, moves, state.size)
            if path == [self.root]:
                priors = self._apply_dirichlet_noise(priors)
            node.expand(priors)
        value = float(max(-1.0, min(1.0, value)))

        leaf_player = players[-1]
        for node, player in zip(path, players):
            node.n_visits += 1
            node.value_sum += value if player == leaf_player else -value

    def get_move_distribution(
        self, state, temperature: float = 1.0
    ) -> List[Dict[str, object]]:
        if state.done:
            return []
        moves = legal_moves(state)
        if not moves:
            return []

        self.root = TreeNode(1.0)
        for _ in range(self.n_simulations):
            self._simulate(state)

        visit_counts = []
        for move in moves:
            child = self.root.children.get(move)
            visit_counts.append(child.n_visits if child else 0)

        if temperature <= 0:
            best_index = max(range(len(visit_counts)), key=visit_counts.__getitem__)
            probs = [0.0 for _ in moves]
            probs[best_index] = 1.0
        else:
            scaled = [count ** (1.0 / temperature) for count in visit_counts]
            total = sum(scaled)
            if total == 0:
                uniform = 1.0 / len(moves)
                probs = [uniform for _ in moves]
            else:
                probs = [value / total for value in scaled]

        distribution = []
        for move, prob in zip(moves, probs):
            distribution.append(
                {"x": move[0], "y": move[1], "direction": move[2], "prob": prob}
            )
        return distribution
