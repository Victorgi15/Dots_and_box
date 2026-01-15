import copy
import math
import random
from typing import Callable, Dict, List, Optional, Protocol, Tuple


Move = Tuple[int, int, str]


class DotsBoxesState(Protocol):
    size: int
    horizontal: List[List[int]]
    vertical: List[List[int]]
    boxes: List[List[int]]
    scores: List[int]
    current_player: int
    done: bool

    def play_move(self, x: int, y: int, direction: str) -> Dict[str, object]:
        ...


PolicyValueFn = Callable[[DotsBoxesState], Tuple[object, float]]


def default_policy_value(_: DotsBoxesState) -> Tuple[object, float]:
    return {}, 0.0


def _default_simulations(size: int) -> int:
    if size <= 3:
        return 700
    if size == 4:
        return 500
    if size == 5:
        return 360
    if size == 6:
        return 260
    if size == 7:
        return 190
    return 150


def _adjacent_boxes(move: Move, size: int) -> List[Tuple[int, int]]:
    x, y, direction = move
    boxes: List[Tuple[int, int]] = []
    if direction == "r":
        if y > 0:
            boxes.append((y - 1, x))
        if y < size:
            boxes.append((y, x))
    else:
        if x > 0:
            boxes.append((y, x - 1))
        if x < size:
            boxes.append((y, x))
    return boxes


def _box_edge_count(state: DotsBoxesState, row: int, col: int) -> int:
    count = 0
    if state.horizontal[row][col] != 0:
        count += 1
    if state.horizontal[row + 1][col] != 0:
        count += 1
    if state.vertical[row][col] != 0:
        count += 1
    if state.vertical[row][col + 1] != 0:
        count += 1
    return count


def _box_edge_counts(state: DotsBoxesState) -> List[List[int]]:
    counts: List[List[int]] = []
    for row in range(state.size):
        row_counts = []
        for col in range(state.size):
            row_counts.append(_box_edge_count(state, row, col))
        counts.append(row_counts)
    return counts


def _two_edge_components(
    state: DotsBoxesState, edge_counts: List[List[int]]
) -> Tuple[List[List[int]], Dict[int, int]]:
    size = state.size
    comp_id = [[-1 for _ in range(size)] for _ in range(size)]
    comp_sizes: Dict[int, int] = {}
    current_id = 0

    for row in range(size):
        for col in range(size):
            if edge_counts[row][col] != 2 or comp_id[row][col] != -1:
                continue
            stack = [(row, col)]
            comp_id[row][col] = current_id
            count = 0
            while stack:
                r, c = stack.pop()
                count += 1
                if r > 0 and edge_counts[r - 1][c] == 2 and state.horizontal[r][c] == 0:
                    if comp_id[r - 1][c] == -1:
                        comp_id[r - 1][c] = current_id
                        stack.append((r - 1, c))
                if r + 1 < size and edge_counts[r + 1][c] == 2 and state.horizontal[r + 1][c] == 0:
                    if comp_id[r + 1][c] == -1:
                        comp_id[r + 1][c] = current_id
                        stack.append((r + 1, c))
                if c > 0 and edge_counts[r][c - 1] == 2 and state.vertical[r][c] == 0:
                    if comp_id[r][c - 1] == -1:
                        comp_id[r][c - 1] = current_id
                        stack.append((r, c - 1))
                if c + 1 < size and edge_counts[r][c + 1] == 2 and state.vertical[r][c + 1] == 0:
                    if comp_id[r][c + 1] == -1:
                        comp_id[r][c + 1] = current_id
                        stack.append((r, c + 1))
            comp_sizes[current_id] = count
            current_id += 1

    return comp_id, comp_sizes


def _move_effects(
    state: DotsBoxesState, move: Move, edge_counts: Optional[List[List[int]]] = None
) -> Tuple[int, int]:
    completed = 0
    risk = 0
    for row, col in _adjacent_boxes(move, state.size):
        edge_count = (
            edge_counts[row][col] if edge_counts is not None else _box_edge_count(state, row, col)
        )
        if edge_count == 3:
            completed += 1
        elif edge_count == 2:
            risk += 1
    return completed, risk


def _count_threat_boxes(state: DotsBoxesState) -> int:
    threats = 0
    for row in range(state.size):
        for col in range(state.size):
            if state.boxes[row][col] != 0:
                continue
            if _box_edge_count(state, row, col) == 3:
                threats += 1
    return threats


def _heuristic_priors(state: DotsBoxesState, moves: List[Move]) -> Dict[Move, float]:
    edge_counts = _box_edge_counts(state)
    comp_id, comp_sizes = _two_edge_components(state, edge_counts)
    existing_threats = _count_threat_boxes(state)
    move_info = []
    has_capture = False
    has_safe = False
    for move in moves:
        completed, risk = _move_effects(state, move, edge_counts)
        if completed:
            has_capture = True
        if completed == 0 and risk == 0:
            has_safe = True
        components = set()
        for row, col in _adjacent_boxes(move, state.size):
            if edge_counts[row][col] == 2:
                cid = comp_id[row][col]
                if cid != -1:
                    components.add(cid)
        chain_len = sum(comp_sizes[cid] for cid in components)
        move_info.append((move, completed, risk, chain_len))
    priors: Dict[Move, float] = {}
    for move, completed, risk, chain_len in move_info:
        if has_capture and completed == 0:
            priors[move] = 0.02
            continue
        switch_turn = completed == 0
        risk_weight = 2.2 if switch_turn else 0.3
        pass_penalty = 0.0
        if switch_turn and existing_threats:
            pass_penalty = existing_threats * 1.4
        safe_bonus = 0.6 if switch_turn and risk == 0 else 0.0
        close_bonus = completed * 9.0
        chain_bonus = 1.4 if completed > 0 else 0.0
        chain_penalty = chain_len * (1.1 if has_safe else 0.6)
        score = (
            1.0
            + close_bonus
            + chain_bonus
            - risk * risk_weight
            - pass_penalty
            - chain_penalty
            + safe_bonus
        )
        if has_safe and risk > 0:
            score *= 0.35
        priors[move] = max(0.05, score)
    return priors


def _heuristic_value(state: DotsBoxesState, moves: List[Move]) -> float:
    current = state.current_player
    other = 2 if current == 1 else 1
    max_score = state.size * state.size
    score_diff = state.scores[current] - state.scores[other]
    threats = _count_threat_boxes(state)
    best_gain = 0
    min_risk = None
    for move in moves:
        gain, risk = _move_effects(state, move)
        best_gain = max(best_gain, gain)
        min_risk = risk if min_risk is None else min(min_risk, risk)
    if min_risk is None:
        min_risk = 0
    value = (score_diff + 0.7 * best_gain - 0.35 * min_risk + 0.2 * threats) / max_score
    return max(-1.0, min(1.0, value))


def _rollout_value(state: DotsBoxesState) -> float:
    sim = _clone_state(state)
    start_player = sim.current_player
    max_moves = (sim.size + 1) * sim.size * 2
    rollout_limit = min(max_moves, 18 + sim.size * 9)
    for _ in range(rollout_limit):
        if sim.done:
            break
        moves = legal_moves(sim)
        if not moves:
            break
        edge_counts = _box_edge_counts(sim)
        comp_id, comp_sizes = _two_edge_components(sim, edge_counts)
        scored: List[Tuple[Move, int, int, int]] = []
        best_gain = 0
        for move in moves:
            gain, risk = _move_effects(sim, move, edge_counts)
            components = set()
            for row, col in _adjacent_boxes(move, sim.size):
                if edge_counts[row][col] == 2:
                    cid = comp_id[row][col]
                    if cid != -1:
                        components.add(cid)
            chain_len = sum(comp_sizes[cid] for cid in components)
            scored.append((move, gain, risk, chain_len))
            best_gain = max(best_gain, gain)
        if best_gain > 0:
            candidates = [move for move, gain, _, _ in scored if gain == best_gain]
        else:
            safe = [move for move, _, risk, _ in scored if risk == 0]
            if safe:
                candidates = safe
            else:
                min_key = min((risk, chain_len) for _, _, risk, chain_len in scored)
                candidates = [
                    move
                    for move, _, risk, chain_len in scored
                    if (risk, chain_len) == min_key
                ]
        move = random.choice(candidates)
        result = sim.play_move(move[0], move[1], move[2])
        if not result.get("valid", False):
            break
    other = 2 if start_player == 1 else 1
    max_score = sim.size * sim.size
    if max_score == 0:
        return 0.0
    value = (sim.scores[start_player] - sim.scores[other]) / max_score
    return max(-1.0, min(1.0, value))


def heuristic_policy_value(state: DotsBoxesState) -> Tuple[object, float]:
    moves = legal_moves(state)
    priors = _heuristic_priors(state, moves)
    rollout = _rollout_value(state)
    heuristic = _heuristic_value(state, moves)
    value = 0.7 * rollout + 0.3 * heuristic
    return priors, value


def legal_moves(state: DotsBoxesState) -> List[Move]:
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


def _state_key(state: DotsBoxesState) -> Tuple[object, ...]:
    return (
        state.current_player,
        tuple(tuple(row) for row in state.horizontal),
        tuple(tuple(row) for row in state.vertical),
        tuple(tuple(row) for row in state.boxes),
    )


def _capture_moves(state: DotsBoxesState) -> List[Tuple[Move, int]]:
    edge_counts = _box_edge_counts(state)
    captures: List[Tuple[Move, int]] = []
    for move in legal_moves(state):
        completed, _ = _move_effects(state, move, edge_counts)
        if completed > 0:
            captures.append((move, completed))
    return captures


def _max_capture_chain(
    state: DotsBoxesState, cache: Optional[Dict[Tuple[object, ...], int]] = None
) -> int:
    if cache is None:
        cache = {}
    key = _state_key(state)
    cached = cache.get(key)
    if cached is not None:
        return cached
    captures = _capture_moves(state)
    if not captures:
        cache[key] = 0
        return 0
    player = state.current_player
    best = 0
    for move, completed in captures:
        sim = _clone_state(state)
        result = sim.play_move(move[0], move[1], move[2])
        if not result.get("valid", False):
            continue
        gained = int(result.get("completed", completed))
        if sim.current_player != player:
            total = gained
        else:
            total = gained + _max_capture_chain(sim, cache)
        if total > best:
            best = total
    cache[key] = best
    return best


def _shallow_move_score(state: DotsBoxesState, move: Move) -> float:
    sim = _clone_state(state)
    result = sim.play_move(move[0], move[1], move[2])
    if not result.get("valid", False):
        return -float("inf")
    completed = int(result.get("completed", 0))
    if completed > 0:
        chain_bonus = _max_capture_chain(sim)
        return float(completed + chain_bonus)
    if sim.current_player == state.current_player:
        return 0.0
    opponent_chain = _max_capture_chain(sim)
    return -float(opponent_chain)


def terminal_value(state: DotsBoxesState) -> Optional[float]:
    if not state.done:
        return None
    s1 = state.scores[1]
    s2 = state.scores[2]
    if s1 == s2:
        return 0.0
    winner = 1 if s1 > s2 else 2
    return 1.0 if state.current_player == winner else -1.0


def _normalize_move(move: object) -> Optional[Move]:
    if isinstance(move, tuple) or isinstance(move, list):
        if len(move) == 3:
            x, y, direction = move
            return int(x), int(y), str(direction)
        return None
    if isinstance(move, dict):
        if {"x", "y", "direction"} <= move.keys():
            return int(move["x"]), int(move["y"]), str(move["direction"])
        return None
    return None


def _policy_priors(policy: object, moves: List[Move]) -> Dict[Move, float]:
    priors: Dict[Move, float] = {move: 0.0 for move in moves}
    if isinstance(policy, dict):
        for move_key, prob in policy.items():
            move = _normalize_move(move_key)
            if move in priors:
                priors[move] = float(prob)
        return priors
    if isinstance(policy, list):
        for item in policy:
            if isinstance(item, tuple) or isinstance(item, list):
                if len(item) == 2:
                    move = _normalize_move(item[0])
                    if move in priors:
                        priors[move] = float(item[1])
                    continue
            if isinstance(item, dict):
                move = _normalize_move(item)
                if move in priors:
                    priors[move] = float(item.get("prob", 0.0))
        return priors
    return priors


def _normalize_priors(priors: Dict[Move, float]) -> Dict[Move, float]:
    total = 0.0
    for value in priors.values():
        if value > 0:
            total += value
    if total <= 0:
        if not priors:
            return {}
        uniform = 1.0 / len(priors)
        return {move: uniform for move in priors}
    return {move: max(prob, 0.0) / total for move, prob in priors.items()}


def _clone_state(state: DotsBoxesState) -> DotsBoxesState:
    return copy.deepcopy(state)


def _apply_move(state: DotsBoxesState, move: Move) -> DotsBoxesState:
    next_state = _clone_state(state)
    result = next_state.play_move(move[0], move[1], move[2])
    if not result.get("valid", False):
        raise ValueError(f"Invalid move applied in MCTS: {result}")
    return next_state


class TreeNode:
    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.children: Dict[Move, TreeNode] = {}
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


class MCTS:
    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        n_simulations: int = 200,
        c_puct: float = 1.4,
    ) -> None:
        self.policy_value_fn = policy_value_fn
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.root = TreeNode(1.0)

    def _simulate(self, root_state: DotsBoxesState) -> None:
        state = _clone_state(root_state)
        node = self.root
        path: List[TreeNode] = [node]
        players: List[int] = [state.current_player]

        while node.children:
            move, node = node.select(self.c_puct)
            state = _apply_move(state, move)
            path.append(node)
            players.append(state.current_player)

        value = terminal_value(state)
        if value is None:
            policy, value = self.policy_value_fn(state)
            if value is None:
                value = 0.0
            priors = _normalize_priors(_policy_priors(policy, legal_moves(state)))
            node.expand(priors)
        value = float(max(-1.0, min(1.0, value)))

        leaf_player = players[-1]
        for node, player in zip(path, players):
            node.n_visits += 1
            node.value_sum += value if player == leaf_player else -value

    def get_move_distribution(
        self, state: DotsBoxesState, temperature: float = 1.0
    ) -> List[Dict[str, object]]:
        if state.size < 2 or state.size > 8:
            raise ValueError("MCTS supports board sizes 2 to 8.")
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


def choose_move_mcts(
    state: DotsBoxesState,
    n_simulations: Optional[int] = None,
    c_puct: float = 1.4,
    temperature: float = 0.0,
) -> Optional[Dict[str, object]]:
    if n_simulations is None:
        n_simulations = _default_simulations(state.size)
    mcts = MCTS(heuristic_policy_value, n_simulations=n_simulations, c_puct=c_puct)
    distribution = mcts.get_move_distribution(state, temperature=temperature)
    if not distribution:
        return None
    best = max(distribution, key=lambda item: item["prob"])
    return {"x": best["x"], "y": best["y"], "direction": best["direction"]}


def choose_move_shallow(state: DotsBoxesState) -> Optional[Dict[str, object]]:
    if state.done:
        return None
    moves = legal_moves(state)
    if not moves:
        return None
    scored: List[Tuple[float, Move]] = []
    for move in moves:
        scored.append((_shallow_move_score(state, move), move))
    best_score = max(score for score, _ in scored)
    candidates = [move for score, move in scored if score == best_score]
    if not candidates:
        return None
    move = random.choice(candidates)
    return {"x": move[0], "y": move[1], "direction": move[2]}


def choose_move(
    state: DotsBoxesState,
    n_simulations: Optional[int] = None,
    c_puct: float = 1.4,
    temperature: float = 0.0,
    mode: str = "shallow",
) -> Optional[Dict[str, object]]:
    if mode.strip().lower() == "mcts":
        return choose_move_mcts(
            state,
            n_simulations=n_simulations,
            c_puct=c_puct,
            temperature=temperature,
        )
    return choose_move_shallow(state)
