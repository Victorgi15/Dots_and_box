# Bots

This folder hosts bots for Dots and Boxes.

- `random_bot.py`: picks a legal move uniformly at random.
- `mcts_puct.py`: PUCT-based MCTS limited to 8x8 and driven by a
  `policy_value(state)` callback that returns `(policy, value)`. The default
  bot wrapper uses heuristic priors plus rollouts for stronger play.
