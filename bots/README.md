# Bots

This folder hosts bots for Dots and Boxes.

- `random_bot.py`: picks a legal move uniformly at random.
- `chatgpt_bot.py`: asks the ChatGPT API to choose a move from the legal set.
- `neural_mcts_bot.py`: AlphaZero-style bot (MCTS + neural policy/value).
- `mcts_puct.py`: PUCT-based MCTS limited to 8x8 and driven by a
  `policy_value(state)` callback that returns `(policy, value)`. The default
  bot wrapper uses heuristic priors plus rollouts for stronger play.
