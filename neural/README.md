# Neural MCTS

This folder hosts the AlphaZero-style pipeline (policy/value network + MCTS).

## Runtime bot
- `NEURAL_MODEL_PATH` points to a `.pt` checkpoint (default: `models/size_N/latest.pt`).
- `NEURAL_MCTS_SIMULATIONS` and `NEURAL_MCTS_C_PUCT` tune search.

## CLI
From repo root:
```
python -m neural.cli run --size 5 --games 25 --epochs 1 --step 1
```
This generates self-play games, trains a network, and writes checkpoints to
`models/size_5/checkpoint_000001.pt` and `models/size_5/latest.pt`.

You can also split it:
```
python -m neural.cli self-play --size 5 --games 50 --out data/self_play.jsonl
python -m neural.cli train --size 5 --data data/self_play.jsonl --epochs 2 --step 2
```

## Checkpoint format
Checkpoints are `torch.save` dicts with:
- `format_version` (int)
- `board_size` (int)
- `step` (int)
- `created_at` (UTC string)
- `state_dict` (model weights)
- `extra` (free-form metadata)

## Training blocks
- `self_play.py` generates `(state_planes, pi, value)` samples.
- `replay_buffer.py` stores samples for training.
- `train.py` trains `DotsBoxesNet` on a buffer.
- `arena.py` evaluates two policies via MCTS matches.

This code expects PyTorch. Install it before running training or the neural bot.
