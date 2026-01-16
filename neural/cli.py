import argparse
import json
import os
import random
import shutil
from typing import Iterable, List, Tuple

from .checkpoint import default_checkpoint_paths, save_checkpoint
from .network import DotsBoxesNet, ModelPolicy, NeuralPolicy
from .replay_buffer import ReplayBuffer
from .self_play import self_play_game
from .train import train_on_buffer


Sample = Tuple[List[List[List[float]]], List[float], float]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_jsonl(path: str, samples: Iterable[Sample]) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for state_planes, pi, value in samples:
            handle.write(json.dumps({"state": state_planes, "pi": pi, "value": value}))
            handle.write("\n")


def _read_jsonl(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append((item["state"], item["pi"], float(item["value"])))
    return samples


def _load_policy(model_path: str, size: int, device: str):
    if model_path:
        return NeuralPolicy(model_path, board_size=size, device=device)
    model = DotsBoxesNet(size)
    return ModelPolicy(model, device=device)


def _cmd_self_play(args: argparse.Namespace) -> None:
    if args.seed is not None:
        _seed_everything(args.seed)
    policy = _load_policy(args.model, args.size, args.device)
    samples: List[Sample] = []
    for _ in range(args.games):
        samples.extend(
            self_play_game(
                policy,
                size=args.size,
                n_simulations=args.simulations,
                c_puct=args.c_puct,
                temperature=args.temperature,
                temperature_cutoff=args.temperature_cutoff,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_epsilon=args.dirichlet_epsilon,
            )
        )
    _write_jsonl(args.out, samples)
    print(f"Wrote {len(samples)} samples to {args.out}")


def _cmd_train(args: argparse.Namespace) -> None:
    if args.seed is not None:
        _seed_everything(args.seed)
    buffer = ReplayBuffer(capacity=args.buffer_size)
    samples = _read_jsonl(args.data)
    buffer.extend(samples)
    model = train_on_buffer(
        buffer,
        board_size=args.size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    checkpoint_path, latest_path = default_checkpoint_paths(args.root, args.size, args.step)
    save_checkpoint(model, checkpoint_path, board_size=args.size, step=args.step)
    shutil.copy2(checkpoint_path, latest_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def _cmd_run(args: argparse.Namespace) -> None:
    if args.seed is not None:
        _seed_everything(args.seed)
    policy = _load_policy(args.model, args.size, args.device)
    buffer = ReplayBuffer(capacity=args.buffer_size)
    samples: List[Sample] = []
    for _ in range(args.games):
        samples.extend(
            self_play_game(
                policy,
                size=args.size,
                n_simulations=args.simulations,
                c_puct=args.c_puct,
                temperature=args.temperature,
                temperature_cutoff=args.temperature_cutoff,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_epsilon=args.dirichlet_epsilon,
            )
        )
    buffer.extend(samples)
    if args.save_data:
        _write_jsonl(args.save_data, samples)
        print(f"Wrote {len(samples)} samples to {args.save_data}")
    model = train_on_buffer(
        buffer,
        board_size=args.size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    checkpoint_path, latest_path = default_checkpoint_paths(args.root, args.size, args.step)
    save_checkpoint(model, checkpoint_path, board_size=args.size, step=args.step)
    shutil.copy2(checkpoint_path, latest_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=120)
    parser.add_argument("--c-puct", type=float, default=1.4, dest="c_puct")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-cutoff", type=int, default=12)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--model", type=str, default="")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--root", type=str, default=os.getcwd())


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural MCTS CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    self_play = sub.add_parser("self-play", help="Generate self-play data.")
    _add_common_args(self_play)
    self_play.add_argument("--games", type=int, default=10)
    self_play.add_argument("--out", type=str, default="data/self_play.jsonl")
    self_play.set_defaults(func=_cmd_self_play)

    train = sub.add_parser("train", help="Train on existing self-play data.")
    _add_common_args(train)
    _add_train_args(train)
    train.add_argument("--data", type=str, required=True)
    train.set_defaults(func=_cmd_train)

    run = sub.add_parser("run", help="Self-play + training in one command.")
    _add_common_args(run)
    _add_train_args(run)
    run.add_argument("--games", type=int, default=10)
    run.add_argument("--save-data", type=str, default="")
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
