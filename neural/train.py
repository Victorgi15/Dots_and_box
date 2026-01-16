from typing import Iterable, List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    torch = None
    F = None
    _torch_import_error = exc

from .checkpoint import save_checkpoint
from .encoder import policy_size
from .network import DotsBoxesNet
from .replay_buffer import ReplayBuffer


Sample = Tuple[List[List[List[float]]], List[float], float]


def _to_tensor(batch: Iterable[Sample], device):
    states, policies, values = zip(*batch)
    state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    policy_tensor = torch.tensor(policies, dtype=torch.float32, device=device)
    value_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return state_tensor, policy_tensor, value_tensor


def train_on_buffer(
    buffer: ReplayBuffer,
    board_size: int,
    batch_size: int = 64,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
    model: Optional[DotsBoxesNet] = None,
) -> DotsBoxesNet:
    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("torch is required for training") from _torch_import_error
    if model is None:
        model = DotsBoxesNet(board_size)
    elif getattr(model, "board_size", board_size) != board_size:
        raise ValueError("Model board size mismatch.")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        if len(buffer) == 0:
            break
        steps = max(1, len(buffer) // batch_size)
        for _ in range(steps):
            batch = buffer.sample(batch_size)
            states, target_pi, target_v = _to_tensor(batch, device)
            logits, values = model(states)
            if logits.shape[1] != policy_size(board_size):
                raise ValueError("Policy size mismatch in training.")
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(values, target_v)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def save_model_checkpoint(
    model: DotsBoxesNet, path: str, board_size: int, step: int = 0
) -> None:
    save_checkpoint(model, path, board_size=board_size, step=step)
