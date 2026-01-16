from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    _torch_import_error = exc

from .checkpoint import load_checkpoint
from .encoder import encode_state, policy_size


class DotsBoxesNet(nn.Module):
    def __init__(self, board_size: int, in_channels: int = 7) -> None:
        super().__init__()
        channels = 64
        self.board_size = board_size
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        grid = board_size + 1
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * grid * grid, policy_size(board_size)),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(grid * grid, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value


class NeuralPolicy:
    def __init__(self, model_path: str, board_size: int, device: Optional[str] = None) -> None:
        if torch is None:  # pragma: no cover - optional dependency
            raise ImportError("torch is required for NeuralPolicy") from _torch_import_error
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DotsBoxesNet(board_size).to(self.device)
        state_dict, meta = load_checkpoint(model_path, device=self.device)
        if meta.get("board_size") and meta["board_size"] != board_size:
            raise ValueError("Checkpoint board size does not match state size.")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, state):
        if torch is None:  # pragma: no cover - optional dependency
            raise ImportError("torch is required for NeuralPolicy") from _torch_import_error
        with torch.no_grad():
            encoded = encode_state(state, device=self.device)
            policy_logits, value = self.model(encoded)
        logits = policy_logits.squeeze(0).detach().cpu().tolist()
        return logits, float(value.item())


class ModelPolicy:
    def __init__(self, model: DotsBoxesNet, device: Optional[str] = None) -> None:
        if torch is None:  # pragma: no cover - optional dependency
            raise ImportError("torch is required for ModelPolicy") from _torch_import_error
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

    def predict(self, state):
        if torch is None:  # pragma: no cover - optional dependency
            raise ImportError("torch is required for ModelPolicy") from _torch_import_error
        with torch.no_grad():
            encoded = encode_state(state, device=self.device)
            policy_logits, value = self.model(encoded)
        logits = policy_logits.squeeze(0).detach().cpu().tolist()
        return logits, float(value.item())
