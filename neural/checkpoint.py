import os
import time
from typing import Any, Dict, Optional, Tuple

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    torch = None
    _torch_import_error = exc


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_checkpoint_paths(root: str, size: int, step: int) -> Tuple[str, str]:
    base = os.path.join(root, "models", f"size_{size}")
    checkpoint = os.path.join(base, f"checkpoint_{step:06d}.pt")
    latest = os.path.join(base, "latest.pt")
    return checkpoint, latest


def save_checkpoint(
    model,
    path: str,
    board_size: int,
    step: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("torch is required to save checkpoints") from _torch_import_error
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "format_version": 1,
        "board_size": board_size,
        "step": step,
        "created_at": _timestamp(),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if torch is None:  # pragma: no cover - optional dependency
        raise ImportError("torch is required to load checkpoints") from _torch_import_error
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload
    return payload, {}
