import math
import os
from pathlib import Path
from typing import Dict, Optional
import json

import torch


def get_device(require_mps: bool = True) -> torch.device:
    """
    Return MPS device or raise. This project is Mac-only; no CUDA/CPU fallback.
    """
    if require_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("MPS not available. Ensure you are on Apple Silicon with PyTorch built with MPS (install nightly if needed).")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, warmup: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    if warmup < 1:
        raise ValueError("warmup must be >= 1")
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], path: Path) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))


def rotate_checkpoints(save_dir: Path, prefix: str, keep: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(save_dir.glob(f"{prefix}*.pt"), key=os.path.getmtime, reverse=True)
    for extra in ckpts[keep:]:
        extra.unlink()


def append_log(log_path: Path, record: Dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(payload + "\n")
