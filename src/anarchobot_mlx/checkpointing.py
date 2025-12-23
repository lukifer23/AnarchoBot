from pathlib import Path
import pickle
from typing import Optional

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map


def _to_mx_array(x):
    if isinstance(x, np.ndarray):
        if x.dtype == np.float16:
            return mx.array(x, dtype=mx.bfloat16)
        if x.dtype == np.float32:
            return mx.array(x, dtype=mx.float32)
        return mx.array(x)
    return x


def _to_numpy(x):
    if isinstance(x, mx.array):
        if x.dtype == mx.bfloat16:
            return np.array(x.astype(mx.float32))
        return np.array(x)
    return x


def load_checkpoint(
    path: Path,
    model,
    optimizer: Optional[object] = None,
    load_optimizer: bool = True,
    target_precision: Optional[str] = None,
) -> int:
    """Load MLX checkpoint into model (and optimizer if provided). Cast back to target_precision if given."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        payload = pickle.load(f)

    params = tree_map(_to_mx_array, payload["params"])
    if target_precision:
        prec = target_precision.lower()
        if prec in ("bfloat16", "bf16"):
            params = tree_map(lambda x: x.astype(mx.bfloat16) if isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating) else x, params)
        elif prec in ("float16", "fp16", "half"):
            params = tree_map(lambda x: x.astype(mx.float16) if isinstance(x, mx.array) and mx.issubdtype(x.dtype, mx.floating) else x, params)
    model.update(params)

    if optimizer is not None and load_optimizer and "opt_state" in payload:
        try:
            optimizer.state = tree_map(_to_mx_array, payload["opt_state"])
        except Exception as exc:
            print(f"⚠️  Optimizer state load failed ({exc}); starting optimizer fresh.")

    return int(payload.get("step", 0))


def save_checkpoint(path: Path, step: int, model, optimizer: Optional[object] = None):
    """Persist MLX checkpoint with numpy-safe payload."""
    payload = {
        "step": int(step),
        "params": tree_map(_to_numpy, model.parameters()),
    }
    if optimizer is not None:
        payload["opt_state"] = tree_map(_to_numpy, optimizer.state)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Saved checkpoint to {path}")
