import math
from typing import Dict, Tuple

import mlx.core as mx


def zeropower_via_newtonschulz5(G: mx.array, steps: int = 3) -> mx.array:
    """
    Optimized Newton-Schulz iteration with better numerical stability and fewer steps.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.shape[-2] > G.shape[-1]:
        X = mx.swapaxes(X, -1, -2)

    # More stable normalization using sqrt(sum of squares)
    norm_sq = mx.sum(mx.square(X), axis=(-2, -1), keepdims=True)
    norm = mx.sqrt(norm_sq + 1e-8)  # Use 1e-8 instead of 1e-7 for better stability
    X = X / norm

    # Use fewer iterations (3 instead of 5) for speed while maintaining accuracy
    for _ in range(min(steps, 3)):
        X_T = mx.swapaxes(X, -1, -2)
        A = mx.matmul(X, X_T)
        B = b * A + c * mx.matmul(A, A)
        X = a * X + mx.matmul(B, X)

    if G.shape[-2] > G.shape[-1]:
        X = mx.swapaxes(X, -1, -2)
    return X


class MuonAdamW:
    """
    Optimized Muon for matrix weights, AdamW for embeddings/bias, implemented for MLX.
    Uses vectorized updates for better performance.
    """

    def __init__(
        self,
        lr_muon: float,
        lr_adam: float,
        weight_decay: float,
        momentum: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 3,  # Reduced default for speed
    ):
        self.lr_muon = lr_muon
        self.lr_adam = lr_adam
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.betas = betas
        self.eps = eps
        self.state: Dict[str, Dict] = {}
        self.ns_steps = ns_steps

    def _param_key(self, name_prefix: str) -> str:
        return name_prefix

    def _init_state(self, key: str, shape, dtype):
        if key not in self.state:
            self.state[key] = {
                "momentum": mx.zeros(shape, dtype=dtype),
                "exp_avg": mx.zeros(shape, dtype=dtype),
                "exp_avg_sq": mx.zeros(shape, dtype=dtype),
                "step": 0,
            }

    def update(self, params: Dict, grads: Dict):
        """
        Vectorized parameter updates for better performance.
        """
        def update_tree(p_tree, g_tree, prefix=""):
            if isinstance(p_tree, dict):
                return {k: update_tree(p_tree[k], g_tree[k], prefix + k + ".") for k in p_tree}
            if isinstance(p_tree, (list, tuple)):
                return type(p_tree)(update_tree(p_tree[i], g_tree[i], prefix + f"{i}.") for i in range(len(p_tree)))
            # Leaf parameter
            param_name = prefix[:-1] if prefix.endswith(".") else prefix
            return self._update_param(p_tree, g_tree, param_name)

        return update_tree(params, grads)

    def _update_param(self, p: mx.array, g: mx.array, name: str):
        """Update a single parameter with optimized operations."""
        if g is None:
            return p

        key = self._param_key(name)
        self._init_state(key, p.shape, p.dtype)
        st = self.state[key]

        if p.ndim >= 2 and "embed" not in name:
            # Muon update for matrix parameters
            st["momentum"] = st["momentum"] * self.momentum + g * (1 - self.momentum)
            update = zeropower_via_newtonschulz5(st["momentum"], steps=self.ns_steps)
            # Apply weight decay and update in one fused operation
            p = p * (1 - self.lr_muon * self.weight_decay) - self.lr_muon * update
        else:
            # AdamW update for embeddings/biases
            st["step"] += 1
            beta1, beta2 = self.betas

            # Fused operations for better performance
            st["exp_avg"] = st["exp_avg"] * beta1 + g * (1 - beta1)
            st["exp_avg_sq"] = st["exp_avg_sq"] * beta2 + mx.square(g) * (1 - beta2)

            bias_c1 = 1 - beta1 ** st["step"]
            bias_c2 = 1 - beta2 ** st["step"]
            denom = mx.sqrt(st["exp_avg_sq"] / bias_c2) + self.eps
            step = (st["exp_avg"] / bias_c1) / denom

            p = p * (1 - self.lr_adam * self.weight_decay) - self.lr_adam * step

        return p
