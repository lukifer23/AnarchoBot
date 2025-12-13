import math
from typing import Dict, Tuple

import mlx.core as mx


def zeropower_via_newtonschulz5(G: mx.array, steps: int = 5) -> mx.array:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.shape[-2] > G.shape[-1]:
        X = mx.swapaxes(X, -1, -2)
    X = X / (mx.linalg.norm(X, axis=(-2, -1), keepdims=True) + 1e-7)
    for _ in range(steps):
        A = mx.matmul(X, mx.swapaxes(X, -1, -2))
        B = b * A + c * mx.matmul(A, A)
        X = a * X + mx.matmul(B, X)
    if G.shape[-2] > G.shape[-1]:
        X = mx.swapaxes(X, -1, -2)
    return X


class MuonAdamW:
    """
    Muon for matrix weights, AdamW for embeddings/bias, implemented for MLX.
    """

    def __init__(self, lr_muon: float, lr_adam: float, weight_decay: float, momentum: float = 0.95, betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-8):
        self.lr_muon = lr_muon
        self.lr_adam = lr_adam
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.betas = betas
        self.eps = eps
        self.state: Dict[str, Dict] = {}

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
        def apply(p, g, name_prefix):
            key = self._param_key(name_prefix)
            self._init_state(key, p.shape, p.dtype)
            st = self.state[key]
            if g is None:
                return p
            if p.ndim >= 2 and "embed" not in name_prefix:
                st["momentum"] = st["momentum"] * self.momentum + g * (1 - self.momentum)
                update = zeropower_via_newtonschulz5(st["momentum"])
                p = p * (1 - self.lr_muon * self.weight_decay)
                p = p - self.lr_muon * update
            else:
                st["step"] += 1
                beta1, beta2 = self.betas
                st["exp_avg"] = st["exp_avg"] * beta1 + g * (1 - beta1)
                st["exp_avg_sq"] = st["exp_avg_sq"] * beta2 + mx.square(g) * (1 - beta2)
                bias_c1 = 1 - beta1 ** st["step"]
                bias_c2 = 1 - beta2 ** st["step"]
                denom = mx.sqrt(st["exp_avg_sq"] / bias_c2) + self.eps
                step = (st["exp_avg"] / bias_c1) / denom
                p = p * (1 - self.lr_adam * self.weight_decay)
                p = p - self.lr_adam * step
            return p

        def apply_tree(p_tree, g_tree, prefix=""):
            if isinstance(p_tree, dict):
                return {k: apply_tree(p_tree[k], g_tree[k], prefix + k + ".") for k in p_tree}
            if isinstance(p_tree, (list, tuple)):
                return type(p_tree)(apply_tree(p_tree[i], g_tree[i], prefix + f"{i}.") for i in range(len(p_tree)))
            return apply(p_tree, g_tree, prefix.rstrip("."))

        return apply_tree(params, grads)
