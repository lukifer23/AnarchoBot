import math
from typing import Optional, Tuple, Callable, Any
from functools import partial

import mlx.core as mx
import mlx.nn as nn


def gradient_checkpoint(func: Callable, *args, **kwargs) -> mx.array:
    """
    Gradient checkpointing for MLX.
    Currently a no-op as MLX relies on graph compilation and lazy evaluation.
    Physical activaton checkpointing is less critical than in eager frameworks,
    but `mx.checkpoint` exists if needed for long sequences.
    """
    # TODO: Implement proper gradient checkpointing for MLX when available
    # For now, just call the function normally
    return func(*args, **kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.layernorm = nn.RMSNorm(dim, eps=eps)

    def __call__(self, x: mx.array) -> mx.array:
        return self.layernorm(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.einsum("i,j->ij", t, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)

    def __call__(self, x: mx.array) -> mx.array:
        seq_len = x.shape[-2]
        cos = self.cos_cached[:seq_len, ::2][None, None, :, :]
        sin = self.sin_cached[:seq_len, ::2][None, None, :, :]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return rotated


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, max_seq_len: int, rope_theta: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len, theta=rope_theta)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        bsz, seq_len, _ = x.shape

        # Fused QKV projection for better memory access
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Optimized reshape and transpose operations
        q = q.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply rotary embeddings
        q = self.rotary(q)
        k = self.rotary(k)

        # Scaled dot-product attention with fused operations
        scale = 1.0 / math.sqrt(self.head_dim)
        att = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            # Apply causal mask by adding it (mask contains -inf values)
            att = att + mask

        att = mx.softmax(att, axis=-1)
        att = self.attn_dropout(att)

        # Attention output
        y = mx.matmul(att, v)
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)

        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiple: float, dropout: float, activation: str = "silu"):
        super().__init__()
        hidden = int(dim * multiple)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.activation = nn.silu if activation == "silu" else nn.gelu
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_multiple: float, dropout: float, max_seq_len: int, rope_theta: float, norm_eps: float):
        super().__init__()
        self.ln1 = RMSNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, n_heads, dropout, max_seq_len, rope_theta)
        self.ln2 = RMSNorm(dim, eps=norm_eps)
        self.mlp = FeedForward(dim, mlp_multiple, dropout)
        self.use_checkpointing = False

    def enable_checkpointing(self, enabled: bool = True):
        self.use_checkpointing = enabled

    def _forward_impl(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        if self.use_checkpointing:
            return gradient_checkpoint(self._forward_impl, x, mask)
        return self._forward_impl(x, mask)


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        mlp_multiple: float,
        dropout: float,
        max_seq_len: int,
        rope_theta: float,
        ffn_activation: str = "silu",
        norm_eps: float = 1e-5,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = [
            TransformerBlock(
                dim=d_model,
                n_heads=n_heads,
                mlp_multiple=mlp_multiple,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ]
        self.gradient_checkpointing = False
        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        causal_mask = mx.tril(mx.ones((max_seq_len, max_seq_len), dtype=mx.float32), k=0)
        self.causal_mask = mx.log(causal_mask).reshape((1, 1, max_seq_len, max_seq_len))

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled
        for block in self.layers:
            block.enable_checkpointing(enabled)

    def __call__(self, idx: mx.array, targets: Optional[mx.array] = None) -> Tuple[mx.array, Optional[mx.array]]:
        bsz, seq_len = idx.shape
        assert seq_len <= self.max_seq_len, "sequence length exceeds max_seq_len"
        tok = self.embed(idx)
        # Use cached causal mask slice; avoid redundant host/device copies
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        x = tok
        for block in self.layers:
            x = block(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # Compute loss in float32 to reduce overflow risk when training in reduced precision
            logits_flat = logits.reshape((-1, logits.shape[-1])).astype(mx.float32)
            targets_flat = targets.reshape((-1,))
            loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
        return logits, loss

    def generate(
        self,
        idx: mx.array,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
    ) -> mx.array:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len :]
            logits, _ = self(idx_cond, None)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                # Simplified top-k sampling for MLX compatibility
                # Sort and keep only top-k values
                sorted_logits = mx.sort(logits)
                top_k_logits = sorted_logits[..., -top_k:]
                # Set everything below top-k threshold to -inf
                threshold = mx.min(top_k_logits, axis=-1, keepdims=True)
                logits = mx.where(logits >= threshold, logits, -1e9)
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probs)
            idx = mx.concatenate([idx, next_token[..., None]], axis=1)
        return idx
