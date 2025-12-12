import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len].unsqueeze(0)  # shape: (1, seq_len, dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0)  # shape: (1, seq_len, dim)

        # Split the last dimension into even/odd indices for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]  # both have shape (..., seq_len, dim//2)

        # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        # cos and sin are broadcasted to match x1, x2 shapes
        cos_split = cos[..., ::2]  # (1, seq_len, dim//2)
        sin_split = sin[..., ::2]  # (1, seq_len, dim//2)

        rotated = torch.cat([
            x1 * cos_split - x2 * sin_split,
            x1 * sin_split + x2 * cos_split
        ], dim=-1)
        return rotated


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, max_seq_len: int, rope_theta: float):
        super().__init__()
        assert dim % num_heads == 0, "d_model must be divisible by n_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len, theta=rope_theta)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        q = self.rotary(q)
        k = self.rotary(k)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            att = att + mask
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)
        y = rearrange(y, "b h s d -> b s (h d)")
        y = self.out_proj(y)
        return self.resid_dropout(y)


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiple: float, dropout: float, activation: str = "silu"):
        super().__init__()
        hidden = int(dim * multiple)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.silu if activation == "silu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = SelfAttention(
            dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
        )
        self.ln2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(
            dim=config.d_model,
            multiple=config.mlp_multiple,
            dropout=config.dropout,
            activation=config.ffn_activation,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        causal_mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len = idx.size()
        assert seq_len <= self.config.max_seq_len, "sequence length exceeds model max_seq_len"
        tok = self.embed(idx)
        mask = (self.causal_mask[:seq_len, :seq_len] == 0).to(tok.device)
        mask = mask.masked_fill(mask, float("-inf")).unsqueeze(0).unsqueeze(0)
        x = tok
        for block in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
    ) -> torch.Tensor:
        """
        Greedy/top-k sampling generation.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, top_idx, top_vals)
                logits = mask
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
