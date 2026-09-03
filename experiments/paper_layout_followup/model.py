"""Models for the DeepSeek-layout sensitivity follow-up.

The study contains exactly three conditions:

``mha``
    Existing MHA baseline: Q/K/V head dimension 64, full RoPE on Q and K.

``mla_current``
    The full-MLA condition from the original 2x2 experiment: the fixed
    64-dimensional Q/K head is split into 32 content + 32 RoPE dimensions,
    and K^R is separate per head.

``mla_deepseek``
    A more paper-faithful sensitivity condition: 64 content dimensions are
    retained and 32 RoPE dimensions are appended, so Q/K have dimension 96;
    K^R is one shared 32-dimensional vector broadcast to all heads. Values
    remain 64-dimensional. Q compression is retained so that the comparison
    to ``mla_current`` changes the RoPE layout, not the Q bottleneck.

This is a focused sensitivity study, not a reproduction of the full
DeepSeek-V2 MoE model.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    attn_mode: str = "mha"
    mla_d_c: int = 128
    mla_d_c_q: int = 192
    mla_d_rope: int = 32
    rope_base: float = 10000.0


def _precompute_rope(
    dim: int, max_len: int, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create RoPE cosine and sine tables with shape (max_len, dim / 2)."""
    assert dim % 2 == 0, "RoPE dimension must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    return frequencies.cos(), frequencies.sin()


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE to a tensor shaped (batch, time, heads, dimension)."""
    sequence_length = x.shape[1]
    c = cos[:sequence_length].unsqueeze(0).unsqueeze(2)
    s = sin[:sequence_length].unsqueeze(0).unsqueeze(2)
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    rotated = torch.stack(
        [x_even * c - x_odd * s, x_even * s + x_odd * c], dim=-1
    )
    return rotated.flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        inv_rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_float * inv_rms).to(dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.attn_mode in ("mha", "mla_current", "mla_deepseek")

        self.mode = config.attn_mode
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_h = config.n_embd // config.n_head
        self.d_rope = config.mla_d_rope
        self.dropout_p = config.dropout

        if self.mode == "mha":
            self.c_attn = nn.Linear(
                config.n_embd, 3 * config.n_embd, bias=config.bias
            )
            rope_dim = self.d_h

        elif self.mode == "mla_current":
            self.d_content = self.d_h - self.d_rope
            assert self.d_content > 0
            self.W_DKV = nn.Linear(config.n_embd, config.mla_d_c, bias=False)
            self.kv_norm = RMSNorm(config.mla_d_c)
            self.W_UK = nn.Linear(
                config.mla_d_c, self.n_head * self.d_content, bias=False
            )
            self.W_UV = nn.Linear(
                config.mla_d_c, self.n_head * self.d_h, bias=False
            )
            # Original experiment: one K^R for every head.
            self.W_KR = nn.Linear(
                config.n_embd, self.n_head * self.d_rope, bias=False
            )
            self.W_DQ = nn.Linear(config.n_embd, config.mla_d_c_q, bias=False)
            self.q_norm = RMSNorm(config.mla_d_c_q)
            self.W_UQ = nn.Linear(
                config.mla_d_c_q, self.n_head * self.d_h, bias=False
            )
            rope_dim = self.d_rope

        else:  # mla_deepseek
            # DeepSeek layout: d_h content dimensions stay intact and the
            # d_rope positional dimensions are appended to Q/K.
            self.d_content = self.d_h
            self.qk_dim = self.d_content + self.d_rope
            self.W_DKV = nn.Linear(config.n_embd, config.mla_d_c, bias=False)
            self.kv_norm = RMSNorm(config.mla_d_c)
            self.W_UK = nn.Linear(
                config.mla_d_c, self.n_head * self.d_content, bias=False
            )
            self.W_UV = nn.Linear(
                config.mla_d_c, self.n_head * self.d_h, bias=False
            )
            # Paper layout: one shared K^R, broadcast to all attention heads.
            self.W_KR = nn.Linear(config.n_embd, self.d_rope, bias=False)
            self.W_DQ = nn.Linear(config.n_embd, config.mla_d_c_q, bias=False)
            self.q_norm = RMSNorm(config.mla_d_c_q)
            self.W_UQ = nn.Linear(
                config.mla_d_c_q, self.n_head * self.d_content, bias=False
            )
            self.W_QR = nn.Linear(
                config.mla_d_c_q, self.n_head * self.d_rope, bias=False
            )
            rope_dim = self.d_rope

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

        cos, sin = _precompute_rope(rope_dim, config.block_size, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        heads, value_dim = self.n_head, self.d_h

        if self.mode == "mha":
            q, k, v = self.c_attn(x).split(channels, dim=2)
            q = q.view(batch_size, sequence_length, heads, value_dim)
            k = k.view(batch_size, sequence_length, heads, value_dim)
            v = v.view(batch_size, sequence_length, heads, value_dim)
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)

        elif self.mode == "mla_current":
            c_kv = self.kv_norm(self.W_DKV(x))
            k_content = self.W_UK(c_kv).view(
                batch_size, sequence_length, heads, self.d_content
            )
            v = self.W_UV(c_kv).view(
                batch_size, sequence_length, heads, value_dim
            )
            k_rope = _apply_rope(
                self.W_KR(x).view(
                    batch_size, sequence_length, heads, self.d_rope
                ),
                self.rope_cos,
                self.rope_sin,
            )
            k = torch.cat([k_content, k_rope], dim=-1)

            c_q = self.q_norm(self.W_DQ(x))
            q_full = self.W_UQ(c_q).view(
                batch_size, sequence_length, heads, value_dim
            )
            q_content = q_full[..., : self.d_content]
            q_rope = _apply_rope(
                q_full[..., self.d_content :], self.rope_cos, self.rope_sin
            )
            q = torch.cat([q_content, q_rope], dim=-1)

        else:  # mla_deepseek
            c_kv = self.kv_norm(self.W_DKV(x))
            k_content = self.W_UK(c_kv).view(
                batch_size, sequence_length, heads, self.d_content
            )
            v = self.W_UV(c_kv).view(
                batch_size, sequence_length, heads, value_dim
            )

            # Shape (B,T,1,R), then broadcast the same rotated K^R to all H.
            k_rope_shared = _apply_rope(
                self.W_KR(x).view(
                    batch_size, sequence_length, 1, self.d_rope
                ),
                self.rope_cos,
                self.rope_sin,
            )
            k_rope = k_rope_shared.expand(-1, -1, heads, -1)
            k = torch.cat([k_content, k_rope], dim=-1)

            c_q = self.q_norm(self.W_DQ(x))
            q_content = self.W_UQ(c_q).view(
                batch_size, sequence_length, heads, self.d_content
            )
            q_rope = _apply_rope(
                self.W_QR(c_q).view(
                    batch_size, sequence_length, heads, self.d_rope
                ),
                self.rope_cos,
                self.rope_sin,
            )
            q = torch.cat([q_content, q_rope], dim=-1)

        # Q and K can have 96 dimensions while V remains 64-dimensional.
        # PyTorch SDPA scales by the Q/K head dimension automatically.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith(("c_proj.weight", "proj.weight")):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, sequence_length = idx.shape
        assert sequence_length <= self.config.block_size
        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=betas,
            fused=fused_available and device_type == "cuda",
        )
