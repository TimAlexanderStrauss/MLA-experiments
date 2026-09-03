"""
GPT model with 4 causal attention variants for the 2×2 ablation study:

  mha       — Standard Multi-Head Attention + full RoPE (baseline)
  mha_rope  — MHA + Decoupled RoPE: K and Q split into content (no RoPE) + rope (RoPE) parts
  mla_norope— MLA: low-rank KV+Q compression, full RoPE post-decompression (Variante a from design doc)
  mla       — Full MLA: low-rank KV+Q compression + Decoupled RoPE

All models use only token embeddings (no learned position table); position is encoded via RoPE.
Weight tying between wte and lm_head is applied throughout.

Architecture params (matching design doc §4):
  n_layer=6, n_embd=512, n_head=8, d_h=64, block_size=512, vocab=50257
  MLA: mla_d_c=128 (KV latent), mla_d_c_q=192 (Q latent), mla_d_rope=32 (RoPE dims per head)

Design notes:
  * RMSNorm is applied to the down-projected latents c_KV and c_Q before up-projection
    (DeepSeek-V2 Eq. 11) to keep latent scales bounded.
  * K^R in the `mla` mode is per-head (n_head × d_rope) rather than the shared d_rope vector
    used in DeepSeek-V2. This is deliberate: it keeps the Decoupled-RoPE layout symmetric
    with the `mha_rope` baseline, which is the comparison the 2×2 design needs.
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
    # "mha" | "mha_rope" | "mla_norope" | "mla"
    attn_mode: str = "mha"
    # MLA / Decoupled-RoPE hyperparameters
    mla_d_c: int = 128     # KV latent dimension  (d_c   in the paper)
    mla_d_c_q: int = 192   # Q  latent dimension  (d_c^Q in the paper)
    mla_d_rope: int = 32   # RoPE dims per head   (d_h^R in the paper)
    rope_base: float = 10000.0


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _precompute_rope(dim: int, max_len: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos, sin buffers of shape (max_len, dim//2)."""
    assert dim % 2 == 0, "RoPE dim must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (max_len, dim/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to x.
    x   : (B, T, H, D)
    cos : (max_len, D//2)  —  sliced to [:T] inside
    sin : (max_len, D//2)
    Returns tensor of same shape with rotated values.
    """
    T = x.shape[1]
    c = cos[:T].unsqueeze(0).unsqueeze(2)   # (1, T, 1, D//2)
    s = sin[:T].unsqueeze(0).unsqueeze(2)
    # Rotate consecutive pairs: (x0, x1) -> (x0*c - x1*s, x0*s + x1*c)
    xe, xo = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([xe * c - xo * s, xe * s + xo * c], dim=-1)
    return rotated.flatten(-2)


# ---------------------------------------------------------------------------
# RMSNorm (DeepSeek-V2 uses RMSNorm on the latents c_KV and c_Q)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm without bias. Same convention as DeepSeek-V2."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for the reduction to keep numerics stable under bf16
        dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype) * self.weight


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.mode = config.attn_mode
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_h = config.n_embd // config.n_head    # 64
        self.dropout_p = config.dropout

        # ---- weight matrices per mode ----
        if self.mode in ("mha", "mha_rope"):
            # Fused Q, K, V projection — same parameter count for both modes
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        elif self.mode == "mla_norope":
            # Low-rank KV+Q compression; RoPE applied post-decompression on full K/Q
            d_c   = config.mla_d_c
            d_c_q = config.mla_d_c_q
            self.W_DKV   = nn.Linear(config.n_embd, d_c, bias=False)
            self.kv_norm = RMSNorm(d_c)
            self.W_UK    = nn.Linear(d_c, self.n_head * self.d_h, bias=False)
            self.W_UV    = nn.Linear(d_c, self.n_head * self.d_h, bias=False)
            self.W_DQ    = nn.Linear(config.n_embd, d_c_q, bias=False)
            self.q_norm  = RMSNorm(d_c_q)
            self.W_UQ    = nn.Linear(d_c_q, self.n_head * self.d_h, bias=False)

        elif self.mode == "mla":
            # Low-rank KV (content) + per-head Decoupled RoPE K + low-rank Q
            d_c   = config.mla_d_c
            d_c_q = config.mla_d_c_q
            self.d_rope    = config.mla_d_rope
            self.d_content = self.d_h - self.d_rope
            self.W_DKV   = nn.Linear(config.n_embd, d_c, bias=False)
            self.kv_norm = RMSNorm(d_c)
            self.W_UK    = nn.Linear(d_c, self.n_head * self.d_content, bias=False)  # content K
            self.W_UV    = nn.Linear(d_c, self.n_head * self.d_h,       bias=False)  # full V
            self.W_KR    = nn.Linear(config.n_embd, self.n_head * self.d_rope, bias=False)  # rope K
            self.W_DQ    = nn.Linear(config.n_embd, d_c_q, bias=False)
            self.q_norm  = RMSNorm(d_c_q)
            self.W_UQ    = nn.Linear(d_c_q, self.n_head * self.d_h, bias=False)       # full Q (split in fwd)
        else:
            raise ValueError(f"Unknown attn_mode: {self.mode!r}")

        self.c_proj   = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

        # ---- RoPE buffers ----
        # Full-dim RoPE for mha and mla_norope; partial-dim for mha_rope and mla
        if self.mode in ("mha", "mla_norope"):
            rope_dim = self.d_h
        else:
            # mha_rope / mla: RoPE applied only to d_rope dims
            self.d_rope    = config.mla_d_rope
            self.d_content = self.d_h - self.d_rope
            rope_dim = self.d_rope

        cos, sin = _precompute_rope(rope_dim, config.block_size, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.d_h

        if self.mode == "mha":
            q, k, v = self.c_attn(x).split(C, dim=2)
            q = q.view(B, T, H, D)
            k = k.view(B, T, H, D)
            v = v.view(B, T, H, D)
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)

        elif self.mode == "mha_rope":
            q, k, v = self.c_attn(x).split(C, dim=2)
            q = q.view(B, T, H, D)
            k = k.view(B, T, H, D)
            v = v.view(B, T, H, D)
            # Apply RoPE only to the rope portion; leave content portion unchanged
            q_c, q_r = q[..., :self.d_content], q[..., self.d_content:]
            k_c, k_r = k[..., :self.d_content], k[..., self.d_content:]
            q = torch.cat([q_c, _apply_rope(q_r, self.rope_cos, self.rope_sin)], dim=-1)
            k = torch.cat([k_c, _apply_rope(k_r, self.rope_cos, self.rope_sin)], dim=-1)

        elif self.mode == "mla_norope":
            c_kv = self.kv_norm(self.W_DKV(x))           # (B, T, d_c) — what would be cached
            k    = self.W_UK(c_kv).view(B, T, H, D)      # decompressed K
            v    = self.W_UV(c_kv).view(B, T, H, D)
            c_q  = self.q_norm(self.W_DQ(x))             # (B, T, d_c_q)
            q    = self.W_UQ(c_q).view(B, T, H, D)
            # Standard full-dim RoPE on Q and K (post-decompression for K)
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)

        else:  # mla
            # KV: content part from latent, rope part from separate projection
            c_kv = self.kv_norm(self.W_DKV(x))                                      # (B, T, d_c)
            k_c  = self.W_UK(c_kv).view(B, T, H, self.d_content)                    # content K
            v    = self.W_UV(c_kv).view(B, T, H, D)
            k_r  = _apply_rope(
                self.W_KR(x).view(B, T, H, self.d_rope),
                self.rope_cos, self.rope_sin
            )
            k = torch.cat([k_c, k_r], dim=-1)                                        # (B,T,H,D)

            # Q: down-project to latent, up-project to full dim, then split into content + rope
            c_q = self.q_norm(self.W_DQ(x))                                          # (B, T, d_c_q)
            q   = self.W_UQ(c_q).view(B, T, H, D)
            q_c = q[..., :self.d_content]
            q_r = _apply_rope(q[..., self.d_content:], self.rope_cos, self.rope_sin)
            q   = torch.cat([q_c, q_r], dim=-1)

        # Flash attention via SDPA  (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ---------------------------------------------------------------------------
# Feed-forward, Block, GPT
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
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
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: token embedding and output projection share weights
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2-style scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith(("c_proj.weight", "proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence {T} exceeds block_size {self.config.block_size}"

        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Only compute logits for the last token (generation mode)
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_parameters_detail(self) -> dict:
        """Return per-module parameter counts for sanity checking."""
        counts = {}
        for name, module in self.named_modules():
            own = sum(p.numel() for p in module.parameters(recurse=False))
            if own > 0:
                counts[name] = own
        return counts

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple,
        device_type: str,
    ) -> torch.optim.AdamW:
        # Only apply weight decay to 2-D weight tensors (not biases, norms, embeddings)
        decay_params = [p for n, p in self.named_parameters()
                        if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters()
                          if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # fused AdamW is faster on CUDA with PyTorch >= 2.0
        use_fused = device_type == "cuda" and hasattr(torch.optim, "AdamW")
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=fused_available and use_fused,
        )
        return optimizer
