"""MoE-backbone replication of the original Low-Rank x Decoupled-RoPE 2x2.

The four attention modes are intentionally identical to ``experiments/model.py``.
Only the feed-forward backbone changes: layer 0 keeps the original dense GELU
MLP and layers 1..L-1 use a single-GPU adaptation of DeepSeekMoE.

DeepSeek-faithful properties:
* shared experts plus sparsely routed experts;
* float32 softmax router and greedy top-k without top-k renormalisation;
* sequence-wise expert load-balancing loss;
* SwiGLU expert FFNs;
* the first Transformer layer remains dense.

The expert counts and widths are scaled for a 12 GB RTX 5070. Device-limited
routing, device/communication losses and token dropping are deliberately absent
because every expert lives on the same GPU.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ATTENTION_MODES = ("mha", "mha_rope", "mla_norope", "mla")


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
    first_dense_layers: int = 1
    n_shared_experts: int = 2
    n_routed_experts: int = 16
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 336
    aux_loss_alpha: float = 0.001


def _precompute_rope(
    dim: int, max_len: int, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dim % 2 == 0, "RoPE dimension must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    return frequencies.cos(), frequencies.sin()


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
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
        inverse_rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(
            self.eps
        ).rsqrt()
        return (x_float * inverse_rms).to(dtype) * self.weight


class CausalSelfAttention(nn.Module):
    """Exact attention implementation used by the original 2x2 study."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.attn_mode in ATTENTION_MODES
        self.mode = config.attn_mode
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_h = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        if self.mode in ("mha", "mha_rope"):
            self.c_attn = nn.Linear(
                config.n_embd, 3 * config.n_embd, bias=config.bias
            )
        elif self.mode == "mla_norope":
            self.W_DKV = nn.Linear(config.n_embd, config.mla_d_c, bias=False)
            self.kv_norm = RMSNorm(config.mla_d_c)
            self.W_UK = nn.Linear(
                config.mla_d_c, self.n_head * self.d_h, bias=False
            )
            self.W_UV = nn.Linear(
                config.mla_d_c, self.n_head * self.d_h, bias=False
            )
            self.W_DQ = nn.Linear(config.n_embd, config.mla_d_c_q, bias=False)
            self.q_norm = RMSNorm(config.mla_d_c_q)
            self.W_UQ = nn.Linear(
                config.mla_d_c_q, self.n_head * self.d_h, bias=False
            )
        else:
            self.d_rope = config.mla_d_rope
            self.d_content = self.d_h - self.d_rope
            assert self.d_content > 0
            self.W_DKV = nn.Linear(config.n_embd, config.mla_d_c, bias=False)
            self.kv_norm = RMSNorm(config.mla_d_c)
            self.W_UK = nn.Linear(
                config.mla_d_c,
                self.n_head * self.d_content,
                bias=False,
            )
            self.W_UV = nn.Linear(
                config.mla_d_c, self.n_head * self.d_h, bias=False
            )
            self.W_KR = nn.Linear(
                config.n_embd, self.n_head * self.d_rope, bias=False
            )
            self.W_DQ = nn.Linear(config.n_embd, config.mla_d_c_q, bias=False)
            self.q_norm = RMSNorm(config.mla_d_c_q)
            self.W_UQ = nn.Linear(
                config.mla_d_c_q, self.n_head * self.d_h, bias=False
            )

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)
        if self.mode in ("mha", "mla_norope"):
            rope_dim = self.d_h
        else:
            self.d_rope = config.mla_d_rope
            self.d_content = self.d_h - self.d_rope
            rope_dim = self.d_rope
        cos, sin = _precompute_rope(rope_dim, config.block_size, config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        heads, head_dim = self.n_head, self.d_h

        if self.mode == "mha":
            q, k, v = self.c_attn(x).split(channels, dim=2)
            q = q.view(batch_size, sequence_length, heads, head_dim)
            k = k.view(batch_size, sequence_length, heads, head_dim)
            v = v.view(batch_size, sequence_length, heads, head_dim)
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)
        elif self.mode == "mha_rope":
            q, k, v = self.c_attn(x).split(channels, dim=2)
            q = q.view(batch_size, sequence_length, heads, head_dim)
            k = k.view(batch_size, sequence_length, heads, head_dim)
            v = v.view(batch_size, sequence_length, heads, head_dim)
            q_content, q_rope = q.split(
                [self.d_content, self.d_rope], dim=-1
            )
            k_content, k_rope = k.split(
                [self.d_content, self.d_rope], dim=-1
            )
            q = torch.cat(
                [q_content, _apply_rope(q_rope, self.rope_cos, self.rope_sin)],
                dim=-1,
            )
            k = torch.cat(
                [k_content, _apply_rope(k_rope, self.rope_cos, self.rope_sin)],
                dim=-1,
            )
        elif self.mode == "mla_norope":
            c_kv = self.kv_norm(self.W_DKV(x))
            k = self.W_UK(c_kv).view(
                batch_size, sequence_length, heads, head_dim
            )
            v = self.W_UV(c_kv).view(
                batch_size, sequence_length, heads, head_dim
            )
            c_q = self.q_norm(self.W_DQ(x))
            q = self.W_UQ(c_q).view(
                batch_size, sequence_length, heads, head_dim
            )
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)
        else:
            c_kv = self.kv_norm(self.W_DKV(x))
            k_content = self.W_UK(c_kv).view(
                batch_size, sequence_length, heads, self.d_content
            )
            v = self.W_UV(c_kv).view(
                batch_size, sequence_length, heads, head_dim
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
                batch_size, sequence_length, heads, head_dim
            )
            q_content, q_rope = q_full.split(
                [self.d_content, self.d_rope], dim=-1
            )
            q = torch.cat(
                [q_content, _apply_rope(q_rope, self.rope_cos, self.rope_sin)],
                dim=-1,
            )

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


class DenseMLP(nn.Module):
    """Original dense GELU FFN, retained only in the first layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class SwiGLUExpert(nn.Module):
    """DeepSeek-V2 expert FFN: SiLU(gate(x)) * up(x), then down."""

    def __init__(self, n_embd: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    """Single-GPU DeepSeekMoE with diagnostics for every routing decision."""

    def __init__(self, config: GPTConfig, layer_index: int):
        super().__init__()
        assert 0 < config.num_experts_per_tok <= config.n_routed_experts
        assert config.n_shared_experts > 0
        self.layer_index = layer_index
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.aux_loss_alpha = config.aux_loss_alpha
        self.router_weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.n_embd)
        )
        nn.init.kaiming_uniform_(self.router_weight, a=math.sqrt(5))
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(config.n_embd, config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        # A single wider SwiGLU is algebraically the sum of independent shared
        # experts and matches the released DeepSeek-V2 implementation.
        self.shared_experts = SwiGLUExpert(
            config.n_embd,
            config.n_shared_experts * config.moe_intermediate_size,
        )

    def forward(
        self, x: torch.Tensor, collect_routing_stats: bool = True
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[dict[str, torch.Tensor | int]],
    ]:
        batch_size, sequence_length, channels = x.shape
        flat = x.reshape(-1, channels)
        scores = F.linear(
            flat.float(), self.router_weight.float(), bias=None
        ).softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_index = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        repeated = flat.repeat_interleave(self.top_k, dim=0)
        flat_index = topk_index.reshape(-1)
        routed = torch.empty_like(repeated)
        for expert_index, expert in enumerate(self.experts):
            mask = flat_index == expert_index
            routed[mask] = expert(repeated[mask]).to(routed.dtype)
        routed = routed.view(flat.shape[0], self.top_k, channels)
        routed = (
            routed * topk_weight.to(routed.dtype).unsqueeze(-1)
        ).sum(dim=1)
        output = routed.view_as(x) + self.shared_experts(x)

        # Eq. 23-25 in DeepSeek-V2, sequence-wise as in V2-Lite config.
        selected = topk_index.view(batch_size, sequence_length * self.top_k)
        counts_by_sequence = torch.zeros(
            batch_size,
            self.n_routed_experts,
            device=x.device,
            dtype=torch.float32,
        )
        counts_by_sequence.scatter_add_(
            1, selected, torch.ones_like(selected, dtype=torch.float32)
        )
        relative_frequency = counts_by_sequence / (
            sequence_length * self.top_k / self.n_routed_experts
        )
        mean_probability = scores.view(
            batch_size, sequence_length, self.n_routed_experts
        ).mean(dim=1)
        raw_balance_loss = (
            relative_frequency * mean_probability
        ).sum(dim=1).mean()
        auxiliary_loss = raw_balance_loss * self.aux_loss_alpha

        stats = None
        if collect_routing_stats:
            stats = {
                "layer": self.layer_index,
                "selected_counts": torch.bincount(
                    flat_index, minlength=self.n_routed_experts
                ).detach(),
                "probability_sums": scores.sum(dim=0).detach(),
                "entropy_sum": (
                    -(scores * scores.clamp_min(1e-12).log()).sum(dim=-1).sum()
                ).detach(),
                "token_count": flat.shape[0],
            }
        return output, auxiliary_loss, stats


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_index: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.is_moe = layer_index >= config.first_dense_layers
        self.mlp: nn.Module
        if self.is_moe:
            self.mlp = DeepSeekMoE(config, layer_index)
        else:
            self.mlp = DenseMLP(config)

    def forward(
        self, x: torch.Tensor, collect_routing_stats: bool = True
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[dict[str, torch.Tensor | int]],
    ]:
        x = x + self.attn(self.ln_1(x))
        if self.is_moe:
            ffn_output, auxiliary_loss, stats = self.mlp(
                self.ln_2(x), collect_routing_stats
            )
        else:
            ffn_output = self.mlp(self.ln_2(x))
            auxiliary_loss = x.new_zeros((), dtype=torch.float32)
            stats = None
        return x + ffn_output, auxiliary_loss, stats


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert 0 <= config.first_dense_layers <= config.n_layer
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [Block(config, index) for index in range(config.n_layer)]
                ),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith(("c_proj.weight", "proj.weight", "down_proj.weight")):
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
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        collect_routing_stats: bool = True,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        list[dict[str, torch.Tensor | int]],
    ]:
        _, sequence_length = idx.shape
        assert sequence_length <= self.config.block_size
        x = self.transformer.drop(self.transformer.wte(idx))
        auxiliary_loss = x.new_zeros((), dtype=torch.float32)
        routing_stats = []
        for block in self.transformer.h:
            x, block_auxiliary_loss, block_stats = block(
                x, collect_routing_stats
            )
            auxiliary_loss = auxiliary_loss + block_auxiliary_loss.float()
            if block_stats is not None:
                routing_stats.append(block_stats)
        x = self.transformer.ln_f(x)
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            cross_entropy = None
        else:
            logits = self.lm_head(x)
            cross_entropy = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, cross_entropy, auxiliary_loss, routing_stats

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def count_active_parameters(self) -> int:
        active = self.count_parameters()
        for block in self.transformer.h:
            if not block.is_moe:
                continue
            moe = block.mlp
            assert isinstance(moe, DeepSeekMoE)
            one_expert = sum(
                parameter.numel() for parameter in moe.experts[0].parameters()
            )
            active -= (moe.n_routed_experts - moe.top_k) * one_expert
        return active

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        decay = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.dim() >= 2
        ]
        no_decay = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.dim() < 2
        ]
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
            fused=fused_available and device_type == "cuda",
        )
