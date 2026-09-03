"""DeepSeek-near 2x2x2 mechanism experiment.

The attention factors are joint low-rank KV compression and the decoupled
RoPE branch. Query compression is disabled in every cell. This follows the
DeepSeek-V2-Lite variant and keeps the low-rank factor specific to KV.

The backbone factor compares a compute-matched dense SwiGLU backbone with a
scaled DeepSeekMoE backbone. Both backbones use the same dense first layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ATTENTION_MODES = (
    "mha",
    "mha_decoupled",
    "mla_coupled",
    "mla_decoupled",
)
BACKBONES = ("dense", "moe")
ATTENTION_FACTORS = {
    "mha": (False, False),
    "mha_decoupled": (False, True),
    "mla_coupled": (True, False),
    "mla_decoupled": (True, True),
}


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
    backbone: str = "dense"
    mla_d_c: int = 256
    mla_d_rope: int = 32
    rope_base: float = 10000.0
    first_dense_layers: int = 1
    n_shared_experts: int = 2
    n_routed_experts: int = 16
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 336
    dense_intermediate_size: int = 1344
    aux_loss_alpha: float = 0.001
    moe_dispatch: str = "batched"
    moe_capacity_factor: float = 2.25


def _precompute_rope(
    dim: int, max_len: int, base: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim <= 0 or dim % 2:
        raise ValueError("RoPE dimension must be a positive even number")
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
    even, odd = x[..., ::2], x[..., 1::2]
    rotated = torch.stack(
        (even * c - odd * s, even * s + odd * c), dim=-1
    )
    return rotated.flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * x.float().pow(2).mean(
            dim=-1, keepdim=True
        ).add(self.eps).rsqrt()
        return normalized.to(dtype) * self.weight


class CausalSelfAttention(nn.Module):
    """Four attention cells with a DeepSeek-near decoupled RoPE branch."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.attn_mode not in ATTENTION_MODES:
            raise ValueError(f"Unknown attention mode: {config.attn_mode}")
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")

        self.mode = config.attn_mode
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.d_h = config.n_embd // config.n_head
        self.d_rope = config.mla_d_rope
        self.dropout_p = config.dropout
        self.low_rank, self.decoupled = ATTENTION_FACTORS[self.mode]

        if self.low_rank:
            self.W_DKV = nn.Linear(config.n_embd, config.mla_d_c, bias=False)
            self.kv_norm = RMSNorm(config.mla_d_c)
            self.W_UK = nn.Linear(
                config.mla_d_c, config.n_head * self.d_h, bias=False
            )
            self.W_UV = nn.Linear(
                config.mla_d_c, config.n_head * self.d_h, bias=False
            )
            if self.decoupled:
                self.W_Q = nn.Linear(
                    config.n_embd,
                    config.n_head * (self.d_h + self.d_rope),
                    bias=config.bias,
                )
                self.W_KR = nn.Linear(config.n_embd, self.d_rope, bias=False)
            else:
                self.W_Q = nn.Linear(
                    config.n_embd, config.n_head * self.d_h, bias=config.bias
                )
        else:
            self.c_attn = nn.Linear(
                config.n_embd, 3 * config.n_embd, bias=config.bias
            )
            if self.decoupled:
                self.W_QR = nn.Linear(
                    config.n_embd,
                    config.n_head * self.d_rope,
                    bias=config.bias,
                )
                self.W_KR = nn.Linear(config.n_embd, self.d_rope, bias=False)

        rope_dim = self.d_rope if self.decoupled else self.d_h
        cos, sin = _precompute_rope(
            rope_dim, config.block_size, config.rope_base
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        shape = (batch_size, sequence_length, self.n_head, self.d_h)

        if not self.low_rank:
            q_content, k_content, v = self.c_attn(x).split(channels, dim=-1)
            q_content = q_content.view(shape)
            k_content = k_content.view(shape)
            v = v.view(shape)
        else:
            c_kv = self.kv_norm(self.W_DKV(x))
            k_content = self.W_UK(c_kv).view(shape)
            v = self.W_UV(c_kv).view(shape)
            q_projected = self.W_Q(x)
            if self.decoupled:
                q_content_flat, q_rope_flat = q_projected.split(
                    [channels, self.n_head * self.d_rope], dim=-1
                )
                q_content = q_content_flat.view(shape)
            else:
                q_content = q_projected.view(shape)

        if self.decoupled:
            if not self.low_rank:
                q_rope_flat = self.W_QR(x)
            q_rope = _apply_rope(
                q_rope_flat.view(
                    batch_size, sequence_length, self.n_head, self.d_rope
                ),
                self.rope_cos,
                self.rope_sin,
            )
            k_rope_shared = _apply_rope(
                self.W_KR(x).view(
                    batch_size, sequence_length, 1, self.d_rope
                ),
                self.rope_cos,
                self.rope_sin,
            )
            k_rope = k_rope_shared.expand(-1, -1, self.n_head, -1)
            q = torch.cat((q_content, q_rope), dim=-1)
            k = torch.cat((k_content, k_rope), dim=-1)
        else:
            q = _apply_rope(q_content, self.rope_cos, self.rope_sin)
            k = _apply_rope(k_content, self.rope_cos, self.rope_sin)

        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        return self.resid_drop(self.c_proj(y))

    def logical_kv_cache_elements_per_token(self) -> int:
        """Return the minimal cache without recomputing prefix keys."""
        if self.mode == "mla_decoupled":
            return self.W_DKV.out_features + self.d_rope
        if self.mode == "mha_decoupled":
            return 2 * self.n_head * self.d_h + self.d_rope
        return 2 * self.n_head * self.d_h

    def logical_kv_cache_elements_with_recompute_per_token(self) -> int:
        """Return the minimal cache when prefix K/V recomputation is allowed."""
        if self.low_rank:
            cache = self.W_DKV.out_features
            return cache + self.d_rope if self.decoupled else cache
        return self.logical_kv_cache_elements_per_token()


class DenseSwiGLU(nn.Module):
    def __init__(self, n_embd: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.up_proj = nn.Linear(n_embd, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoE(nn.Module):
    """Single-GPU DeepSeekMoE with loop and packed batched dispatch."""

    def __init__(self, config: GPTConfig, layer_index: int):
        super().__init__()
        if not 0 < config.num_experts_per_tok <= config.n_routed_experts:
            raise ValueError("Invalid routed top-k")
        if config.moe_dispatch not in ("loop", "batched"):
            raise ValueError("moe_dispatch must be 'loop' or 'batched'")
        self.layer_index = layer_index
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.moe_intermediate_size
        self.aux_loss_alpha = config.aux_loss_alpha
        self.dispatch = config.moe_dispatch
        self.capacity_factor = config.moe_capacity_factor

        self.router_weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.n_embd)
        )
        self.expert_gate = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                config.moe_intermediate_size,
                config.n_embd,
            )
        )
        self.expert_up = nn.Parameter(torch.empty_like(self.expert_gate))
        self.expert_down = nn.Parameter(
            torch.empty(
                config.n_routed_experts,
                config.n_embd,
                config.moe_intermediate_size,
            )
        )
        nn.init.kaiming_uniform_(self.router_weight, a=math.sqrt(5))
        nn.init.normal_(self.expert_gate, mean=0.0, std=0.02)
        nn.init.normal_(self.expert_up, mean=0.0, std=0.02)
        nn.init.normal_(self.expert_down, mean=0.0, std=0.02)
        self.shared_experts = DenseSwiGLU(
            config.n_embd,
            config.n_shared_experts * config.moe_intermediate_size,
        )

    def _expert_forward(
        self, expert_index: int, x: torch.Tensor
    ) -> torch.Tensor:
        gate = F.linear(x, self.expert_gate[expert_index])
        up = F.linear(x, self.expert_up[expert_index])
        return F.linear(F.silu(gate) * up, self.expert_down[expert_index])

    def _loop_dispatch(
        self,
        flat: torch.Tensor,
        topk_index: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        token_index = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(
            self.top_k
        )
        expert_index = topk_index.reshape(-1)
        weights = topk_weight.reshape(-1).to(flat.dtype)
        output = torch.zeros_like(flat)
        for expert in range(self.n_routed_experts):
            positions = torch.nonzero(expert_index == expert, as_tuple=False).flatten()
            selected_tokens = token_index.index_select(0, positions)
            selected_input = flat.index_select(0, selected_tokens)
            selected_output = self._expert_forward(expert, selected_input)
            weighted = selected_output * weights.index_select(0, positions).unsqueeze(-1)
            output = output.index_add(0, selected_tokens, weighted)
        return output

    def _batched_dispatch(
        self,
        flat: torch.Tensor,
        topk_index: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Pack expert inputs and use three strided batched matrix multiplies."""
        token_count, channels = flat.shape
        token_index = torch.arange(token_count, device=flat.device).repeat_interleave(
            self.top_k
        )
        expert_index = topk_index.reshape(-1)
        assignment_count = expert_index.numel()
        counts = torch.bincount(
            expert_index, minlength=self.n_routed_experts
        )
        order = torch.argsort(expert_index, stable=True)
        sorted_experts = expert_index.index_select(0, order)
        group_starts = counts.cumsum(dim=0) - counts
        sorted_slots = torch.arange(
            assignment_count, device=flat.device
        ) - group_starts.index_select(0, sorted_experts)
        slots = torch.empty_like(sorted_slots).scatter(0, order, sorted_slots)

        if not torch.compiler.is_compiling():
            required = int(counts.max().item())
            capacity = max(8, math.ceil(required / 8) * 8)
        else:
            capacity = math.ceil(
                assignment_count
                / self.n_routed_experts
                * self.capacity_factor
                / 8
            ) * 8
            torch._assert(
                counts.max() <= capacity,
                "MoE routing exceeded the compiled capacity. Increase "
                "moe_capacity_factor or use eager batched dispatch.",
            )

        packed = flat.new_zeros(
            self.n_routed_experts, capacity, channels
        )
        packed = packed.index_put(
            (expert_index, slots), flat.index_select(0, token_index)
        )
        gate = torch.bmm(packed, self.expert_gate.transpose(1, 2))
        up = torch.bmm(packed, self.expert_up.transpose(1, 2))
        hidden = F.silu(gate) * up
        packed_output = torch.bmm(hidden, self.expert_down.transpose(1, 2))
        selected_output = packed_output[expert_index, slots]
        weighted = selected_output * topk_weight.reshape(-1).to(
            flat.dtype
        ).unsqueeze(-1)
        return torch.zeros_like(flat).index_add(0, token_index, weighted)

    def forward(
        self, x: torch.Tensor, collect_routing_stats: bool = False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[dict[str, torch.Tensor | int]],
    ]:
        batch_size, sequence_length, channels = x.shape
        flat = x.reshape(-1, channels)
        scores = F.linear(
            flat.float(), self.router_weight.float()
        ).softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_index = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )
        if self.dispatch == "batched":
            routed = self._batched_dispatch(flat, topk_index, topk_weight)
        else:
            routed = self._loop_dispatch(flat, topk_index, topk_weight)
        output = routed.view_as(x) + self.shared_experts(x)

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
            flat_index = topk_index.reshape(-1)
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
        self.input_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.n_embd)
        self.is_moe = (
            config.backbone == "moe"
            and layer_index >= config.first_dense_layers
        )
        if self.is_moe:
            self.ffn: nn.Module = DeepSeekMoE(config, layer_index)
        else:
            self.ffn = DenseSwiGLU(
                config.n_embd, config.dense_intermediate_size
            )

    def forward(
        self, x: torch.Tensor, collect_routing_stats: bool = False
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[dict[str, torch.Tensor | int]],
    ]:
        x = x + self.attn(self.input_norm(x))
        normalized = self.post_attn_norm(x)
        if self.is_moe:
            ffn_output, auxiliary_loss, stats = self.ffn(
                normalized, collect_routing_stats
            )
        else:
            ffn_output = self.ffn(normalized)
            auxiliary_loss = x.new_zeros((), dtype=torch.float32)
            stats = None
        return x + ffn_output, auxiliary_loss, stats


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone: {config.backbone}")
        expected_dense_width = (
            (config.n_shared_experts + config.num_experts_per_tok)
            * config.moe_intermediate_size
        )
        if config.dense_intermediate_size != expected_dense_width:
            raise ValueError(
                "dense_intermediate_size must match the active MoE width "
                f"({expected_dense_width})"
            )
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [Block(config, index) for index in range(config.n_layer)]
                ),
                "norm": RMSNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith(("c_proj.weight", "down_proj.weight", "expert_down")):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
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
        collect_routing_stats: bool = False,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        list[dict[str, torch.Tensor | int]],
    ]:
        _, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError("Input is longer than block_size")
        x = self.transformer.drop(self.transformer.wte(idx))
        auxiliary_loss = x.new_zeros((), dtype=torch.float32)
        routing_stats = []
        for block in self.transformer.h:
            x, block_auxiliary_loss, stats = block(
                x, collect_routing_stats
            )
            auxiliary_loss = auxiliary_loss + block_auxiliary_loss.float()
            if stats is not None:
                routing_stats.append(stats)
        x = self.transformer.norm(x)
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            cross_entropy = None
        else:
            logits = self.lm_head(x)
            cross_entropy = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, cross_entropy, auxiliary_loss, routing_stats

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def count_active_parameters(self) -> int:
        active = self.count_parameters()
        for block in self.transformer.h:
            if not block.is_moe:
                continue
            moe = block.ffn
            assert isinstance(moe, DeepSeekMoE)
            one_expert = (
                moe.expert_gate[0].numel()
                + moe.expert_up[0].numel()
                + moe.expert_down[0].numel()
            )
            active -= (moe.n_routed_experts - moe.top_k) * one_expert
        return active

    def logical_kv_cache_elements_per_token(self) -> int:
        attention = self.transformer.h[0].attn
        return self.config.n_layer * attention.logical_kv_cache_elements_per_token()

    def logical_kv_cache_elements_with_recompute_per_token(self) -> int:
        attention = self.transformer.h[0].attn
        return (
            self.config.n_layer
            * attention.logical_kv_cache_elements_with_recompute_per_token()
        )

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
