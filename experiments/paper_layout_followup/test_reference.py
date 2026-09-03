"""Numerical reference tests for all three attention modes."""

import math

import torch
import torch.nn.functional as F

from model import CausalSelfAttention, GPTConfig, _apply_rope


MODES = ("mha", "mla_current", "mla_deepseek")
TOLERANCE = 1e-4


def make_attention(mode: str) -> CausalSelfAttention:
    config = GPTConfig(
        block_size=64,
        vocab_size=512,
        n_layer=1,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        attn_mode=mode,
        mla_d_c=32,
        mla_d_c_q=48,
        mla_d_rope=16,
    )
    return CausalSelfAttention(config).float().eval()


def manual_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    # Inputs: (B,T,H,Dqk), (B,T,H,Dqk), (B,T,H,Dv).
    _, sequence_length, _, qk_dimension = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(qk_dimension)
    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(mask, float("-inf"))
    output = F.softmax(scores, dim=-1) @ v
    return output.transpose(1, 2).contiguous()


def naive_forward(x: torch.Tensor, module: CausalSelfAttention) -> torch.Tensor:
    batch_size, sequence_length, channels = x.shape
    heads, d_h = module.n_head, module.d_h

    if module.mode == "mha":
        q, k, v = F.linear(x, module.c_attn.weight).split(channels, dim=2)
        q = _apply_rope(
            q.view(batch_size, sequence_length, heads, d_h),
            module.rope_cos,
            module.rope_sin,
        )
        k = _apply_rope(
            k.view(batch_size, sequence_length, heads, d_h),
            module.rope_cos,
            module.rope_sin,
        )
        v = v.view(batch_size, sequence_length, heads, d_h)

    elif module.mode == "mla_current":
        c_kv = module.kv_norm(F.linear(x, module.W_DKV.weight))
        k_content = F.linear(c_kv, module.W_UK.weight).view(
            batch_size, sequence_length, heads, module.d_content
        )
        v = F.linear(c_kv, module.W_UV.weight).view(
            batch_size, sequence_length, heads, d_h
        )
        k_rope = _apply_rope(
            F.linear(x, module.W_KR.weight).view(
                batch_size, sequence_length, heads, module.d_rope
            ),
            module.rope_cos,
            module.rope_sin,
        )
        k = torch.cat([k_content, k_rope], dim=-1)
        c_q = module.q_norm(F.linear(x, module.W_DQ.weight))
        q_full = F.linear(c_q, module.W_UQ.weight).view(
            batch_size, sequence_length, heads, d_h
        )
        q = torch.cat(
            [
                q_full[..., : module.d_content],
                _apply_rope(
                    q_full[..., module.d_content :],
                    module.rope_cos,
                    module.rope_sin,
                ),
            ],
            dim=-1,
        )

    else:
        c_kv = module.kv_norm(F.linear(x, module.W_DKV.weight))
        k_content = F.linear(c_kv, module.W_UK.weight).view(
            batch_size, sequence_length, heads, module.d_content
        )
        v = F.linear(c_kv, module.W_UV.weight).view(
            batch_size, sequence_length, heads, d_h
        )
        k_shared = _apply_rope(
            F.linear(x, module.W_KR.weight).view(
                batch_size, sequence_length, 1, module.d_rope
            ),
            module.rope_cos,
            module.rope_sin,
        )
        k = torch.cat(
            [k_content, k_shared.expand(-1, -1, heads, -1)], dim=-1
        )
        c_q = module.q_norm(F.linear(x, module.W_DQ.weight))
        q_content = F.linear(c_q, module.W_UQ.weight).view(
            batch_size, sequence_length, heads, module.d_content
        )
        q_rope = _apply_rope(
            F.linear(c_q, module.W_QR.weight).view(
                batch_size, sequence_length, heads, module.d_rope
            ),
            module.rope_cos,
            module.rope_sin,
        )
        q = torch.cat([q_content, q_rope], dim=-1)

    output = manual_causal_attention(q, k, v).view(
        batch_size, sequence_length, channels
    )
    return F.linear(output, module.c_proj.weight)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 32, 128)
    for mode in MODES:
        module = make_attention(mode)
        with torch.no_grad():
            production = module(x)
            reference = naive_forward(x, module)
        difference = (production - reference).abs().max().item()
        print(f"{mode:14s}: max|diff|={difference:.3e}")
        assert difference < TOLERANCE
    print("All numerical reference tests passed.")
