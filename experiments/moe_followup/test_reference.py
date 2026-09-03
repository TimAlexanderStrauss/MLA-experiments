"""Numerical reference test for the four unchanged attention cells."""

import math

import torch
import torch.nn.functional as F

from model import ATTENTION_MODES, CausalSelfAttention, GPTConfig, _apply_rope


def make_attention(mode: str) -> CausalSelfAttention:
    return CausalSelfAttention(
        GPTConfig(
            block_size=32,
            vocab_size=256,
            n_layer=1,
            n_head=4,
            n_embd=64,
            attn_mode=mode,
            mla_d_c=16,
            mla_d_c_q=24,
            mla_d_rope=8,
        )
    ).float().eval()


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    _, length, _, dimension = q.shape
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    scores = q @ k.transpose(-2, -1) / math.sqrt(dimension)
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous()


def naive_forward(x: torch.Tensor, module: CausalSelfAttention) -> torch.Tensor:
    batch, length, channels = x.shape
    heads, head_dim = module.n_head, module.d_h
    if module.mode in ("mha", "mha_rope"):
        q, k, v = F.linear(x, module.c_attn.weight).split(channels, dim=2)
        q = q.view(batch, length, heads, head_dim)
        k = k.view(batch, length, heads, head_dim)
        v = v.view(batch, length, heads, head_dim)
        if module.mode == "mha":
            q = _apply_rope(q, module.rope_cos, module.rope_sin)
            k = _apply_rope(k, module.rope_cos, module.rope_sin)
        else:
            q_c, q_r = q.split([module.d_content, module.d_rope], dim=-1)
            k_c, k_r = k.split([module.d_content, module.d_rope], dim=-1)
            q = torch.cat([q_c, _apply_rope(q_r, module.rope_cos, module.rope_sin)], -1)
            k = torch.cat([k_c, _apply_rope(k_r, module.rope_cos, module.rope_sin)], -1)
    elif module.mode == "mla_norope":
        c_kv = module.kv_norm(F.linear(x, module.W_DKV.weight))
        k = F.linear(c_kv, module.W_UK.weight).view(batch, length, heads, head_dim)
        v = F.linear(c_kv, module.W_UV.weight).view(batch, length, heads, head_dim)
        c_q = module.q_norm(F.linear(x, module.W_DQ.weight))
        q = F.linear(c_q, module.W_UQ.weight).view(batch, length, heads, head_dim)
        q = _apply_rope(q, module.rope_cos, module.rope_sin)
        k = _apply_rope(k, module.rope_cos, module.rope_sin)
    else:
        c_kv = module.kv_norm(F.linear(x, module.W_DKV.weight))
        k_c = F.linear(c_kv, module.W_UK.weight).view(
            batch, length, heads, module.d_content
        )
        v = F.linear(c_kv, module.W_UV.weight).view(batch, length, heads, head_dim)
        k_r = _apply_rope(
            F.linear(x, module.W_KR.weight).view(batch, length, heads, module.d_rope),
            module.rope_cos,
            module.rope_sin,
        )
        k = torch.cat([k_c, k_r], -1)
        c_q = module.q_norm(F.linear(x, module.W_DQ.weight))
        q_full = F.linear(c_q, module.W_UQ.weight).view(batch, length, heads, head_dim)
        q_c, q_r = q_full.split([module.d_content, module.d_rope], dim=-1)
        q = torch.cat(
            [q_c, _apply_rope(q_r, module.rope_cos, module.rope_sin)], -1
        )
    output = manual_attention(q, k, v).view(batch, length, channels)
    return F.linear(output, module.c_proj.weight)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 16, 64)
    for mode in ATTENTION_MODES:
        module = make_attention(mode)
        with torch.no_grad():
            actual = module(x)
            expected = naive_forward(x, module)
        difference = (actual - expected).abs().max().item()
        print(f"{mode:12s}: max|diff|={difference:.3e}")
        assert difference < 1e-4
    print("All numerical attention references passed.")
