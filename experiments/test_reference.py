"""
Numerical reference test for the MLA / Decoupled-RoPE implementation in model.py.

For each of the four attention modes we re-implement a *naive*, step-by-step
version of the attention computation from the design doc / DeepSeek-V2 paper
and compare the output of the production `CausalSelfAttention` module against
it on shared weights and identical inputs.

A diff above 1e-4 (fp32) is reported as a bug.

Why this exists: `test_correctness.py` checks structure (shapes, no NaN,
loss decreases). It does NOT verify that the optimized forward implements
the *intended* mathematics. A subtle indexing bug — swapped content/rope
slices, wrong RoPE application order, mis-shaped K^R — would survive that
test and only manifest as "MLA underperforms MHA" in the final results,
which we would falsely interpret as a real ablation effect.

Usage:
  cd experiments
  python test_reference.py
"""

import torch
import torch.nn.functional as F

from model import (
    CausalSelfAttention,
    GPTConfig,
    RMSNorm,
    _apply_rope,
    _precompute_rope,
)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
TOL_FP32 = 1e-4   # production vs naive in fp32
TOL_FP32_TIGHT = 5e-5

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Naive reference implementations (one per attention mode)
# ---------------------------------------------------------------------------

def _naive_causal_softmax_attn(q, k, v):
    """
    Naive (B, T, H, D) -> (B, T, H, D) causal attention.
    Computes softmax(QK^T / sqrt(D)) V manually instead of via SDPA.
    """
    B, T, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    # (B, H, T, D)
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale     # (B, H, T, T)
    mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out  = torch.matmul(attn, vh)                               # (B, H, T, D)
    return out.transpose(1, 2).contiguous()                     # (B, T, H, D)


def naive_mha(x, mod: CausalSelfAttention):
    """Standard MHA + full RoPE."""
    B, T, C = x.shape
    H, D = mod.n_head, mod.d_h
    qkv = F.linear(x, mod.c_attn.weight, mod.c_attn.bias)
    q, k, v = qkv.split(C, dim=2)
    q = q.view(B, T, H, D)
    k = k.view(B, T, H, D)
    v = v.view(B, T, H, D)
    q = _apply_rope(q, mod.rope_cos, mod.rope_sin)
    k = _apply_rope(k, mod.rope_cos, mod.rope_sin)
    y = _naive_causal_softmax_attn(q, k, v).view(B, T, C)
    return F.linear(y, mod.c_proj.weight, mod.c_proj.bias)


def naive_mha_rope(x, mod: CausalSelfAttention):
    """MHA + Decoupled RoPE: K/Q each split into content (no RoPE) + rope (RoPE)."""
    B, T, C = x.shape
    H, D = mod.n_head, mod.d_h
    d_c, d_r = mod.d_content, mod.d_rope
    qkv = F.linear(x, mod.c_attn.weight, mod.c_attn.bias)
    q, k, v = qkv.split(C, dim=2)
    q = q.view(B, T, H, D)
    k = k.view(B, T, H, D)
    v = v.view(B, T, H, D)

    q_content, q_rope = q[..., :d_c], q[..., d_c:]
    k_content, k_rope = k[..., :d_c], k[..., d_c:]
    q_rope = _apply_rope(q_rope, mod.rope_cos, mod.rope_sin)
    k_rope = _apply_rope(k_rope, mod.rope_cos, mod.rope_sin)
    q = torch.cat([q_content, q_rope], dim=-1)
    k = torch.cat([k_content, k_rope], dim=-1)

    y = _naive_causal_softmax_attn(q, k, v).view(B, T, C)
    return F.linear(y, mod.c_proj.weight, mod.c_proj.bias)


def naive_mla_norope(x, mod: CausalSelfAttention):
    """MLA without Decoupled RoPE: KV+Q both compressed via latents; full-dim RoPE post-decompression."""
    B, T, C = x.shape
    H, D = mod.n_head, mod.d_h

    c_kv = F.linear(x, mod.W_DKV.weight)
    c_kv = mod.kv_norm(c_kv)
    k = F.linear(c_kv, mod.W_UK.weight).view(B, T, H, D)
    v = F.linear(c_kv, mod.W_UV.weight).view(B, T, H, D)

    c_q = F.linear(x, mod.W_DQ.weight)
    c_q = mod.q_norm(c_q)
    q = F.linear(c_q, mod.W_UQ.weight).view(B, T, H, D)

    q = _apply_rope(q, mod.rope_cos, mod.rope_sin)
    k = _apply_rope(k, mod.rope_cos, mod.rope_sin)

    y = _naive_causal_softmax_attn(q, k, v).view(B, T, C)
    return F.linear(y, mod.c_proj.weight, mod.c_proj.bias)


def naive_mla(x, mod: CausalSelfAttention):
    """Full MLA: low-rank KV + Decoupled RoPE + low-rank Q."""
    B, T, C = x.shape
    H, D = mod.n_head, mod.d_h
    d_content, d_rope = mod.d_content, mod.d_rope

    c_kv = mod.kv_norm(F.linear(x, mod.W_DKV.weight))
    k_c  = F.linear(c_kv, mod.W_UK.weight).view(B, T, H, d_content)
    v    = F.linear(c_kv, mod.W_UV.weight).view(B, T, H, D)

    k_r  = F.linear(x, mod.W_KR.weight).view(B, T, H, d_rope)
    k_r  = _apply_rope(k_r, mod.rope_cos, mod.rope_sin)
    k    = torch.cat([k_c, k_r], dim=-1)

    c_q = mod.q_norm(F.linear(x, mod.W_DQ.weight))
    q   = F.linear(c_q, mod.W_UQ.weight).view(B, T, H, D)
    q_c = q[..., :d_content]
    q_r = _apply_rope(q[..., d_content:], mod.rope_cos, mod.rope_sin)
    q   = torch.cat([q_c, q_r], dim=-1)

    y = _naive_causal_softmax_attn(q, k, v).view(B, T, C)
    return F.linear(y, mod.c_proj.weight, mod.c_proj.bias)


NAIVE = {
    "mha":        naive_mha,
    "mha_rope":   naive_mha_rope,
    "mla_norope": naive_mla_norope,
    "mla":        naive_mla,
}


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def _make_attn(mode: str, n_embd=128, n_head=4, block_size=64,
               d_c=32, d_c_q=48, d_rope=16) -> CausalSelfAttention:
    cfg = GPTConfig(
        block_size=block_size,
        vocab_size=512,
        n_layer=1,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,
        bias=False,
        attn_mode=mode,
        mla_d_c=d_c,
        mla_d_c_q=d_c_q,
        mla_d_rope=d_rope,
    )
    return CausalSelfAttention(cfg).to(DEVICE).to(torch.float32)


def test_reference_match():
    print("Comparing production forward against naive reference (fp32)\n")
    B, T, C = 2, 32, 128

    for mode in ("mha", "mha_rope", "mla_norope", "mla"):
        attn = _make_attn(mode)
        attn.eval()
        # SDPA and the naive softmax disagree on what they do with dropout in train()
        # — both are dropout=0 in our config, but eval() removes that source of variance.

        x = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            y_prod  = attn(x)
            y_naive = NAIVE[mode](x, attn)

        diff = (y_prod - y_naive).abs().max().item()
        status = "OK " if diff < TOL_FP32 else "FAIL"
        print(f"  [{status}] {mode:12s}  max|diff| = {diff:.2e}   tol = {TOL_FP32:.0e}")
        assert diff < TOL_FP32, (
            f"{mode}: production vs naive diverge by {diff:.2e} > {TOL_FP32:.0e}"
        )


def test_rmsnorm_against_manual():
    """Sanity-check our RMSNorm against a hand-written one."""
    print("\nRMSNorm vs manual reference (fp32)")
    dim = 64
    norm = RMSNorm(dim).to(DEVICE).to(torch.float32)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(dim, device=DEVICE) * 0.5 + 1.0)
    x = torch.randn(4, 16, dim, device=DEVICE, dtype=torch.float32)

    y_mod = norm(x)
    rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
    y_ref = (x.float() * rms).to(x.dtype) * norm.weight

    diff = (y_mod - y_ref).abs().max().item()
    status = "OK " if diff < TOL_FP32_TIGHT else "FAIL"
    print(f"  [{status}] max|diff| = {diff:.2e}   tol = {TOL_FP32_TIGHT:.0e}")
    assert diff < TOL_FP32_TIGHT


def test_causal_mask_is_strict():
    """A non-causal forward must differ from the causal one for T >= 2."""
    print("\nCausal mask sanity (mha)")
    attn = _make_attn("mha")
    attn.eval()
    B, T, C = 1, 8, 128

    # Two inputs that differ only in the LAST token. With a correct causal mask,
    # the outputs at positions [0..T-2] must be IDENTICAL.
    x1 = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
    x2 = x1.clone()
    x2[:, -1] += torch.randn_like(x2[:, -1])

    with torch.no_grad():
        y1 = attn(x1)
        y2 = attn(x2)

    early_diff = (y1[:, :-1] - y2[:, :-1]).abs().max().item()
    late_diff  = (y1[:, -1]  - y2[:, -1]).abs().max().item()
    status = "OK " if early_diff < TOL_FP32_TIGHT and late_diff > 1e-3 else "FAIL"
    print(f"  [{status}] tokens [0..T-2] diff = {early_diff:.2e} (should be 0); "
          f"last-token diff = {late_diff:.2e} (should be > 0)")
    assert early_diff < TOL_FP32_TIGHT, "Causal mask leaking future info"
    assert late_diff > 1e-3, "Last-token output insensitive to last-token input — bug"


def test_rope_position_dependence():
    """Same token vector at different positions must produce different K/Q."""
    print("\nRoPE position dependence")
    dim, T = 32, 16
    cos, sin = _precompute_rope(dim, T, 10000.0)
    cos, sin = cos.to(DEVICE), sin.to(DEVICE)

    # Identical content at every position, distinct only via RoPE
    x = torch.ones(1, T, 2, dim, device=DEVICE, dtype=torch.float32)
    y = _apply_rope(x, cos, sin)

    diff_01 = (y[0, 0] - y[0, 1]).abs().max().item()
    diff_far = (y[0, 0] - y[0, T - 1]).abs().max().item()
    status = "OK " if diff_01 > 0.01 and diff_far > 0.1 else "FAIL"
    print(f"  [{status}] pos0-vs-pos1 diff = {diff_01:.3f}; pos0-vs-far diff = {diff_far:.3f}")
    assert diff_01 > 0.01
    assert diff_far > 0.1


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    test_reference_match()
    test_rmsnorm_against_manual()
    test_causal_mask_is_strict()
    test_rope_position_dependence()
    print("\nAll reference tests passed.")
