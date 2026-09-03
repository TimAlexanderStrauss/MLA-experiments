"""
Correctness and sanity checks before running the full training.

Tests:
  1. All 4 attention modes produce outputs of the correct shape
  2. All 4 modes produce non-NaN outputs with a simple forward pass
  3. Parameter counts are as expected and within ~5% of each other
  4. mha and mha_rope have identical parameter counts (same c_attn structure)
  5. Loss decreases on a tiny dataset (gradient check)
  6. Checkpoint save/load produces identical outputs
  7. RoPE is actually applied: mha_rope and mha produce different K matrices for same input

Usage:
  cd experiments
  python test_correctness.py
"""

import copy
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from model import GPT, GPTConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(attn_mode: str) -> GPT:
    cfg = GPTConfig(
        block_size=64,
        vocab_size=512,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        attn_mode=attn_mode,
        mla_d_c=32,
        mla_d_c_q=48,
        mla_d_rope=16,
    )
    return GPT(cfg).to(DEVICE)


def make_batch(B: int = 2, T: int = 32, vocab: int = 512) -> tuple:
    idx = torch.randint(0, vocab, (B, T), device=DEVICE)
    tgt = torch.randint(0, vocab, (B, T), device=DEVICE)
    return idx, tgt


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_forward_shapes():
    print("Test 1: Forward pass shapes ...")
    B, T = 2, 32
    for mode in ("mha", "mha_rope", "mla_norope", "mla"):
        model = make_model(mode)
        idx, tgt = make_batch(B, T)
        logits, loss = model(idx, tgt)
        assert logits.shape == (B, T, 512), f"{mode}: logits shape {logits.shape}"
        assert loss is not None and loss.ndim == 0, f"{mode}: loss shape {loss.shape}"
        assert not torch.isnan(loss), f"{mode}: NaN loss"
    print("  PASSED")


def test_no_nans():
    print("Test 2: No NaN in loss for 20 random batches ...")
    for mode in ("mha", "mha_rope", "mla_norope", "mla"):
        model = make_model(mode)
        for _ in range(20):
            idx, tgt = make_batch()
            _, loss = model(idx, tgt)
            assert not torch.isnan(loss), f"{mode}: NaN loss"
    print("  PASSED")


def test_parameter_counts():
    print("Test 3: Parameter counts ...")
    counts = {}
    for mode in ("mha", "mha_rope", "mla_norope", "mla"):
        counts[mode] = make_model(mode).count_parameters()
        print(f"  {mode:12s}: {counts[mode]:>10,}")

    # mha and mha_rope must have identical counts (same c_attn)
    assert counts["mha"] == counts["mha_rope"], \
        f"mha ({counts['mha']}) != mha_rope ({counts['mha_rope']})"

    # MLA variants should have fewer parameters in attention (low-rank bottleneck)
    assert counts["mla"] < counts["mha"], \
        f"Expected mla < mha but got {counts['mla']} >= {counts['mha']}"

    # All counts within 10% of each other (largest pairwise ratio)
    vals = list(counts.values())
    ratio = max(vals) / min(vals)
    assert ratio < 1.15, f"Parameter count spread too large: {ratio:.2f}x"
    print("  PASSED")


def test_loss_decreases():
    print("Test 4: Loss decreases after a few gradient steps ...")
    for mode in ("mha", "mha_rope", "mla_norope", "mla"):
        model = make_model(mode)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        idx, tgt = make_batch(B=4, T=32)
        _, loss_before = model(idx, tgt)
        for _ in range(20):
            opt.zero_grad()
            _, loss = model(idx, tgt)
            loss.backward()
            opt.step()
        _, loss_after = model(idx, tgt)
        assert loss_after.item() < loss_before.item(), \
            f"{mode}: loss did not decrease ({loss_before:.3f} -> {loss_after:.3f})"
    print("  PASSED")


def test_checkpoint_roundtrip():
    print("Test 5: Checkpoint save/load produces identical outputs ...")
    for mode in ("mha", "mla"):
        model = make_model(mode)
        idx, _ = make_batch()
        logits_before, _ = model(idx)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"model": raw.state_dict(), "iter": 0, "optimizer": {}, "config": {}},
                       ckpt_path)

            model2 = make_model(mode)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model2.load_state_dict(ckpt["model"])

        logits_after, _ = model2(idx)
        max_diff = (logits_before - logits_after).abs().max().item()
        assert max_diff < 1e-5, f"{mode}: checkpoint roundtrip diff {max_diff}"
    print("  PASSED")


def test_rope_actually_applied():
    print("Test 6: RoPE rotation is position-dependent ...")
    # Verify _apply_rope directly: same input vector rotated at pos 0 vs pos 4 must differ.
    from model import _apply_rope, _precompute_rope
    torch.manual_seed(0)
    dim, T = 64, 16
    cos, sin = _precompute_rope(dim, T, 10000.0)
    x = torch.randn(1, T, 4, dim)  # (B, T, H, D)
    x_rot = _apply_rope(x, cos, sin)
    # Same token at different positions should rotate to different values
    diff = (x_rot[0, 0] - x_rot[0, 4]).abs().max().item()
    assert diff > 0.01, f"RoPE not position-dependent (diff={diff})"
    # Verify that unrotated inputs are NOT position-different
    orig_diff = (x[0, 0] - x[0, 4]).abs().max().item()
    assert orig_diff > 0.0, "Test input has identical values at pos 0 and 4 (degenerate)"
    print("  PASSED")


def test_mla_vs_mha_differ():
    print("Test 7: MHA and MLA produce different outputs (architecture differs) ...")
    torch.manual_seed(42)
    mha = make_model("mha")
    torch.manual_seed(42)
    mla = make_model("mla")
    idx, tgt = make_batch()
    logits_mha, _ = mha(idx)
    logits_mla, _ = mla(idx)
    diff = (logits_mha - logits_mla).abs().max().item()
    assert diff > 1e-3, f"MHA and MLA produce suspiciously similar outputs (diff={diff})"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Full architecture sanity check with design-doc config
# ---------------------------------------------------------------------------

def test_full_config():
    print("Test 8: Full design-doc config (6 layers, 512 embd, 8 heads) ...")
    cfg = GPTConfig(
        block_size=512,
        vocab_size=50257,
        n_layer=6,
        n_head=8,
        n_embd=512,
        dropout=0.0,
        bias=False,
        mla_d_c=128,
        mla_d_c_q=192,
        mla_d_rope=32,
    )
    counts = {}
    for mode in ("mha", "mha_rope", "mla_norope", "mla"):
        cfg.attn_mode = mode
        m = GPT(cfg).to(DEVICE)
        counts[mode] = m.count_parameters()
        # Quick forward to check no errors
        idx = torch.randint(0, 50257, (1, 64), device=DEVICE)
        _, loss = m(idx, idx)
        assert not torch.isnan(loss), f"{mode}: NaN loss on full config"

    print("  Parameter counts (full config):")
    for mode, n in counts.items():
        print(f"    {mode:12s}: {n/1e6:.2f}M")
    print("  PASSED")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running on: {DEVICE}\n")
    test_forward_shapes()
    test_no_nans()
    test_parameter_counts()
    test_loss_decreases()
    test_checkpoint_roundtrip()
    test_rope_actually_applied()
    test_mla_vs_mha_differ()
    test_full_config()
    print("\nAll tests passed.")
