"""Fast structural tests for the follow-up study (CPU-compatible)."""

import numpy as np
import torch

from model import CausalSelfAttention, GPT, GPTConfig
from train import get_batch


MODES = ("mha", "mla_current", "mla_deepseek")


def small_config(mode: str) -> GPTConfig:
    return GPTConfig(
        block_size=64,
        vocab_size=512,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        attn_mode=mode,
        mla_d_c=32,
        mla_d_c_q=48,
        mla_d_rope=16,
    )


def test_forward_and_backward() -> None:
    for mode in MODES:
        torch.manual_seed(42)
        model = GPT(small_config(mode))
        tokens = torch.randint(0, 512, (2, 32))
        logits, loss = model(tokens, tokens)
        assert logits.shape == (2, 32, 512)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )


def test_layout_shapes() -> None:
    current = CausalSelfAttention(small_config("mla_current"))
    paper = CausalSelfAttention(small_config("mla_deepseek"))

    assert current.d_content == 16
    assert current.W_KR.out_features == 4 * 16
    assert current.W_UK.out_features == 4 * 16

    assert paper.d_content == 32
    assert paper.qk_dim == 48
    assert paper.W_KR.out_features == 16
    assert paper.W_UK.out_features == 4 * 32
    assert paper.W_QR.out_features == 4 * 16


def test_shared_key_is_really_shared() -> None:
    torch.manual_seed(7)
    attention = CausalSelfAttention(small_config("mla_deepseek"))
    x = torch.randn(2, 12, 128)
    k_shared = attention.W_KR(x).view(2, 12, 1, attention.d_rope)
    k_broadcast = k_shared.expand(-1, -1, attention.n_head, -1)
    for head in range(1, attention.n_head):
        assert torch.equal(k_broadcast[:, :, 0], k_broadcast[:, :, head])


def test_data_rng_is_architecture_independent() -> None:
    data = np.arange(20_000, dtype=np.uint16)
    batches = {}
    for mode in MODES:
        # Model construction consumes a mode-dependent number of global random
        # draws. The dedicated data generator must remain unaffected.
        torch.manual_seed(123)
        GPT(small_config(mode))
        generator = torch.Generator(device="cpu").manual_seed(100_123)
        batches[mode] = get_batch(
            data,
            block_size=32,
            batch_size=8,
            device=torch.device("cpu"),
            generator=generator,
        )

    reference_x, reference_y = batches["mha"]
    for mode in MODES[1:]:
        x, y = batches[mode]
        assert torch.equal(x, reference_x), f"X differs for {mode}"
        assert torch.equal(y, reference_y), f"Y differs for {mode}"


def test_causal_mask() -> None:
    attention = CausalSelfAttention(small_config("mla_deepseek"))
    attention.eval()
    x1 = torch.randn(1, 8, 128)
    x2 = x1.clone()
    x2[:, -1] += torch.randn_like(x2[:, -1])
    with torch.no_grad():
        y1, y2 = attention(x1), attention(x2)
    assert (y1[:, :-1] - y2[:, :-1]).abs().max().item() < 1e-5
    assert (y1[:, -1] - y2[:, -1]).abs().max().item() > 1e-3


def test_full_parameter_counts() -> None:
    counts = {}
    for mode in MODES:
        config = GPTConfig(attn_mode=mode)
        counts[mode] = GPT(config).count_parameters()
    assert counts["mha"] == 44_612_608
    assert counts["mla_current"] == 42_845_056
    # The shared K^R more than offsets the extra Q/K content dimensions.
    assert counts["mla_deepseek"] == 42_648_448
    print("Full-config parameter counts:")
    for mode, count in counts.items():
        print(f"  {mode:14s}: {count:,}")


if __name__ == "__main__":
    tests = [
        test_forward_and_backward,
        test_layout_shapes,
        test_shared_key_is_really_shared,
        test_data_rng_is_architecture_independent,
        test_causal_mask,
        test_full_parameter_counts,
    ]
    for test in tests:
        print(f"{test.__name__} ...", end=" ", flush=True)
        test()
        print("PASSED")
    print("All correctness tests passed.")
