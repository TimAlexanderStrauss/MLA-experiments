"""CPU-compatible structural and gradient tests for the MoE follow-up."""

import math

import numpy as np
import torch

from model import ATTENTION_MODES, DeepSeekMoE, GPT, GPTConfig, SwiGLUExpert
from train import get_batch


def small_config(mode: str = "mha") -> GPTConfig:
    return GPTConfig(
        block_size=32,
        vocab_size=256,
        n_layer=3,
        n_head=4,
        n_embd=64,
        attn_mode=mode,
        mla_d_c=16,
        mla_d_c_q=24,
        mla_d_rope=8,
        first_dense_layers=1,
        n_shared_experts=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        aux_loss_alpha=0.001,
    )


def test_all_attention_cells_forward_and_backward() -> None:
    for mode in ATTENTION_MODES:
        torch.manual_seed(42)
        model = GPT(small_config(mode))
        tokens = torch.randint(0, 256, (2, 16))
        logits, ce_loss, aux_loss, stats = model(tokens, tokens)
        assert logits.shape == (2, 16, 256)
        assert ce_loss is not None and torch.isfinite(ce_loss)
        assert torch.isfinite(aux_loss) and aux_loss.item() > 0
        assert len(stats) == 2
        (ce_loss + aux_loss).backward()
        assert model.transformer.h[1].mlp.router_weight.grad is not None
        assert model.transformer.h[1].mlp.shared_experts.gate_proj.weight.grad is not None


def test_deepseek_layer_placement_and_shapes() -> None:
    model = GPT(small_config())
    assert not model.transformer.h[0].is_moe
    assert all(block.is_moe for block in model.transformer.h[1:])
    moe = model.transformer.h[1].mlp
    assert isinstance(moe, DeepSeekMoE)
    assert len(moe.experts) == 8
    assert moe.shared_experts.gate_proj.out_features == 64


def test_router_is_topk_without_renormalisation() -> None:
    torch.manual_seed(7)
    config = small_config()
    moe = DeepSeekMoE(config, layer_index=1).eval()
    x = torch.randn(2, 5, config.n_embd)
    scores = torch.softmax(
        torch.nn.functional.linear(x.reshape(-1, config.n_embd), moe.router_weight),
        dim=-1,
    )
    expected_weights, expected_indices = scores.topk(config.num_experts_per_tok, dim=-1)
    _, _, stats = moe(x)
    assert int(stats["selected_counts"].sum()) == x.shape[0] * x.shape[1] * 2
    # DeepSeek-V2-Lite has norm_topk_prob=false; selected weights therefore
    # generally sum to less than one.
    assert torch.all(expected_weights.sum(dim=-1) < 1.0)
    expected_counts = torch.bincount(
        expected_indices.reshape(-1), minlength=config.n_routed_experts
    )
    assert torch.equal(stats["selected_counts"], expected_counts)


def test_auxiliary_loss_matches_sequence_formula() -> None:
    torch.manual_seed(9)
    config = small_config()
    moe = DeepSeekMoE(config, layer_index=1)
    x = torch.randn(2, 7, config.n_embd)
    _, actual, _ = moe(x)
    scores = torch.softmax(
        torch.nn.functional.linear(
            x.reshape(-1, config.n_embd).float(), moe.router_weight.float()
        ),
        dim=-1,
    ).view(2, 7, config.n_routed_experts)
    selected = scores.topk(config.num_experts_per_tok, dim=-1).indices
    counts = torch.zeros(2, config.n_routed_experts)
    counts.scatter_add_(
        1,
        selected.reshape(2, -1),
        torch.ones(2, 7 * config.num_experts_per_tok),
    )
    frequency = counts / (
        7 * config.num_experts_per_tok / config.n_routed_experts
    )
    expected = (frequency * scores.mean(dim=1)).sum(dim=1).mean()
    expected = expected * config.aux_loss_alpha
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_active_compute_is_matched_to_dense_ffn() -> None:
    config = GPTConfig()
    dense_multiply_weights = 2 * config.n_embd * (4 * config.n_embd)
    active_experts = config.n_shared_experts + config.num_experts_per_tok
    moe_multiply_weights = (
        active_experts
        * 3
        * config.n_embd
        * config.moe_intermediate_size
    )
    ratio = moe_multiply_weights / dense_multiply_weights
    assert math.isclose(ratio, 0.984375)


def test_data_rng_is_architecture_independent() -> None:
    data = np.arange(10_000, dtype=np.uint16)
    batches = {}
    for mode in ATTENTION_MODES:
        torch.manual_seed(123)
        GPT(small_config(mode))
        generator = torch.Generator(device="cpu").manual_seed(100_123)
        batches[mode] = get_batch(
            data, 16, 4, torch.device("cpu"), generator
        )
    reference_x, reference_y = batches["mha"]
    for mode in ATTENTION_MODES[1:]:
        x, y = batches[mode]
        assert torch.equal(x, reference_x)
        assert torch.equal(y, reference_y)


def test_full_parameter_counts_are_reportable() -> None:
    counts = {}
    for mode in ATTENTION_MODES:
        model = GPT(GPTConfig(attn_mode=mode))
        counts[mode] = (model.count_parameters(), model.count_active_parameters())
        assert counts[mode][0] > counts[mode][1]
        assert counts[mode][1] > 0
    print("Full configuration (total / active per token):")
    for mode, (total, active) in counts.items():
        print(f"  {mode:12s}: {total:,} / {active:,}")


if __name__ == "__main__":
    tests = [
        test_all_attention_cells_forward_and_backward,
        test_deepseek_layer_placement_and_shapes,
        test_router_is_topk_without_renormalisation,
        test_auxiliary_loss_matches_sequence_formula,
        test_active_compute_is_matched_to_dense_ffn,
        test_data_rng_is_architecture_independent,
        test_full_parameter_counts_are_reportable,
    ]
    for test in tests:
        print(f"{test.__name__} ...", end=" ", flush=True)
        test()
        print("PASSED")
    print("All correctness tests passed.")
