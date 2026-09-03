"""Structural, gradient, routing, and design tests."""

import argparse
import copy
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import (
    ATTENTION_FACTORS,
    ATTENTION_MODES,
    BACKBONES,
    DeepSeekMoE,
    GPT,
    GPTConfig,
)
from benchmark_gpu import ARCHITECTURE as BENCHMARK_ARCHITECTURE
from run_experiments import ARCHITECTURE as RUN_ARCHITECTURE, balanced_plan
from train import get_batch, save_checkpoint
from analyze_results import compute_factorial_effects


def small_config(
    attention: str = "mha", backbone: str = "dense", dispatch: str = "batched"
) -> GPTConfig:
    return GPTConfig(
        block_size=32,
        vocab_size=128,
        n_layer=3,
        n_head=4,
        n_embd=64,
        attn_mode=attention,
        backbone=backbone,
        mla_d_c=16,
        mla_d_rope=8,
        first_dense_layers=1,
        n_shared_experts=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        dense_intermediate_size=64,
        moe_dispatch=dispatch,
    )


def test_all_eight_cells_forward_and_backward() -> None:
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            torch.manual_seed(42)
            model = GPT(small_config(attention, backbone))
            tokens = torch.randint(0, 128, (2, 16))
            logits, ce, auxiliary, stats = model(
                tokens, tokens, collect_routing_stats=True
            )
            assert logits.shape == (2, 16, 128)
            assert ce is not None and torch.isfinite(ce)
            if backbone == "moe":
                assert auxiliary.item() > 0 and len(stats) == 2
            else:
                assert auxiliary.item() == 0 and not stats
            (ce + auxiliary).backward()


def test_attention_factors_and_deepseek_layout() -> None:
    assert set(ATTENTION_FACTORS.values()) == {
        (False, False), (False, True), (True, False), (True, True)
    }
    for mode in ATTENTION_MODES:
        attention = GPT(small_config(mode)).transformer.h[0].attn
        assert not hasattr(attention, "W_DQ")
        if ATTENTION_FACTORS[mode][1]:
            assert attention.W_KR.out_features == attention.d_rope
    expected_without_recompute = {
        "mha": 128,
        "mha_decoupled": 136,
        "mla_coupled": 128,
        "mla_decoupled": 24,
    }
    expected_with_recompute = {
        "mha": 128,
        "mha_decoupled": 136,
        "mla_coupled": 16,
        "mla_decoupled": 24,
    }
    for mode in ATTENTION_MODES:
        attention = GPT(small_config(mode)).transformer.h[0].attn
        assert (
            attention.logical_kv_cache_elements_per_token()
            == expected_without_recompute[mode]
        )
        assert (
            attention.logical_kv_cache_elements_with_recompute_per_token()
            == expected_with_recompute[mode]
        )


def test_backbone_placement_and_active_compute() -> None:
    dense = GPT(small_config(backbone="dense"))
    moe_model = GPT(small_config(backbone="moe"))
    assert not any(block.is_moe for block in dense.transformer.h)
    assert not moe_model.transformer.h[0].is_moe
    assert all(block.is_moe for block in moe_model.transformer.h[1:])
    config = GPTConfig()
    dense_ops = 3 * config.n_embd * config.dense_intermediate_size
    active_moe_ops = (
        3
        * config.n_embd
        * config.moe_intermediate_size
        * (config.n_shared_experts + config.num_experts_per_tok)
    )
    assert dense_ops == active_moe_ops


def test_batched_dispatch_matches_loop_and_gradients() -> None:
    torch.manual_seed(7)
    loop = DeepSeekMoE(small_config(backbone="moe", dispatch="loop"), 1)
    batched = DeepSeekMoE(small_config(backbone="moe", dispatch="batched"), 1)
    batched.load_state_dict(copy.deepcopy(loop.state_dict()))
    x_loop = torch.randn(2, 9, 64, requires_grad=True)
    x_batched = x_loop.detach().clone().requires_grad_(True)
    output_loop, auxiliary_loop, stats_loop = loop(x_loop, True)
    output_batched, auxiliary_batched, stats_batched = batched(x_batched, True)
    assert torch.allclose(output_loop, output_batched, atol=2e-6, rtol=2e-5)
    assert torch.allclose(auxiliary_loop, auxiliary_batched, atol=1e-8)
    assert torch.equal(stats_loop["selected_counts"], stats_batched["selected_counts"])
    (output_loop.square().mean() + auxiliary_loop).backward()
    (output_batched.square().mean() + auxiliary_batched).backward()
    assert torch.allclose(x_loop.grad, x_batched.grad, atol=2e-6, rtol=2e-5)
    for name in ("expert_gate", "expert_up", "expert_down", "router_weight"):
        assert torch.allclose(
            getattr(loop, name).grad,
            getattr(batched, name).grad,
            atol=2e-6,
            rtol=2e-5,
        )


def test_compiled_capacity_guard_aborts() -> None:
    if not hasattr(torch, "compile"):
        return
    config = small_config(backbone="moe")
    config.moe_capacity_factor = 0.5
    moe = DeepSeekMoE(config, 1).eval()
    with torch.no_grad():
        moe.router_weight.zero_()
    x = torch.randn(1, 16, config.n_embd)
    eager_output, _, _ = moe(x)
    assert eager_output.shape == x.shape
    compiled = torch.compile(
        moe, backend="eager", fullgraph=False, dynamic=False
    )
    try:
        compiled(x)
    except AssertionError as error:
        assert "compiled capacity" in str(error)
    else:
        raise AssertionError("Compiled MoE capacity guard did not abort")


def test_router_formula_and_no_topk_renormalization() -> None:
    torch.manual_seed(9)
    config = small_config(backbone="moe")
    moe = DeepSeekMoE(config, 1)
    x = torch.randn(2, 7, config.n_embd)
    _, actual, _ = moe(x)
    scores = torch.softmax(
        torch.nn.functional.linear(x.reshape(-1, config.n_embd), moe.router_weight),
        dim=-1,
    ).view(2, 7, config.n_routed_experts)
    weights, selected = scores.topk(config.num_experts_per_tok, dim=-1)
    assert torch.all(weights.sum(dim=-1) < 1.0)
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


def test_data_rng_is_cell_independent() -> None:
    data = np.arange(10_000, dtype=np.uint16)
    batches = []
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            torch.manual_seed(123)
            GPT(small_config(attention, backbone))
            generator = torch.Generator(device="cpu").manual_seed(100_123)
            batches.append(get_batch(data, 16, 4, torch.device("cpu"), generator))
    reference_x, reference_y = batches[0]
    for x, y in batches[1:]:
        assert torch.equal(x, reference_x)
        assert torch.equal(y, reference_y)


def test_run_plan_is_complete_and_balanced() -> None:
    assert BENCHMARK_ARCHITECTURE == RUN_ARCHITECTURE
    plan = balanced_plan()
    assert len(plan) == 24
    assert len(set(plan)) == 24
    assert set(plan) == {
        (backbone, attention, seed)
        for backbone in BACKBONES
        for attention in ATTENTION_MODES
        for seed in (42, 123, 456)
    }


def test_factorial_contrast_definitions() -> None:
    rows = []
    for seed in (42, 123, 456):
        for backbone in BACKBONES:
            for attention, (low_rank, decoupled) in ATTENTION_FACTORS.items():
                rows.append(
                    {
                        "seed": seed,
                        "backbone": backbone,
                        "attn_mode": attention,
                        "low_rank": low_rank,
                        "decoupled_rope": decoupled,
                        "final_val_loss": 3.0 + 0.1 * low_rank + 0.001 * seed,
                    }
                )
    effects = compute_factorial_effects(pd.DataFrame(rows)).set_index("effect")
    assert math.isclose(
        effects.loc["Low-rank KV", "mean_difference"], 0.1, abs_tol=1e-12
    )
    for effect in effects.index.drop("Low-rank KV"):
        assert math.isclose(
            effects.loc[effect, "mean_difference"], 0.0, abs_tol=1e-12
        )


def test_end_to_end_causality() -> None:
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            torch.manual_seed(5)
            model = GPT(small_config(attention, backbone)).eval()
            tokens = torch.randint(0, 128, (1, 16))
            changed = tokens.clone()
            changed[:, 10:] = torch.randint(0, 128, (1, 6))
            with torch.no_grad():
                original_logits, _, _, _ = model(tokens, tokens)
                changed_logits, _, _, _ = model(changed, changed)
            assert torch.allclose(
                original_logits[:, :10],
                changed_logits[:, :10],
                atol=1e-6,
                rtol=1e-6,
            )


def tiny_training_config(attention: str, backbone: str) -> GPTConfig:
    return GPTConfig(
        block_size=16,
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_embd=32,
        attn_mode=attention,
        backbone=backbone,
        mla_d_c=16,
        mla_d_rope=4,
        first_dense_layers=1,
        n_shared_experts=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        dense_intermediate_size=32,
    )


def _optimization_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    data: np.ndarray,
    generator: torch.Generator,
) -> float:
    x, y = get_batch(data, 8, 2, torch.device("cpu"), generator)
    optimizer.zero_grad(set_to_none=True)
    _, cross_entropy, auxiliary_loss, _ = model(x, y)
    assert cross_entropy is not None
    loss = cross_entropy + auxiliary_loss
    loss.backward()
    optimizer.step()
    return float(cross_entropy.detach())


def test_loss_decreases() -> None:
    tokens = np.tile(np.arange(32, dtype=np.uint16), 128)
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            torch.manual_seed(17)
            model = GPT(tiny_training_config(attention, backbone))
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
            generator = torch.Generator(device="cpu").manual_seed(23)
            fixed_generator = torch.Generator(device="cpu").manual_seed(91)
            fixed_x, fixed_y = get_batch(
                tokens, 8, 4, torch.device("cpu"), fixed_generator
            )
            with torch.no_grad():
                _, before, _, _ = model(fixed_x, fixed_y)
            for _ in range(12):
                _optimization_step(model, optimizer, tokens, generator)
            with torch.no_grad():
                _, after, _, _ = model(fixed_x, fixed_y)
            assert before is not None and after is not None
            assert after < before, (
                f"{backbone}/{attention}: loss did not decrease "
                f"({float(before):.4f} -> {float(after):.4f})"
            )


def test_checkpoint_resume_equivalence() -> None:
    data = np.tile(np.arange(32, dtype=np.uint16), 128)
    config = tiny_training_config("mla_decoupled", "moe")

    torch.manual_seed(101)
    uninterrupted = GPT(config)
    uninterrupted_optimizer = torch.optim.AdamW(
        uninterrupted.parameters(), lr=0.005
    )
    uninterrupted_generator = torch.Generator(device="cpu").manual_seed(202)

    torch.manual_seed(101)
    interrupted = GPT(config)
    interrupted_optimizer = torch.optim.AdamW(
        interrupted.parameters(), lr=0.005
    )
    interrupted_generator = torch.Generator(device="cpu").manual_seed(202)
    validation_generator = torch.Generator(device="cpu").manual_seed(303)

    for _ in range(4):
        _optimization_step(
            uninterrupted,
            uninterrupted_optimizer,
            data,
            uninterrupted_generator,
        )
    for _ in range(2):
        _optimization_step(
            interrupted,
            interrupted_optimizer,
            data,
            interrupted_generator,
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
        save_checkpoint(
            checkpoint_path,
            interrupted,
            interrupted_optimizer,
            1,
            argparse.Namespace(test="resume"),
            interrupted_generator,
            validation_generator,
            12.5,
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        assert checkpoint["elapsed_s"] == 12.5
        resumed = GPT(config)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.005)
        resumed.load_state_dict(checkpoint["model"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        resumed_generator = torch.Generator(device="cpu")
        resumed_generator.set_state(checkpoint["train_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        for _ in range(2):
            _optimization_step(
                resumed,
                resumed_optimizer,
                data,
                resumed_generator,
            )

    for expected, actual in zip(
        uninterrupted.parameters(), resumed.parameters()
    ):
        assert torch.equal(expected, actual)
    expected_batch = get_batch(
        data, 8, 2, torch.device("cpu"), uninterrupted_generator
    )
    actual_batch = get_batch(
        data, 8, 2, torch.device("cpu"), resumed_generator
    )
    assert torch.equal(expected_batch[0], actual_batch[0])
    assert torch.equal(expected_batch[1], actual_batch[1])


def test_full_configuration_parameter_spread() -> None:
    active_counts = []
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            model = GPT(GPTConfig(backbone=backbone, attn_mode=attention))
            active_counts.append(model.count_active_parameters())
    assert max(active_counts) / min(active_counts) < 1.05


def test_full_configuration_is_reportable() -> None:
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            model = GPT(GPTConfig(backbone=backbone, attn_mode=attention))
            assert model.count_parameters() >= model.count_active_parameters() > 0
            assert model.logical_kv_cache_elements_per_token() > 0
            print(
                f"{backbone:5s}/{attention:16s}: "
                f"total={model.count_parameters():,} "
                f"active={model.count_active_parameters():,} "
                f"cache={model.logical_kv_cache_elements_per_token():,}"
            )


def main() -> None:
    tests = [
        test_all_eight_cells_forward_and_backward,
        test_attention_factors_and_deepseek_layout,
        test_backbone_placement_and_active_compute,
        test_batched_dispatch_matches_loop_and_gradients,
        test_compiled_capacity_guard_aborts,
        test_router_formula_and_no_topk_renormalization,
        test_data_rng_is_cell_independent,
        test_run_plan_is_complete_and_balanced,
        test_factorial_contrast_definitions,
        test_end_to_end_causality,
        test_loss_decreases,
        test_checkpoint_resume_equivalence,
        test_full_configuration_parameter_spread,
        test_full_configuration_is_reportable,
    ]
    for test in tests:
        print(f"{test.__name__} ...", end=" ", flush=True)
        test()
        print("PASSED")
    print("All correctness tests passed.")


if __name__ == "__main__":
    main()
