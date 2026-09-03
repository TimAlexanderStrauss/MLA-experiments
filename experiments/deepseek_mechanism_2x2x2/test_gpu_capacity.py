"""Verify that compiled CUDA MoE dispatch aborts on insufficient capacity."""

import torch

from model import DeepSeekMoE, GPTConfig


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this capacity test")
    config = GPTConfig(
        block_size=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_embd=64,
        backbone="moe",
        n_shared_experts=2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        dense_intermediate_size=64,
        moe_capacity_factor=0.5,
    )
    module = DeepSeekMoE(config, 1).cuda().eval()
    with torch.no_grad():
        module.router_weight.zero_()
    compiled = torch.compile(
        module, mode="reduce-overhead", fullgraph=False, dynamic=False
    )
    inputs = torch.randn(1, 16, config.n_embd, device="cuda")
    try:
        compiled(inputs)
        torch.cuda.synchronize()
    except (AssertionError, RuntimeError) as error:
        message = str(error).lower()
        expected = (
            "capacity" in message
            or "assert" in message
            or "index out of bounds" in message
        )
        if not expected:
            raise
        print(f"Compiled CUDA capacity guard aborted as expected: {error}")
        return
    raise AssertionError("Compiled CUDA capacity guard did not abort")


if __name__ == "__main__":
    main()
