"""Run one bf16 forward/backward pass for all eight production cells."""

import gc

import torch

from model import ATTENTION_MODES, BACKBONES, GPT, GPTConfig


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    for backbone in BACKBONES:
        for attention in ATTENTION_MODES:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            config = GPTConfig(
                block_size=128,
                attn_mode=attention,
                backbone=backbone,
                moe_dispatch="batched",
            )
            model = GPT(config).cuda().train()
            tokens = torch.randint(
                config.vocab_size, (2, config.block_size), device="cuda"
            )
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, cross_entropy, auxiliary, _ = model(tokens, tokens)
                assert cross_entropy is not None
                loss = cross_entropy + auxiliary
            loss.backward()
            assert torch.isfinite(loss)
            assert torch.isfinite(logits).all()
            peak = torch.cuda.max_memory_reserved() / 1024**3
            print(
                f"{backbone:5s}/{attention:16s}: "
                f"loss={float(loss.detach()):.4f}, peak={peak:.2f} GiB"
            )
            del model, tokens, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    print("All eight CUDA smoke-test cells passed.")


if __name__ == "__main__":
    main()
