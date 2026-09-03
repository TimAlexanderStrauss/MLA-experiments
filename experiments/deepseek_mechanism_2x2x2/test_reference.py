"""Independent numerical reference tests for the four attention cells."""

import math

import torch
import torch.nn.functional as F

from model import ATTENTION_MODES, CausalSelfAttention, GPTConfig


TOLERANCE = 1e-4


def reference_apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply adjacent-pair RoPE without using the production helper."""
    sequence_length = x.shape[1]
    result = torch.empty_like(x)
    for position in range(sequence_length):
        for pair in range(x.shape[-1] // 2):
            left = x[:, position, :, 2 * pair]
            right = x[:, position, :, 2 * pair + 1]
            cosine = cos[position, pair]
            sine = sin[position, pair]
            result[:, position, :, 2 * pair] = (
                left * cosine - right * sine
            )
            result[:, position, :, 2 * pair + 1] = (
                left * sine + right * cosine
            )
    return result


def make_attention(mode: str) -> CausalSelfAttention:
    return CausalSelfAttention(
        GPTConfig(
            block_size=32,
            vocab_size=128,
            n_layer=1,
            n_head=4,
            n_embd=128,
            attn_mode=mode,
            mla_d_c=32,
            mla_d_rope=16,
        )
    ).float().eval()


def manual_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    _, sequence_length, _, qk_dim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(qk_dim)
    mask = torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(mask, float("-inf"))
    return (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous()


def reference_forward(
    x: torch.Tensor, module: CausalSelfAttention
) -> torch.Tensor:
    batch_size, sequence_length, channels = x.shape
    shape = (batch_size, sequence_length, module.n_head, module.d_h)
    if not module.low_rank:
        q_content, k_content, v = F.linear(
            x, module.c_attn.weight, module.c_attn.bias
        ).split(channels, dim=-1)
        q_content = q_content.view(shape)
        k_content = k_content.view(shape)
        v = v.view(shape)
    else:
        c_kv = module.kv_norm(F.linear(x, module.W_DKV.weight))
        k_content = F.linear(c_kv, module.W_UK.weight).view(shape)
        v = F.linear(c_kv, module.W_UV.weight).view(shape)
        q_projected = F.linear(x, module.W_Q.weight, module.W_Q.bias)
        if module.decoupled:
            q_content_flat, q_rope_flat = q_projected.split(
                [channels, module.n_head * module.d_rope], dim=-1
            )
            q_content = q_content_flat.view(shape)
        else:
            q_content = q_projected.view(shape)

    if module.decoupled:
        if not module.low_rank:
            q_rope_flat = F.linear(x, module.W_QR.weight, module.W_QR.bias)
        q_rope = reference_apply_rope(
            q_rope_flat.view(
                batch_size, sequence_length, module.n_head, module.d_rope
            ),
            module.rope_cos,
            module.rope_sin,
        )
        k_shared = reference_apply_rope(
            F.linear(x, module.W_KR.weight).view(
                batch_size, sequence_length, 1, module.d_rope
            ),
            module.rope_cos,
            module.rope_sin,
        )
        q = torch.cat((q_content, q_rope), dim=-1)
        k = torch.cat(
            (k_content, k_shared.expand(-1, -1, module.n_head, -1)), dim=-1
        )
    else:
        q = reference_apply_rope(q_content, module.rope_cos, module.rope_sin)
        k = reference_apply_rope(k_content, module.rope_cos, module.rope_sin)
    output = manual_attention(q, k, v).view(
        batch_size, sequence_length, channels
    )
    return F.linear(output, module.c_proj.weight, module.c_proj.bias)


def test_rope_is_position_dependent() -> None:
    vector = torch.tensor([1.0, 2.0] * 8).view(1, 1, 1, 16)
    repeated = vector.expand(1, 4, 1, 16).clone()
    module = make_attention("mha_decoupled")
    rotated = reference_apply_rope(
        repeated, module.rope_cos, module.rope_sin
    )
    assert torch.equal(rotated[:, 0], repeated[:, 0])
    assert not torch.allclose(rotated[:, 0], rotated[:, 3])


def main() -> None:
    torch.manual_seed(0)
    test_rope_is_position_dependent()
    print("Independent RoPE position test passed.")
    x = torch.randn(2, 16, 128)
    for mode in ATTENTION_MODES:
        module = make_attention(mode)
        with torch.no_grad():
            production = module(x)
            reference = reference_forward(x, module)
        difference = (production - reference).abs().max().item()
        print(f"{mode:16s}: max|diff|={difference:.3e}")
        assert difference < TOLERANCE
    print("All numerical attention reference tests passed.")


if __name__ == "__main__":
    main()
