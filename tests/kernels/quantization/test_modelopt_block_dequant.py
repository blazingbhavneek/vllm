# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SM90 fused packed-block dequant kernels."""

import pytest
import torch

from vllm.model_executor.layers.quantization.modelopt_block_quant_hopper import (
    Q8_0_BLOCK_BYTES,
    block_linear,
    decode_blocks,
    decode_expert_gather,
    hopper_block_kernel_enabled,
)


def make_blocks(rows: int, blocks: int, codec: str) -> torch.Tensor:
    block_bytes = 66 if codec == "iq2_xxs" else Q8_0_BLOCK_BYTES
    packed = torch.randint(
        0, 256, (rows, blocks, block_bytes), dtype=torch.uint8, device="cuda"
    )
    packed[..., 0] = 0x3C
    packed[..., 1] = 0x00
    return packed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not hopper_block_kernel_enabled(), reason="requires SM90 and VLLM_SM90_BLOCK_KERNEL"
)
@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
def test_decode_bf16_perturbations(codec: str) -> None:
    packed = make_blocks(4, 6, codec)
    f32 = decode_blocks(packed, codec, torch.float32)
    bf16 = decode_blocks(packed, codec)
    assert not f32.isnan().any() and not f32.isinf().any()
    assert bf16.dtype == torch.bfloat16
    assert torch.isfinite(bf16).all()
    assert (bf16.float() - f32).abs().max() <= 0.125


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not hopper_block_kernel_enabled(), reason="requires SM90 and VLLM_SM90_BLOCK_KERNEL"
)
@pytest.mark.parametrize("codec", ["iq2_xxs", "q8_0"])
@pytest.mark.parametrize("m", [1, 16, 300])
def test_block_linear_matches_decoded_reference(codec: str, m: int) -> None:
    torch.manual_seed(m)
    weights = 256 if codec == "iq2_xxs" else 32
    n, blocks = 257, 3
    packed = make_blocks(n, blocks, codec)
    x = torch.randn(m, blocks * weights, dtype=torch.bfloat16, device="cuda") * 0.05

    got = block_linear(x, packed, codec)
    want = torch.nn.functional.linear(x, decode_blocks(packed, codec).reshape(n, -1))
    assert got.shape == want.shape
    rel = (got.float() - want.float()).abs().max() / want.float().abs().max()
    assert rel < 2e-2, f"codec={codec} m={m} rel={rel:.6g}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not hopper_block_kernel_enabled(), reason="requires SM90 and VLLM_SM90_BLOCK_KERNEL"
)
def test_expert_gather_matches_rows() -> None:
    torch.manual_seed(0)
    e, rows, blocks = 4, 3, 2
    ids = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    packed = make_blocks(e * rows, blocks, "iq2_xxs").reshape(e, rows, blocks, 66)
    got = decode_expert_gather(packed, ids, "iq2_xxs")
    flat = packed.reshape(-1, blocks, 66)
    want = decode_blocks(flat, "iq2_xxs").reshape(e, rows, -1)[ids]
    assert got.shape == want.shape
    assert (got != want).sum() <= want.numel() // 512


def test_unsupported_codec_rejected() -> None:
    packed = make_blocks(2, 2, "iq2_xxs")
    with pytest.raises(NotImplementedError):
        decode_blocks(packed, "iq2_xs")
