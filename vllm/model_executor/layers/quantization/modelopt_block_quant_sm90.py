# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 execution fallback for ModelOpt packed block codecs.

b12x only ships kernels for SM120/SM121. On Hopper the packed GGML payloads
are decoded transiently into a reusable scratch buffer just before the GEMM
and never persist in VRAM at rest, so the checkpoint keeps its 2-bit
residency. This is a correctness-first path: the same kernels used for
unquantized BF16 MoE/linear layers consume the decoded tiles.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

try:
    from vllm.model_executor.layers.quantization.modelopt_block_quant_hopper import (
        prewarm_device_checks,
    )
except ImportError:  # pragma: no cover - optional triton dependency

    def prewarm_device_checks() -> None:
        return None

_IQ2_BLOCK_WEIGHTS = 256
_Q8_0_BLOCK_WEIGHTS = 32
# Bound the temporary decode footprint of one MoE expert chunk.
_MOE_CHUNK_BYTES = 2 << 30
# Decode packed blocks in slices so the fp32 intermediate stays small.
_BLOCK_SLICE = 1 << 17

_IQ2_LUT_CACHE: dict[tuple[str, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_SM90_FALLBACK_CACHE: bool | None = None


def _hopper():
    if not sm90_fallback_enabled():
        return None
    from vllm.model_executor.layers.quantization import (
        modelopt_block_quant_hopper as hopper,
    )

    return hopper if hopper.hopper_block_kernel_enabled() else None


def sm90_fallback_enabled() -> bool:
    """Whether the current device needs the packed-block fallback."""
    global _SM90_FALLBACK_CACHE
    if _SM90_FALLBACK_CACHE is not None:
        return _SM90_FALLBACK_CACHE
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    _SM90_FALLBACK_CACHE = capability is not None and capability[0] == 9
    return _SM90_FALLBACK_CACHE


def _iq2_luts(codec: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (codec, device)
    cached = _IQ2_LUT_CACHE.get(key)
    if cached is not None:
        return cached

    from b12x._lib.quant.iq2_xs import iq2_xs_grid_cpu

    grid = iq2_xs_grid_cpu(codec).to(device=device, dtype=torch.float32)
    masks = torch.arange(256, device=device, dtype=torch.int64)
    bits = torch.arange(8, device=device, dtype=torch.int64)
    signs = (1 - 2 * ((masks[:, None] >> bits[None, :]) & 1)).to(torch.float32)
    _IQ2_LUT_CACHE[key] = (grid, signs)
    return grid, signs


def _iq2_xxs_payload(
    blocks: torch.Tensor, grid: torch.Tensor, signs: torch.Tensor
) -> torch.Tensor:
    base = blocks[:, :2].contiguous().view(torch.float16).float().reshape(-1)
    payload = blocks[:, 2:].reshape(-1, 8, 8)
    grid_idx = payload[..., :4].to(torch.int64)
    control_bytes = payload[..., 4:8].to(torch.int64)
    control = (
        control_bytes[..., 0]
        | (control_bytes[..., 1] << 8)
        | (control_bytes[..., 2] << 16)
        | (control_bytes[..., 3] << 24)
    )
    scale = base[:, None] * ((control >> 28).float() + 0.5) * 0.25

    groups = []
    for group in range(4):
        values = grid[grid_idx[..., group]]
        sign_index = (control >> (7 * group)) & 127
        parity = torch.zeros_like(sign_index)
        for bit in range(7):
            parity ^= (sign_index >> bit) & 1
        mask = sign_index | (parity << 7)
        values = values * signs[mask] * scale[:, :, None]
        groups.append(values)

    return torch.stack(groups, dim=-2).reshape(-1, _IQ2_BLOCK_WEIGHTS)


def _dequantize_blocks(
    blocks: torch.Tensor, codec: str, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    block_weights = _Q8_0_BLOCK_WEIGHTS if codec == "q8_0" else _IQ2_BLOCK_WEIGHTS
    block_bytes = blocks.shape[-1]
    flat = blocks.reshape(-1, block_bytes)
    out = torch.empty(
        (flat.shape[0], block_weights), dtype=out_dtype, device=blocks.device
    )

    if codec == "q8_0":
        for start in range(0, flat.shape[0], _BLOCK_SLICE):
            stop = min(start + _BLOCK_SLICE, flat.shape[0])
            scales = (
                flat[start:stop, :2]
                .contiguous()
                .view(torch.float16)
                .float()
                .reshape(-1, 1)
            )
            values = flat[start:stop, 2:].contiguous().view(torch.int8).float()
            out[start:stop] = (values * scales).to(out_dtype)
    elif codec == "iq2_xxs":
        grid, signs = _iq2_luts(codec, blocks.device)
        for start in range(0, flat.shape[0], _BLOCK_SLICE):
            stop = min(start + _BLOCK_SLICE, flat.shape[0])
            decoded = _iq2_xxs_payload(flat[start:stop], grid, signs)
            out[start:stop] = decoded.to(out_dtype)
    else:
        raise NotImplementedError(f"SM90 block fallback does not support {codec}")

    return out.reshape(*blocks.shape[:-2], -1)


def dequantize_linear_weight(
    weight: torch.Tensor, codec: str, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Decode packed ``[N, K/blocks, bytes]`` weights into ``[N, K]``."""
    hopper = _hopper()
    if hopper is not None:
        return hopper.decode_blocks(weight.detach(), codec, out_dtype)
    return _dequantize_blocks(weight.detach(), codec, out_dtype)


def apply_linear_fallback(
    layer, x: torch.Tensor, bias: torch.Tensor | None, codec: str
) -> torch.Tensor:
    weight = layer.weight.data
    hopper = _hopper()
    if hopper is not None and x.dtype == torch.bfloat16:
        out = hopper.block_linear(x, weight, codec)
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], weight.shape[0])
    dense = dequantize_linear_weight(weight, codec)
    out = F.linear(x.reshape(-1, dense.shape[1]), dense, None)
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], dense.shape[0])


def apply_embedding_fallback(
    input_: torch.Tensor, weight: torch.Tensor, codec: str
) -> torch.Tensor:
    table = dequantize_linear_weight(weight.data, codec)
    return F.embedding(input_, table)


def _expert_chunk_count(w13: torch.Tensor, w2: torch.Tensor) -> int:
    per_expert = (
        (w13.shape[1] * w13.shape[2] + w2.shape[1] * w2.shape[2])
        * _IQ2_BLOCK_WEIGHTS
        * 2
    )
    return max(1, _MOE_CHUNK_BYTES // max(per_expert, 1))


def apply_moe_fallback(
    layer,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    codec: str,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    w13_packed = layer.w13_weight.data
    w2_packed = layer.w2_weight.data
    global_num_experts = layer.global_num_experts

    activation = MoEActivation(layer.activation)
    if (
        _hopper() is not None
        and codec == "iq2_xxs"
        and activation == MoEActivation.RELU2_NO_MUL
        and not layer.apply_router_weight_on_input
        and layer.expert_map is None
        and global_num_experts == w13_packed.shape[0]
    ):
        return _hopper().moe_forward(
            x,
            w13_packed,
            w2_packed,
            topk_weights,
            topk_ids,
        )

    chunk = _expert_chunk_count(w13_packed, w2_packed)

    # Decoding all experts on every token would move far more data than the
    # model actually touches. Only materialize the experts selected by this
    # batch, and chunk that selection to bound the scratch footprint.
    active = torch.unique(topk_ids.reshape(-1))
    if active.numel() == 0:
        return torch.zeros_like(x)

    hidden_states = x.contiguous()
    output = torch.zeros_like(hidden_states)
    hopper = _hopper()
    for start in range(0, active.numel(), chunk):
        idx = active[start : start + chunk]
        if hopper is not None:
            w13 = hopper.decode_expert_gather(w13_packed, idx, codec)
            w2 = hopper.decode_expert_gather(w2_packed, idx, codec)
        else:
            w13 = dequantize_linear_weight(w13_packed[idx], codec)
            w2 = dequantize_linear_weight(w2_packed[idx], codec)

        expert_map = torch.full(
            (global_num_experts,), -1, dtype=torch.int32, device=x.device
        )
        expert_map[idx] = torch.arange(idx.numel(), dtype=torch.int32, device=x.device)
        output += fused_experts(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )
    return output
