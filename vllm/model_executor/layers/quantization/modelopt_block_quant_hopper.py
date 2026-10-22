# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused packed-block dequant GEMM kernels for Hopper (SM90).

ModelOpt checkpoints carry IQ2_XXS (66B -> 256 weights) and Q8_0 (34B -> 32
weights) ggml byte blocks. Packed weights stay resident in their original
byte form at rest. These kernels decode tiles in registers into bf16 immediately
before the tensor-core GEMM (or a GEMV for a single token), so no dense copy
of a weight matrix is ever materialized persistently.

Decode arithmetic mirrors the b12x GGML oracles: every intermediate product
is exactly representable in fp32, so fp32 multiply chains are bit-exact
against the oracles and the final weights round to bf16 the same way.
"""

from __future__ import annotations

import os
import threading

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except ImportError:  # pragma: no cover
    _TRITON_OK = False

IQ2_BLOCK_BYTES = 66
IQ2_BLOCK_WEIGHTS = 256
Q8_0_BLOCK_BYTES = 34
Q8_0_BLOCK_WEIGHTS = 32

_GRID_CACHE: dict[tuple[str, torch.device], torch.Tensor] = {}
_GRID_CACHE_LOCK = threading.Lock()
_ENABLED_CACHE: bool | None = None


def prewarm_device_checks() -> None:
    """Resolve device-dependent gates eagerly (outside torch.compile regions)."""
    hopper_block_kernel_enabled()


def hopper_block_kernel_enabled() -> bool:
    """Whether the fused SM90 block kernels should be used on this device."""
    global _ENABLED_CACHE
    if _ENABLED_CACHE is not None:
        return _ENABLED_CACHE
    flag = os.environ.get("VLLM_SM90_BLOCK_KERNEL", "1").strip().lower()
    enabled = flag not in {"0", "false", "off", "disabled"}
    if enabled and not _TRITON_OK:
        enabled = False
    if enabled:
        if not torch.cuda.is_available():
            enabled = False
        else:
            from vllm.platforms import current_platform

            capability = current_platform.get_device_capability()
            enabled = capability is not None and capability[0] == 9
    _ENABLED_CACHE = enabled
    return _ENABLED_CACHE


def _iq2_grid(codec: str, device: torch.device) -> torch.Tensor:
    key = (codec, device)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        from b12x._lib.quant.iq2_xs import iq2_xs_grid_cpu

        grid = (
            iq2_xs_grid_cpu(codec).contiguous().to(device=device, dtype=torch.float32)
        )
        assert grid.shape == (256, 8)
        _GRID_CACHE[key] = grid
    return grid


if _TRITON_OK:

    @triton.jit
    def _iq2_decode_tile(
        w_ptr,
        blk_base,  # [R] absolute byte offset of each block
        rec,  # [256] record id per weight
        grp,  # [256] group id per weight
        lane,  # [256] lane id per weight
        lut_ptr,
        row_mask,  # [R]
    ):
        rb = 2 + rec * 8
        cb = rb + 4
        row_w = blk_base[:, None]

        # base d: little-endian fp16 pair at bytes 0..1
        d0 = tl.load(w_ptr + blk_base, mask=row_mask, other=0).to(tl.uint16)
        d1 = tl.load(w_ptr + blk_base + 1, mask=row_mask, other=0).to(tl.uint16)
        d = (d0 | (d1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        c0 = tl.load(
            w_ptr + row_w + cb[None, :] + 0, mask=row_mask[:, None], other=0
        ).to(tl.uint32)
        c1 = tl.load(
            w_ptr + row_w + cb[None, :] + 1, mask=row_mask[:, None], other=0
        ).to(tl.uint32)
        c2 = tl.load(
            w_ptr + row_w + cb[None, :] + 2, mask=row_mask[:, None], other=0
        ).to(tl.uint32)
        c3 = tl.load(
            w_ptr + row_w + cb[None, :] + 3, mask=row_mask[:, None], other=0
        ).to(tl.uint32)
        control = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24)

        scale = (d[:, None] * ((control >> 28).to(tl.float32) + 0.5)) * 0.25

        sign_index = (control >> (7 * grp[None, :])).to(tl.int32) & 127
        parity = sign_index ^ (sign_index >> 4)
        parity = parity ^ (parity >> 2)
        parity = parity ^ (parity >> 1)
        parity = parity & 1
        mask8 = sign_index | (parity << 7)
        sign = (1 - 2 * ((mask8 >> lane[None, :]) & 1)).to(tl.float32)

        grid_idx = tl.load(
            w_ptr + row_w + rb[None, :] + grp[None, :],
            mask=row_mask[:, None],
            other=0,
        ).to(tl.int32)
        lut_v = tl.load(lut_ptr + grid_idx * 8 + lane[None, :])

        return lut_v * scale * sign

    @triton.jit
    def _q8_decode_tile(w_ptr, blk_base, lane, row_mask):
        # scales are fp16 at bytes 0..1; 34-byte blocks keep them 2B-aligned
        d0 = tl.load(w_ptr + blk_base, mask=row_mask, other=0).to(tl.uint16)
        d1 = tl.load(w_ptr + blk_base + 1, mask=row_mask, other=0).to(tl.uint16)
        scale = (d0 | (d1 << 8)).to(tl.float16, bitcast=True).to(tl.float32)[:, None]

        values = tl.load(
            w_ptr + blk_base[:, None] + 2 + lane[None, :],
            mask=row_mask[:, None],
            other=0,
        )
        return values.to(tl.int8).to(tl.float32) * scale

else:
    _iq2_decode_tile = None
    _q8_decode_tile = None


if _TRITON_OK:

    @triton.jit
    def _iq2_gemv_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        lut_ptr,
        N,
        NB,
        stride_wn,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        i = tl.arange(0, 256)
        rec = i // 32
        grp = (i // 8) % 4
        lane = i % 8
        row = offs_n.to(tl.int64) * stride_wn

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for kb in range(NB):
            blk = row + kb * 66
            w = _iq2_decode_tile(w_ptr, blk, rec, grp, lane, lut_ptr, mask_n)
            xv = tl.load(x_ptr + kb * 256 + i).to(tl.float32)
            acc += tl.sum(w * xv[None, :], 1)
        tl.store(y_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)

    @triton.jit
    def _q8_gemv_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        N,
        NB,
        stride_wn,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        lane = tl.arange(0, 32)
        row = offs_n.to(tl.int64) * stride_wn

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for kb in range(NB):
            blk = row + kb * 34
            w = _q8_decode_tile(w_ptr, blk, lane, mask_n)
            xv = tl.load(x_ptr + kb * 32 + lane).to(tl.float32)
            acc += tl.sum(w * xv[None, :], 1)
        tl.store(y_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)

    @triton.jit
    def _iq2_gemm_kernel(
        x_ptr,
        w_ptr,
        o_ptr,
        lut_ptr,
        M,
        N,
        K,
        NB,
        stride_wn,
        BN: tl.constexpr,
        BM: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m = pid_m * BM + tl.arange(0, BM)
        mask_n = offs_n < N
        mask_m = offs_m < M
        i = tl.arange(0, 256)
        rec = i // 32
        grp = (i // 8) % 4
        lane = i % 8
        row = offs_n.to(tl.int64) * stride_wn

        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for kb in range(NB):
            blk = row + kb * 66
            wt = _iq2_decode_tile(w_ptr, blk, rec, grp, lane, lut_ptr, mask_n)
            wb = wt.to(tl.bfloat16)
            a = tl.load(
                x_ptr + offs_m[:, None] * K + (kb * 256 + i)[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            acc = tl.dot(a, tl.trans(wb), acc)
        tl.store(
            o_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _q8_gemm_kernel(
        x_ptr,
        w_ptr,
        o_ptr,
        M,
        N,
        K,
        NB,
        stride_wn,
        BN: tl.constexpr,
        BM: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m = pid_m * BM + tl.arange(0, BM)
        mask_n = offs_n < N
        mask_m = offs_m < M
        lane = tl.arange(0, 32)
        row = offs_n.to(tl.int64) * stride_wn

        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for kb in range(NB):
            blk = row + kb * 34
            wt = _q8_decode_tile(w_ptr, blk, lane, mask_n)
            wb = wt.to(tl.bfloat16)
            a = tl.load(
                x_ptr + offs_m[:, None] * K + (kb * 32 + lane)[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            acc = tl.dot(a, tl.trans(wb), acc)
        tl.store(
            o_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _iq2_gemm_split_kernel(
        x_ptr,
        w_ptr,
        p_ptr,
        lut_ptr,
        M,
        N,
        K,
        NB,
        step,
        stride_wn,
        BN: tl.constexpr,
        BM: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m = pid_m * BM + tl.arange(0, BM)
        mask_n = offs_n < N
        mask_m = offs_m < M
        i = tl.arange(0, 256)
        rec = i // 32
        grp = (i // 8) % 4
        lane = i % 8
        row = offs_n.to(tl.int64) * stride_wn
        k_lo = pid_k * step

        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for kbb in range(step):
            kb = k_lo + kbb
            blk = row + kb * 66
            wt = _iq2_decode_tile(w_ptr, blk, rec, grp, lane, lut_ptr, mask_n)
            wb = wt.to(tl.bfloat16)
            a = tl.load(
                x_ptr + offs_m[:, None] * K + (kb * 256 + i)[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            acc = tl.dot(a, tl.trans(wb), acc)
        base = pid_k.to(tl.int64) * M * N
        tl.store(
            p_ptr + base + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _q8_gemm_split_kernel(
        x_ptr,
        w_ptr,
        p_ptr,
        M,
        N,
        K,
        NB,
        step,
        stride_wn,
        BN: tl.constexpr,
        BM: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_m = pid_m * BM + tl.arange(0, BM)
        mask_n = offs_n < N
        mask_m = offs_m < M
        lane = tl.arange(0, 32)
        row = offs_n.to(tl.int64) * stride_wn
        k_lo = pid_k * step

        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for kbb in range(step):
            kb = k_lo + kbb
            blk = row + kb * 34
            wt = _q8_decode_tile(w_ptr, blk, lane, mask_n)
            wb = wt.to(tl.bfloat16)
            a = tl.load(
                x_ptr + offs_m[:, None] * K + (kb * 32 + lane)[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
            acc = tl.dot(a, tl.trans(wb), acc)
        base = pid_k.to(tl.int64) * M * N
        tl.store(
            p_ptr + base + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _combine_kernel(
        p_ptr,
        o_ptr,
        total,
        SK,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for sid in range(SK):
            acc += tl.load(p_ptr + sid.to(tl.int64) * total + offs, mask=mask, other=0)
        tl.store(o_ptr + offs, acc.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _decode_blocks_kernel(
        src_ptr,
        out_ptr,
        lut_ptr,
        idx_ptr,
        total_blocks,
        gather_span,
        HAS_IDX: tl.constexpr,
        CODEC: tl.constexpr,
        ROWS: tl.constexpr,
        STORE_F32: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_b = pid.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        mask_b = offs_b < total_blocks
        if HAS_IDX:
            e = tl.load(
                idx_ptr + (offs_b // gather_span).to(tl.int32), mask=mask_b, other=0
            )
            src_b = e.to(tl.int64) * gather_span + offs_b % gather_span
        else:
            src_b = offs_b

        if CODEC == 1:
            blk = src_b * 66
            i = tl.arange(0, 256)
            rec = i // 32
            grp = (i // 8) % 4
            lane = i % 8
            w = _iq2_decode_tile(src_ptr, blk, rec, grp, lane, lut_ptr, mask_b)
            out_ptrs = out_ptr + offs_b[:, None] * 256 + i[None, :]
        else:
            blk = src_b * 34
            lane = tl.arange(0, 32)
            w = _q8_decode_tile(src_ptr, blk, lane, mask_b)
            out_ptrs = out_ptr + offs_b[:, None] * 32 + lane[None, :]

        if STORE_F32:
            tl.store(out_ptrs, w, mask=mask_b[:, None])
        else:
            tl.store(out_ptrs, w.to(tl.bfloat16), mask=mask_b[:, None])


def _codec_layout(codec: str) -> tuple[int, int, int, int]:
    """(block_bytes, block_weights, codec flag, lut codec or '')."""
    if codec == "q8_0":
        return Q8_0_BLOCK_BYTES, Q8_0_BLOCK_WEIGHTS, 0, ""
    if codec == "iq2_xxs":
        return IQ2_BLOCK_BYTES, IQ2_BLOCK_WEIGHTS, 1, "iq2_xxs"
    raise NotImplementedError(
        f"SM90 block kernels do not support {codec}; only q8_0 and iq2_xxs"
    )


def decode_blocks(
    packed: torch.Tensor,
    codec: str,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode packed ``[..., K_blocks, bytes]`` blocks into ``[..., weights]``."""
    block_bytes, block_weights, flag, _ = _codec_layout(codec)
    packed = packed.contiguous()
    lead_shape = packed.shape[:-2]
    total_blocks = int(packed.numel() // block_bytes)
    flat = packed.reshape(total_blocks, block_bytes)
    out = torch.empty(
        (total_blocks, block_weights), dtype=out_dtype, device=packed.device
    )
    if total_blocks == 0:
        return out.reshape(*lead_shape, -1)

    rows = 64 if flag == 0 else 8
    grid = (triton.cdiv(total_blocks, rows),)
    store_f32 = out_dtype == torch.float32
    if flag == 1:
        lut = _iq2_grid("iq2_xxs", packed.device)
        _decode_blocks_kernel[grid](
            flat,
            out,
            lut,
            flat,
            total_blocks,
            0,
            HAS_IDX=False,
            CODEC=1,
            ROWS=rows,
            STORE_F32=store_f32,
            num_warps=4,
        )
    else:
        _decode_blocks_kernel[grid](
            flat,
            out,
            flat,
            flat,
            total_blocks,
            0,
            HAS_IDX=False,
            CODEC=0,
            ROWS=rows,
            STORE_F32=store_f32,
            num_warps=4,
        )
    return out.reshape(*lead_shape, -1)


def decode_expert_gather(
    packed: torch.Tensor,
    expert_ids: torch.Tensor,
    codec: str,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode a gathered subset of experts without copying packed rows.

    ``packed`` is ``[E, N, K_blocks, bytes]`` and ``expert_ids`` is a 1-D
    tensor of expert indices; the result is ``[len(ids), N, K]``.
    """
    assert packed.dim() == 4 and expert_ids.dim() == 1
    block_bytes, block_weights, flag, _ = _codec_layout(codec)
    e, rows, blocks, _ = packed.shape
    gather_span = rows * blocks
    packed = packed.contiguous()
    expert_ids = expert_ids.contiguous()
    out = torch.empty(
        (expert_ids.numel() * gather_span, block_weights),
        dtype=out_dtype,
        device=packed.device,
    )
    if out.shape[0] == 0:
        return out.view(expert_ids.numel(), rows, -1)

    rows_per_prog = 64 if flag == 0 else 8
    grid = (triton.cdiv(out.shape[0], rows_per_prog),)
    store_f32 = out_dtype == torch.float32
    if flag == 1:
        lut = _iq2_grid("iq2_xxs", packed.device)
        _decode_blocks_kernel[grid](
            packed.reshape(e * gather_span, block_bytes),
            out,
            lut,
            expert_ids,
            out.shape[0],
            gather_span,
            HAS_IDX=True,
            CODEC=1,
            ROWS=rows_per_prog,
            STORE_F32=store_f32,
            num_warps=4,
        )
    else:
        _decode_blocks_kernel[grid](
            packed.reshape(e * gather_span, block_bytes),
            out,
            expert_ids,
            expert_ids,
            out.shape[0],
            gather_span,
            HAS_IDX=True,
            CODEC=0,
            ROWS=rows_per_prog,
            STORE_F32=store_f32,
            num_warps=4,
        )
    return out.view(expert_ids.numel(), rows, -1)


def _launch_gemm(
    split_kernel,
    plain_kernel,
    x2,
    packed_flat,
    y,
    M,
    N,
    K,
    num_blocks,
    *,
    lut=None,
    BM,
    BN,
    warps,
    stages,
):
    stride_wn = packed_flat.shape[-1]
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    split_k = 1
    if tiles < 256:
        candidate = 256 // tiles
        while candidate > 1 and num_blocks % candidate != 0:
            candidate //= 2
        split_k = candidate

    def _plain():
        grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
        args = (x2, packed_flat, y, M, N, K, num_blocks, stride_wn)
        if lut is not None:
            args = args[:3] + (lut,) + args[3:]
        plain_kernel[grid](*args, BN=BN, BM=BM, num_warps=warps, num_stages=stages)

    if split_k == 1:
        _plain()
        return y
    partial = torch.empty((split_k, M, N), dtype=torch.float32, device=x2.device)
    step = num_blocks // split_k
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM), split_k)
    args = (x2, packed_flat, partial, M, N, K, num_blocks, step, stride_wn)
    if lut is not None:
        args = args[:3] + (lut,) + args[3:]
    split_kernel[grid](*args, BN=BN, BM=BM, num_warps=warps, num_stages=stages)
    _combine_kernel[(triton.cdiv(M * N, 1024),)](
        partial, y, M * N, split_k, BLOCK=1024, num_warps=4
    )
    return y


def block_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    codec: str,
) -> torch.Tensor:
    """``x @ dequant(packed).T`` decoding weight tiles only inside registers."""
    assert x.dtype == torch.bfloat16, "block_linear expects BF16 activations"
    block_bytes, block_weights, flag, _ = _codec_layout(codec)
    packed = packed.contiguous()
    N, num_blocks = packed.shape[0], packed.shape[1]
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    assert num_blocks * block_weights == K, (K, num_blocks, block_weights)

    if x2.stride() != (K, 1):
        x2 = x2.contiguous()
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if M == 0:
        return y.view(*x.shape[:-1], N) if x.dim() > 2 else y

    packed_flat = packed.reshape(N, num_blocks * block_bytes)
    if M == 1 and N >= 4096:
        gemv_grid = triton.cdiv(N, 32 if flag == 1 else 64)
        if flag == 1:
            _iq2_gemv_kernel[(gemv_grid,)](
                x2,
                packed_flat,
                y,
                _iq2_grid("iq2_xxs", x.device),
                N,
                num_blocks,
                packed_flat.shape[-1],
                BLOCK_N=32,
                num_warps=4,
                num_stages=1,
            )
        else:
            _q8_gemv_kernel[(gemv_grid,)](
                x2,
                packed_flat,
                y,
                N,
                num_blocks,
                packed_flat.shape[-1],
                BLOCK_N=64,
                num_warps=4,
                num_stages=1,
            )
    else:
        if M <= 16:
            BM = 16
        elif M <= 32:
            BM = 32
        elif M <= 64:
            BM = 64
        else:
            BM = 128
        BN = 64 if N >= 128 else 32
        warps = 8 if BM * BN >= 8192 else 4
        if flag == 1:
            _launch_gemm(
                _iq2_gemm_split_kernel,
                _iq2_gemm_kernel,
                x2,
                packed_flat,
                y,
                M,
                N,
                K,
                num_blocks,
                lut=_iq2_grid("iq2_xxs", x.device),
                BM=BM,
                BN=BN,
                warps=warps,
                stages=2,
            )
        else:
            _launch_gemm(
                _q8_gemm_split_kernel,
                _q8_gemm_kernel,
                x2,
                packed_flat,
                y,
                M,
                N,
                K,
                num_blocks,
                BM=BM,
                BN=BN,
                warps=warps,
                stages=2,
            )
    if x.dim() > 2:
        return y.view(*x.shape[:-1], N)
    return y


if _TRITON_OK:

    @triton.jit
    def _moe_pass1_iq2_kernel(
        x_ptr,
        w_ptr,
        c1_ptr,
        lut_ptr,
        sorted_ptr,
        expert_ptr,
        post_ptr,
        num_valid,
        N,
        K,
        NB,
        top_k: tl.constexpr,
        RELU2: tl.constexpr,
        BN: tl.constexpr,
        BM: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if pid_m * BM >= tl.load(post_ptr):
            return
        offs_tok = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        tok_mask = offs_tok < num_valid
        e = tl.load(expert_ptr + pid_m).to(tl.int64)
        offs_n = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        mask_n = offs_n < N
        i = tl.arange(0, 256)
        rec = i // 32
        grp = (i // 8) % 4
        lane = i % 8
        row = e * N * NB * 66 + offs_n * NB * 66

        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for kb in range(NB):
            blk = row + kb * 66
            wt = _iq2_decode_tile(w_ptr, blk, rec, grp, lane, lut_ptr, mask_n)
            wb = wt.to(tl.bfloat16)
            a = tl.load(
                x_ptr + (offs_tok // top_k)[:, None] * K + (kb * 256 + i)[None, :],
                mask=tok_mask[:, None],
                other=0.0,
            )
            acc = tl.dot(a, tl.trans(wb), acc)
        v = acc
        if RELU2:
            v = tl.maximum(v, 0.0)
            v = v * v
        tl.store(
            c1_ptr + offs_tok[:, None] * N + offs_n[None, :],
            v.to(tl.bfloat16),
            mask=tok_mask[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _moe_pass2_iq2_kernel(
        x_ptr,
        w_ptr,
        c3_ptr,
        lut_ptr,
        sorted_ptr,
        expert_ptr,
        post_ptr,
        tw_ptr,
        num_valid,
        N2,
        K2,
        NB2,
        BN: tl.constexpr,
        BM: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if pid_m * BM >= tl.load(post_ptr):
            return
        offs_tok = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        tok_mask = offs_tok < num_valid
        e = tl.load(expert_ptr + pid_m).to(tl.int64)
        offs_n = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        mask_n = offs_n < N2
        i = tl.arange(0, 256)
        rec = i // 32
        grp = (i // 8) % 4
        lane = i % 8
        row = e * N2 * NB2 * 66 + offs_n * NB2 * 66

        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for kb in range(NB2):
            blk = row + kb * 66
            wt = _iq2_decode_tile(w_ptr, blk, rec, grp, lane, lut_ptr, mask_n)
            wb = wt.to(tl.bfloat16)
            a = tl.load(
                x_ptr + offs_tok[:, None] * K2 + (kb * 256 + i)[None, :],
                mask=tok_mask[:, None],
                other=0.0,
            )
            acc = tl.dot(a, tl.trans(wb), acc)
        routed = tl.load(tw_ptr + offs_tok, mask=tok_mask, other=0.0)
        tl.store(
            c3_ptr + offs_tok[:, None] * N2 + offs_n[None, :],
            (acc * routed[:, None]).to(tl.bfloat16),
            mask=tok_mask[:, None] & mask_n[None, :],
        )


def moe_forward(
    x: torch.Tensor,
    w13_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """CUDA-graph-safe packed-IQ2_XXS MoE forward (relu2, router on output).

    Weights stay packed; the experts selected by ``topk_ids`` are decoded
    inside the GEMM kernels. Routing uses ``moe_align_block_size`` with no
    host-dependent dispatch, so the whole call is capturable.
    """
    x = x.contiguous()
    w13_packed = w13_packed.contiguous()
    w2_packed = w2_packed.contiguous()
    topk_weights = topk_weights.contiguous().view(-1)
    topk_ids = topk_ids.contiguous()
    M, K = x.shape
    topk = topk_ids.size(1)
    E, N13, nb1, _ = w13_packed.shape
    N2, nb2 = w2_packed.shape[1], w2_packed.shape[2]
    assert nb1 * IQ2_BLOCK_WEIGHTS == K, "up_proj input dim must match K"
    assert nb2 * IQ2_BLOCK_WEIGHTS == N13, "down_proj input dim must match N13"
    num_valid = M * topk
    BM, BN = 16, 64

    from vllm._custom_ops import moe_sum
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, BM, E, None, ignore_invalid_experts=True
    )
    EM = sorted_ids.numel()
    if M < BM:
        EM = min(EM, num_valid * BM)
    grid_m = triton.cdiv(EM, BM)

    lut = _iq2_grid("iq2_xxs", x.device)
    cache1 = torch.empty((num_valid, N13), dtype=x.dtype, device=x.device)
    _moe_pass1_iq2_kernel[(grid_m, triton.cdiv(N13, BN))](
        x,
        w13_packed.reshape(-1),
        cache1,
        lut,
        sorted_ids,
        expert_ids,
        num_post,
        num_valid,
        N13,
        K,
        nb1,
        top_k=topk,
        RELU2=True,
        BN=BN,
        BM=BM,
        num_warps=4,
        num_stages=2,
    )
    cache3 = torch.empty((num_valid, N2), dtype=x.dtype, device=x.device)
    _moe_pass2_iq2_kernel[(grid_m, triton.cdiv(N2, BN))](
        cache1,
        w2_packed.reshape(-1),
        cache3,
        lut,
        sorted_ids,
        expert_ids,
        num_post,
        topk_weights,
        num_valid,
        N2,
        N13,
        nb2,
        BN=BN,
        BM=BM,
        num_warps=4,
        num_stages=2,
    )
    out = torch.empty_like(x)
    moe_sum(cache3.view(M, topk, N2), out)
    return out
