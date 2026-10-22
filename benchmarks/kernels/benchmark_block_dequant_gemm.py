# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the SM90 fused packed-block dequant GEMM (IQ2_XXS / Q8_0).

Times `x @ dequant(W).T` where W stays in packed ggml blocks and tiles are
decoded in registers (never materialized) against the materialize-then-cuBLAS
reference. Shapes mirror the Nemotron Bolt 74B checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
        benchmarks/kernels/benchmark_block_dequant_gemm.py
"""

import argparse

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.modelopt_block_quant_hopper import (
    block_linear,
)
from vllm.model_executor.layers.quantization.modelopt_block_quant_sm90 import (
    _dequantize_blocks,
)

Q8_SHAPES = [
    (131072, 128, 34),  # lm_head
    (1024, 128, 34),  # fc1_latent_proj
    (4096, 32, 34),  # fc2_latent_proj
]
IQ2_SHAPES = [
    (5376, 16, 66),  # shared expert up
    (4096, 21, 66),  # shared expert down
]

TEST_MS = [1, 64, 256]


def make_packed(shape: tuple[int, int, int], codec: str) -> torch.Tensor:
    torch.manual_seed(0)
    packed = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    packed[..., 0] = 0x3C
    packed[..., 1] = 0x00
    if codec == "q8_0":
        packed[..., 0] = 0x00
        packed[..., 1] = 0x3C
    return packed


def bench(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    import time

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def run(codec: str, shapes: list, args: argparse.Namespace) -> None:
    weights = 32 if codec == "q8_0" else 256
    for shape in shapes:
        packed = make_packed(shape, codec)
        n, k = shape[0], shape[1] * weights
        for m in TEST_MS:
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.05

            def time_new(x=x, packed=packed, codec=codec) -> torch.Tensor:
                return block_linear(x, packed, codec)

            def time_ref(x=x, packed=packed, codec=codec) -> torch.Tensor:
                return F.linear(x, _dequantize_blocks(packed, codec))

            new = bench(time_new, args.iters)
            ref = bench(time_ref, max(1, args.iters // 5))
            slice_rows = min(n, 4096)
            packed_slice = packed[:slice_rows].contiguous()
            err = (
                (
                    block_linear(x[:, :k], packed_slice, codec).float()
                    - F.linear(x, _dequantize_blocks(packed_slice, codec)).float()
                )
                .abs()
                .max()
                .item()
            )
            del packed_slice
            torch.cuda.empty_cache()
            print(
                f"  {codec:8s} {str(shape):18s} M={m:<4d} "
                f"fused {new:7.3f} ms  ref {ref:7.3f} ms  "
                f"x {ref / new:5.1f}  max_abs_diff {err:.3e}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SM90 fused packed-block dequant GEMM."
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="timing iterations per call"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    capacity = torch.cuda.get_device_properties(0).major
    if capacity != 9:
        raise SystemExit(f"requires Hopper (SM90), got SM{capacity}")

    print("Bolt checkpoint dense shapes, packed [rows, K_blocks, bytes]:")
    run("q8_0", Q8_SHAPES, args)
    run("iq2_xxs", IQ2_SHAPES, args)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
