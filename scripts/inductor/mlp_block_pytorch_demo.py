#!/usr/bin/env python3
# mypy: ignore-errors
"""
Functional MLP (``addmm`` -> ``F.gelu`` -> ``addmm``), same math pattern as
``scripts/inductor/mlp_block_torch_compile_demo.py``.

This script runs both:
  - eager PyTorch
  - ``torch.compile`` (default backend)

No custom Inductor NUMA flags are set here.

Run:
  python scripts/inductor/mlp_block_pytorch_demo.py

Custom dims:
  python scripts/inductor/mlp_block_pytorch_demo.py --s 2 --h 4 --d 8 --hout 3

Gemma-1B-like MLP dims (h → d → hout):
  python scripts/inductor/mlp_block_pytorch_demo.py --s 2 --h 1152 --d 6912 --hout 1152

Timing:
  python scripts/inductor/mlp_block_pytorch_demo.py --s 2 --h 1152 --d 6912 --hout 1152 --warmup 5 --iters 50
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch._inductor.config.max_autotune = True



class MLPBlock(nn.Module):
    """
    Megatron-style 2-layer MLP fragment: first linear + GELU + second linear.

    Shapes: ``x`` [s, h], ``a`` [h, d], ``b1`` [d], ``b`` [d, hout], ``b2`` [hout]
    -> out [s, hout].
    """

    def forward(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b1: torch.Tensor,
        b: torch.Tensor,
        b2: torch.Tensor,
    ) -> torch.Tensor:
        y = torch.addmm(b1, x, a)
        y = torch.nn.functional.gelu(y)
        return torch.addmm(b2, y, b)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=2, help="Batch/sequence dimension")
    parser.add_argument("--h", type=int, default=4, help="Input hidden dimension")
    parser.add_argument("--d", type=int, default=8, help="MLP intermediate dimension")
    parser.add_argument("--hout", type=int, default=3, help="Output hidden dimension")
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        metavar="N",
        help="Warmup forward passes before timing (not counted).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        metavar="N",
        help="Timed forward passes; mean time is reported over this many runs.",
    )
    return parser.parse_args()


def _mean_forward_seconds(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    if iters <= 0:
        return float("nan")
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cpu")
    s, h, d, hout = args.s, args.h, args.d, args.hout

    x = torch.randn(s, h, device=device)  # [2, 4]
    a = torch.randn(h, d, device=device)  # [4, 8]
    b1 = torch.randn(d, device=device)  # [8]
    b = torch.randn(d, hout, device=device)  # [8, 3]
    b2 = torch.randn(hout, device=device)  # [3]

    model = MLPBlock().to(device)

    def run_eager() -> None:
        nonlocal y_eager
        y_eager = model(x, a, b1, b, b2)

    def run_compiled() -> None:
        nonlocal y_compiled
        y_compiled = compiled_model(x, a, b1, b, b2)

    # Eager execution
    y_eager = model(x, a, b1, b, b2)
    y_eager = model(x, a, b1, b, b2)

    # Compiled execution
    compiled_model = torch.compile(model)
    y_compiled = compiled_model(x, a, b1, b, b2)
    y_compiled = compiled_model(x, a, b1, b, b2)

    eager_mean_s = _mean_forward_seconds(run_eager, warmup=args.warmup, iters=args.iters)
    compiled_mean_s = _mean_forward_seconds(run_compiled, warmup=args.warmup, iters=args.iters)

    print("MLPBlock (addmm -> gelu -> addmm) output shape:", tuple(y_eager.shape))
    print("Mode: eager + torch.compile")
    print("allclose(eager, compiled):", torch.allclose(y_eager, y_compiled, atol=1e-5, rtol=1e-5))
    print(
        f"Mean forward time (warmup={args.warmup}, iters={args.iters}): "
        f"eager {eager_mean_s * 1e3:.4f} ms/iter, "
        f"compiled {compiled_mean_s * 1e3:.4f} ms/iter"
    )


if __name__ == "__main__":
    main()
