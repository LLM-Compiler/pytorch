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

Gemma-1B-like MLP dims:
  python scripts/inductor/mlp_block_pytorch_demo.py --s 2 --h 1152 --d 6912 --hout 1152
"""

from __future__ import annotations

import argparse

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
    return parser.parse_args()


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

    # Eager execution
    y_eager = model(x, a, b1, b, b2)
    y_eager = model(x, a, b1, b, b2)

    # Compiled execution
    compiled_model = torch.compile(model)
    y_compiled = compiled_model(x, a, b1, b, b2)
    y_compiled = compiled_model(x, a, b1, b, b2)

    print("MLPBlock (addmm -> gelu -> addmm) output shape:", tuple(y_eager.shape))
    print("Mode: eager + torch.compile")
    print("Eager output tensor:")
    print(y_eager)
    print("Compiled output tensor:")
    print(y_compiled)
    print("allclose(eager, compiled):", torch.allclose(y_eager, y_compiled, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    main()
