#!/usr/bin/env python3
# mypy: ignore-errors
"""
Two consecutive functional MLP blocks:
  addmm -> gelu -> addmm -> addmm -> gelu -> addmm

Runs both:
  - eager PyTorch
  - torch.compile (default backend)

No custom Inductor NUMA flags are set here.

Run:
  python "scripts/inductor/2 consec MLPs/two_consec_mlp_pytorch_demo.py"
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch._inductor.config.max_autotune = True


class TwoConsecutiveMLP(nn.Module):
    """
    Two chained MLP blocks, each in Megatron-style functional form.

    Block-1:
      x [s,h], a1 [h,d], b1_1 [d], b1_w [d,h], b1_2 [h] -> y1 [s,h]
    Block-2:
      y1 [s,h], a2 [h,d], b2_1 [d], b2_w [d,h], b2_2 [h] -> y2 [s,h]
    """

    def forward(
        self,
        x: torch.Tensor,
        a1: torch.Tensor,
        b1_1: torch.Tensor,
        b1_w: torch.Tensor,
        b1_2: torch.Tensor,
        a2: torch.Tensor,
        b2_1: torch.Tensor,
        b2_w: torch.Tensor,
        b2_2: torch.Tensor,
    ) -> torch.Tensor:
        y1 = torch.addmm(b1_1, x, a1)
        y1 = torch.nn.functional.gelu(y1)
        y1 = torch.addmm(b1_2, y1, b1_w)

        y2 = torch.addmm(b2_1, y1, a2)
        y2 = torch.nn.functional.gelu(y2)
        y2 = torch.addmm(b2_2, y2, b2_w)
        return y2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=2, help="Batch/sequence dimension")
    parser.add_argument("--h", type=int, default=4, help="Hidden/input-output dimension")
    parser.add_argument("--d", type=int, default=8, help="MLP intermediate dimension")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cpu")
    s, h, d = args.s, args.h, args.d

    x = torch.randn(s, h, device=device)
    a1 = torch.randn(h, d, device=device)
    b1_1 = torch.randn(d, device=device)
    b1_w = torch.randn(d, h, device=device)
    b1_2 = torch.randn(h, device=device)

    a2 = torch.randn(h, d, device=device)
    b2_1 = torch.randn(d, device=device)
    b2_w = torch.randn(d, h, device=device)
    b2_2 = torch.randn(h, device=device)

    model = TwoConsecutiveMLP().to(device)

    y_eager = model(x, a1, b1_1, b1_w, b1_2, a2, b2_1, b2_w, b2_2)
    y_eager = model(x, a1, b1_1, b1_w, b1_2, a2, b2_1, b2_w, b2_2)

    compiled_model = torch.compile(model)
    y_compiled = compiled_model(x, a1, b1_1, b1_w, b1_2, a2, b2_1, b2_w, b2_2)
    y_compiled = compiled_model(x, a1, b1_1, b1_w, b1_2, a2, b2_1, b2_w, b2_2)

    print("TwoConsecutiveMLP output shape:", tuple(y_eager.shape))
    print("Mode: eager + torch.compile")
    print("allclose(eager, compiled):", torch.allclose(y_eager, y_compiled, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    main()
