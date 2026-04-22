#!/usr/bin/env python3
# mypy: ignore-errors
"""
Functional attention block (addmm Q/K/V -> MATH-backend SDPA -> addmm output),
matching the pattern the NUMA attention TP pass targets.

This script runs both:
  - eager PyTorch (with ``sdpa_kernel(SDPBackend.MATH)``)
  - ``torch.compile`` (default backend, also under MATH SDPA)

No custom Inductor NUMA flags are set here; this is the eager/reference
demo. See ``attention_block_torch_compile_demo.py`` for the Inductor
NUMA tensor-parallel driver.

Run:
  python scripts/inductor/attention_block_pytorch_demo.py

Custom dims:
  python scripts/inductor/attention_block_pytorch_demo.py \
      --batch 1 --seq 6 --h-in 12 --num-heads 4 --head-dim 8

Timing (mean forward per iteration after warmup):
  python scripts/inductor/attention_block_pytorch_demo.py --warmup 5 --iters 50

Gemma 3 1B attention (``google/gemma-3-1b-it``): ``hidden_size=1152``, ``num_attention_heads=4``,
  ``head_dim=256`` → ``H_qkv = 4 * 256 = 1024`` (not ``1152 / 4``; FFN 6912 is unrelated):
  python scripts/inductor/attention_block_pytorch_demo.py \\
      --batch 1 --seq 512 --h-in 1152 --num-heads 4 --head-dim 256
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor_config  # noqa: F401 - ensures submodule import
from torch.nn.attention import sdpa_kernel, SDPBackend

torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch._inductor.config.max_autotune = True


class AttentionBlock(nn.Module):
    """Functional MHA block whose compute graph aligns 1:1 with
    ``numa_attention_tp_pass``'s matcher.

    Shapes:
        x  : [B*S, H_in]           - packed token activations
        Wq/Wk/Wv : [H_in, H*E]     - column-parallel projection weights
        bq/bk/bv : [H*E]
        Wo : [H*E, H_in]           - row-parallel output projection weight
        bo : [H_in]
    Output:
        out: [B*S, H_in]
    """

    def __init__(self, batch: int, seq: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.B = batch
        self.S = seq
        self.H = num_heads
        self.E = head_dim

    def forward(
        self,
        x: torch.Tensor,
        Wq: torch.Tensor,
        Wk: torch.Tensor,
        Wv: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        Wo: torch.Tensor,
        bo: torch.Tensor,
    ) -> torch.Tensor:
        B, S, H, E = self.B, self.S, self.H, self.E
        q = torch.addmm(bq, x, Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(bk, x, Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(bv, x, Wv).view(B, S, H, E).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        return torch.addmm(bo, y, Wo)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch size B")
    parser.add_argument("--seq", type=int, default=6, help="Sequence length S")
    parser.add_argument("--h-in", type=int, default=12, help="Input hidden dim H_in")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads H")
    parser.add_argument("--head-dim", type=int, default=8, help="Per-head dim E")
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

    B = args.batch
    S = args.seq
    H_in = args.h_in
    H = args.num_heads
    E = args.head_dim
    H_qkv = H * E

    x = torch.randn(B * S, H_in, device=device)
    Wq = torch.randn(H_in, H_qkv, device=device)
    Wk = torch.randn(H_in, H_qkv, device=device)
    Wv = torch.randn(H_in, H_qkv, device=device)
    bq = torch.randn(H_qkv, device=device)
    bk = torch.randn(H_qkv, device=device)
    bv = torch.randn(H_qkv, device=device)
    Wo = torch.randn(H_qkv, H_in, device=device)
    bo = torch.randn(H_in, device=device)

    model = AttentionBlock(B, S, H, E).to(device)

    def run_eager() -> None:
        nonlocal y_eager
        y_eager = model(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)

    def run_compiled() -> None:
        nonlocal y_compiled
        y_compiled = compiled(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)

    # Eager
    y_eager = model(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)
    y_eager = model(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)

    # Compiled (default backend). No NUMA pass; just the math-decomposed SDPA.
    compiled = torch.compile(model)
    y_compiled = compiled(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)
    y_compiled = compiled(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)

    eager_mean_s = _mean_forward_seconds(run_eager, warmup=args.warmup, iters=args.iters)
    compiled_mean_s = _mean_forward_seconds(run_compiled, warmup=args.warmup, iters=args.iters)

    print(f"AttentionBlock (B={B}, S={S}, H_in={H_in}, H={H}, E={E}, H_qkv={H_qkv})")
    print(f"Output shape: {tuple(y_eager.shape)}")
    print(f"scale used (1/sqrt(E)): {1.0 / math.sqrt(E):.6f}")
    print("allclose(eager, compiled):",
          torch.allclose(y_eager, y_compiled, atol=1e-5, rtol=1e-5))
    print(
        f"Mean forward time (warmup={args.warmup}, iters={args.iters}): "
        f"eager {eager_mean_s * 1e3:.4f} ms/iter, "
        f"compiled {compiled_mean_s * 1e3:.4f} ms/iter"
    )


if __name__ == "__main__":
    main()
