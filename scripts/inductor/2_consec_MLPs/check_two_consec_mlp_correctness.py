#!/usr/bin/env python3
# mypy: ignore-errors
"""
Correctness checker for two consecutive functional MLP blocks.

Pattern:
  addmm -> gelu -> addmm -> addmm -> gelu -> addmm

Compares:
  - eager reference
  - torch.compile output

Optional NUMA TP mode:
  --numa --num-tp-shards 4

Examples:
  python scripts/inductor/2_consec_MLPs/check_two_consec_mlp_correctness.py
  python scripts/inductor/2_consec_MLPs/check_two_consec_mlp_correctness.py --s 512 --h 1152 --d 6912
  TORCH_COMPILE_DEBUG=1 python scripts/inductor/2_consec_MLPs/check_two_consec_mlp_correctness.py --s 512 --h 1152 --d 6912 --numa --num-tp-shards 4
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch._inductor.config.max_autotune = True


class TwoConsecutiveMLP(nn.Module):
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--dump-max-error",
        action="store_true",
        help="Print max_abs_diff and max_rel_diff for debugging.",
    )
    parser.add_argument(
        "--numa",
        action="store_true",
        help="Enable NUMA TP compile path (_numa_mlp_tp).",
    )
    parser.add_argument(
        "--num-tp-shards",
        type=int,
        default=4,
        help="NUMA TP shard/world size (used only with --numa).",
    )
    return parser.parse_args()


def _max_error_metrics(actual: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float]:
    actual64 = actual.detach().to(torch.float64)
    ref64 = ref.detach().to(torch.float64)
    abs_diff = (actual64 - ref64).abs()
    max_abs = float(abs_diff.max().item())
    max_rel = float((abs_diff / ref64.abs().clamp_min(1e-12)).max().item())
    return max_abs, max_rel


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)
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
    call_args = (x, a1, b1_1, b1_w, b1_2, a2, b2_1, b2_w, b2_2)
    ref = model(*call_args)
    ref = model(*call_args)

    compiled = None
    cpu_mod = getattr(torch._C, "_cpu", None)
    old_numa_enabled = None
    try:
        if args.numa:
            if d % args.num_tp_shards != 0:
                raise RuntimeError("d must be divisible by --num-tp-shards in NUMA mode.")
            if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
                raise RuntimeError("NUMA runtime hooks unavailable in this build.")
            if not hasattr(cpu_mod, "_is_numa_enabled") or not hasattr(cpu_mod, "_num_numa_nodes"):
                raise RuntimeError("Need _is_numa_enabled and _num_numa_nodes hooks.")
            old_numa_enabled = cpu_mod._is_numa_enabled()
            cpu_mod._set_numa_enabled(True)
            if not cpu_mod._is_numa_enabled():
                raise RuntimeError("Failed to enable NUMA runtime.")
            n_nodes = int(cpu_mod._num_numa_nodes())
            if n_nodes < args.num_tp_shards:
                raise RuntimeError(
                    f"Need at least {args.num_tp_shards} NUMA nodes, got {n_nodes}."
                )
            with inductor_config.patch(
                {
                    "_numa_mlp_tp": True,
                    "_numa_mlp_tp_world_size": args.num_tp_shards,
                    "_numa_mlp_tp_compute_shard": 0,
                }
            ):
                compiled = torch.compile(model)
                out = compiled(*call_args)
                out = compiled(*call_args)
        else:
            compiled = torch.compile(model)
            out = compiled(*call_args)
            out = compiled(*call_args)
    finally:
        if args.numa and cpu_mod is not None and old_numa_enabled is not None:
            cpu_mod._set_numa_enabled(old_numa_enabled)

    ok = torch.allclose(out, ref, atol=args.atol, rtol=args.rtol)
    print("TwoConsecutiveMLP output shape:", tuple(out.shape))
    print(f"mode: {'NUMA torch.compile' if args.numa else 'standard torch.compile'}")
    print(f"allclose(eager, compiled): {ok}")
    if args.dump_max_error:
        max_abs, max_rel = _max_error_metrics(out, ref)
        print(f"max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}")

    if os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1":
        try:
            from torch._dynamo.utils import get_debug_dir

            log_path = os.path.join(get_debug_dir(), "torchdynamo", "debug.log")
            print("TORCH_COMPILE_DEBUG=1 is set; Dynamo file log:", log_path)
        except Exception as e:
            print("Could not resolve Dynamo debug log path:", e)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
