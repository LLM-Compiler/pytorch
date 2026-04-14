#!/usr/bin/env python3
# mypy: ignore-errors
"""
Two consecutive functional MLP blocks compiled with torch.compile and NUMA TP enabled.

Pattern:
  addmm -> gelu -> addmm -> addmm -> gelu -> addmm

Run:
  TORCH_COMPILE_DEBUG=1 python "scripts/inductor/2 consec MLPs/two_consec_mlp_torch_compile_demo.py"
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

NUM_TP_SHARDS = 4

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
    parser.add_argument(
        "--num-tp-shards",
        type=int,
        default=NUM_TP_SHARDS,
        help="NUMA TP shard/world size",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cpu")
    s, h, d = args.s, args.h, args.d
    num_tp_shards = args.num_tp_shards
    assert d % num_tp_shards == 0, "d must be divisible by num-tp-shards"

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
    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        raise RuntimeError("NUMA runtime hooks are unavailable in this build.")
    if not hasattr(cpu_mod, "_is_numa_enabled") or not hasattr(cpu_mod, "_num_numa_nodes"):
        raise RuntimeError("Need _is_numa_enabled and _num_numa_nodes hooks.")

    old_numa_enabled = cpu_mod._is_numa_enabled()
    cpu_mod._set_numa_enabled(True)
    if not cpu_mod._is_numa_enabled():
        raise RuntimeError("Failed to enable NUMA runtime.")
    n_nodes = int(cpu_mod._num_numa_nodes())
    if n_nodes < num_tp_shards:
        raise RuntimeError(
            f"Need at least {num_tp_shards} NUMA nodes, but runtime reports {n_nodes}."
        )

    try:
        with inductor_config.patch(
            {
                "_numa_mlp_tp": True,
                "_numa_mlp_tp_world_size": num_tp_shards,
                "_numa_mlp_tp_compute_shard": 0,
            }
        ):
            compiled = torch.compile(model)
            y = compiled(x, a1, b1_1, b1_w, b1_2, a2, b2_1, b2_w, b2_2)
            y = compiled(x, a1, b1_1, b1_w, b1_2, a2, b2_1, b2_w, b2_2)
    finally:
        cpu_mod._set_numa_enabled(old_numa_enabled)

    print("TwoConsecutiveMLP output shape:", tuple(y.shape))
    print(
        "Inductor: _numa_mlp_tp=True, "
        f"_numa_mlp_tp_world_size={num_tp_shards}, _numa_mlp_tp_compute_shard=0"
    )
    print(f"NUMA runtime: enabled=True, _num_numa_nodes()={n_nodes}")

    if os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1":
        try:
            from torch._dynamo.utils import get_debug_dir

            log_path = os.path.join(get_debug_dir(), "torchdynamo", "debug.log")
            print("TORCH_COMPILE_DEBUG=1 is set; Dynamo file log:", log_path)
        except Exception as e:
            print("Could not resolve Dynamo debug log path:", e)


if __name__ == "__main__":
    main()
