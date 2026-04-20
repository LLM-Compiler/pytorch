#!/usr/bin/env python3
# mypy: ignore-errors
"""
Functional MLP (``addmm`` → ``F.gelu`` → ``addmm``), same pattern as
``scripts/inductor/numa_mlp_tp_graph_demo.py``, compiled with ``torch.compile``.

This script enables ``torch._inductor.config._numa_mlp_tp`` so Inductor runs
``numa_mlp_tp_pass`` during post-grad (see ``post_grad.py``). The inner dim ``d``
must be divisible by ``NUM_TP_SHARDS``.

Architecture:
  - FX pass rewrites only dimensions (slice nodes) and annotates ``numa_shard_plan``
    metadata. No clone nodes are inserted at the FX level.
  - Inductor wrapper codegen reads ``numa_shard_plan`` and emits per-NUMA-node
    ``aten.clone.default(..., numa_node=N)`` calls for all input/output buffers.
  - Triton CPU launcher receives ``numa_args`` (per-node pointer tuples) and
    launches the same kernel grid N times with OpenMP threads pinned to each
    NUMA node via ``sched_setaffinity``.

Run with extra Dynamo debug logging:

  TORCH_COMPILE_DEBUG=1 python scripts/inductor/mlp_block_torch_compile_demo.py

For Inductor graph/code logging:

  TORCH_LOGS="+graph_code" python scripts/inductor/mlp_block_torch_compile_demo.py

Custom dims / shard count:
  python scripts/inductor/mlp_block_torch_compile_demo.py --s 2 --h 4 --d 8 --hout 3 --num-tp-shards 4

Gemma-1B-like MLP dims:
  python scripts/inductor/mlp_block_torch_compile_demo.py --s 2 --h 1152 --d 6912 --hout 1152 --num-tp-shards 4
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

# Match ``numa_mlp_tp_graph_demo.py``; ``d`` below must be divisible by this.
NUM_TP_SHARDS = 4

torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch._inductor.config.max_autotune = True



class MLPBlock(nn.Module):
    """
    Megatron-style 2-layer MLP fragment: first linear + GELU + second linear, all functional.

    Shapes: ``x`` [s, h], ``a`` [h, d], ``b1`` [d], ``b`` [d, hout], ``b2`` [hout] → out [s, hout].
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
        "--num-tp-shards",
        type=int,
        default=None,
        help=(
            "NUMA TP shard/world size "
            f"(default: min({NUM_TP_SHARDS}, num NUMA nodes))"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cpu")
    s, h, d, hout = args.s, args.h, args.d, args.hout

    x = torch.randn(s, h, device=device) # [2, 4]
    a = torch.randn(h, d, device=device) # [4, 8]
    b1 = torch.randn(d, device=device) # [8]
    b = torch.randn(d, hout, device=device) # [8, 3]
    b2 = torch.randn(hout, device=device) # [3]

    model = MLPBlock().to(device)
    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        raise RuntimeError(
            "NUMA runtime control hooks are unavailable in this build "
            "(missing torch._C._cpu._set_numa_enabled)."
        )
    if not hasattr(cpu_mod, "_is_numa_enabled") or not hasattr(cpu_mod, "_num_numa_nodes"):
        raise RuntimeError(
            "NUMA runtime query hooks are unavailable in this build "
            "(need _is_numa_enabled and _num_numa_nodes)."
        )

    old_numa_enabled = cpu_mod._is_numa_enabled()
    cpu_mod._set_numa_enabled(True)
    if not cpu_mod._is_numa_enabled():
        raise RuntimeError("Failed to enable NUMA runtime via torch._C._cpu._set_numa_enabled(True)")
    n_nodes = int(cpu_mod._num_numa_nodes())
    if args.num_tp_shards is None:
        num_tp_shards = min(NUM_TP_SHARDS, max(1, n_nodes))
        print(
            f"num-tp-shards not set; using min({NUM_TP_SHARDS}, NUMA nodes) = {num_tp_shards}"
        )
    else:
        num_tp_shards = args.num_tp_shards
        if n_nodes < num_tp_shards:
            raise RuntimeError(
                f"Need at least {num_tp_shards} NUMA nodes, but runtime reports {n_nodes}. "
                "Lower num-tp-shards or run on a machine with more NUMA nodes."
            )

    assert d % num_tp_shards == 0, (
        "d must be divisible by num_tp_shards for numa_mlp_tp_pass"
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
            y = compiled(x, a, b1, b, b2)
            y = compiled(x, a, b1, b, b2)
    finally:
        cpu_mod._set_numa_enabled(old_numa_enabled)

    print("MLPBlock (addmm → gelu → addmm) output shape:", tuple(y.shape))
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
