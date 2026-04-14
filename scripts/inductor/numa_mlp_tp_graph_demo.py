#!/usr/bin/env python3
# mypy: ignore-errors
"""
Demonstrate ``numa_mlp_tp_pass`` on a traced addmm → GELU → addmm MLP.

Build (NUMA-aware ``clone``):
  Configure PyTorch with NUMA enabled, e.g. ``USE_NUMA=1`` (sets ``C10_USE_NUMA``;
  needs ``libnuma`` / ``numa.h`` on Linux). See ``docs/source/notes/clone_numa.rst``.

Runtime:
  ``torch._C._cpu._set_numa_enabled(True)`` turns on ``caffe2_cpu_numa_enabled`` so
  ``aten.clone(..., numa_node=...)`` can run. Shard indices must satisfy
  ``0 <= numa_node < torch._C._cpu._num_numa_nodes()`` on the host.

Run:
  conda activate <env> && cd /path/to/pytorch && python scripts/inductor/numa_mlp_tp_graph_demo.py
"""

from __future__ import annotations

import torch
from torch._inductor import config
from torch._inductor.fx_passes.numa_mlp_tp import numa_mlp_tp_pass
from torch.fx.passes.shape_prop import ShapeProp

# Tensor-parallel degree for this demo (must divide the reduction dim ``d`` below).
NUM_TP_SHARDS = 4


class MLPBlock(torch.nn.Module):
    """Megatron-style 2-layer MLP fragment: Linear -> GELU -> Linear (functional)."""

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


def main() -> None:
    torch.manual_seed(0)
    # d=8 divisible by NUM_TP_SHARDS=4 => shard width 2 along the expert dim.
    s, h, d, hout = 2, 4, 8, 3
    assert d % NUM_TP_SHARDS == 0, "d must divide NUM_TP_SHARDS"

    x = torch.randn(s, h)
    a = torch.randn(h, d)
    b1 = torch.randn(d)
    b = torch.randn(d, hout)
    b2 = torch.randn(hout)

    gm = torch.fx.symbolic_trace(MLPBlock())
    ShapeProp(gm).propagate(x, a, b1, b, b2)

    print("=" * 72)
    print("Sample MLP FX graph BEFORE numa_mlp_tp_pass (tabular)")
    print("=" * 72)
    gm.graph.print_tabular()

    print()
    with config.patch(
        {
            "_numa_mlp_tp_world_size": NUM_TP_SHARDS,
            "_numa_mlp_tp_compute_shard": 0,
            "_numa_mlp_tp_debug": True,
        }
    ):
        numa_mlp_tp_pass(gm.graph)

    gm.recompile()
    print()
    print("=" * 72)
    print("Sample MLP FX graph AFTER numa_mlp_tp_pass (tabular)")
    print("=" * 72)
    gm.graph.print_tabular()

    cpu_mod = getattr(torch._C, "_cpu", None)
    old_numa = False
    patched_numa = False
    if cpu_mod is not None and hasattr(cpu_mod, "_set_numa_enabled"):
        old_numa = cpu_mod._is_numa_enabled()
        cpu_mod._set_numa_enabled(True)
        patched_numa = True

    print()
    try:
        if cpu_mod is not None and hasattr(cpu_mod, "_num_numa_nodes"):
            n_nodes = int(cpu_mod._num_numa_nodes())
            eff = cpu_mod._is_numa_enabled()
            print(
                f"NUMA runtime: enabled={eff}, _num_numa_nodes()={n_nodes} "
                f"(need >= {NUM_TP_SHARDS} for all clone numa_node ids used in this graph)"
            )

        out = gm(x, a, b1, b, b2)
    except RuntimeError as e:
        msg = str(e)
        if "numa_node" in msg or "NUMA" in msg:
            print(
                "forward failed: NUMA clone requires a NUMA-enabled build (USE_NUMA), "
                "runtime _set_numa_enabled(True), libnuma, and enough NUMA nodes "
                f"(graph uses numa_node 0..{NUM_TP_SHARDS - 1})."
            )
            print("RuntimeError:", msg)
            return
        raise
    else:
        print("forward ok, output shape:", tuple(out.shape))
    finally:
        if patched_numa and cpu_mod is not None:
            cpu_mod._set_numa_enabled(old_numa)


if __name__ == "__main__":
    main()
