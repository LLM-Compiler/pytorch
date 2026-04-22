#!/usr/bin/env python3
# mypy: ignore-errors
"""Side-by-side torch.profiler breakdown: standard compile vs NUMA TP compile.

Runs the same ``SimpleLmStack`` (imported from the NUMA demo) twice — once with
NUMA passes disabled, once enabled — at identical shapes, then prints per-op
self-CPU-time tables sorted descending so it's obvious where the remaining
NUMA gap goes (GEMM compute vs clone/alloc vs all-reduce reassembly vs pool
dispatch).

Run after activating triton_cpu_integrated. Example:

  python scripts/inductor/profile_numa_vs_standard.py \\
      --num-layers 1 --batch 1 --seq 512 --h-in 1152 --num-heads 4 \\
      --head-dim 256 --mlp-hidden 6912 --warmup 3 --iters 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simple_llm_numa_tp_torch_compile_demo import (  # type: ignore[import-not-found]
    SimpleLmStack,
    _auto_num_tp_shards,
)

inductor_config.max_autotune_gemm_backends = "TRITON"
inductor_config.max_autotune = True


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--h-in", type=int, default=1152)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--mlp-hidden", type=int, default=6912)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--top", type=int, default=25, help="Rows in each profile table")
    return p.parse_args()


def _build_model(args: argparse.Namespace) -> tuple[SimpleLmStack, torch.Tensor]:
    torch.manual_seed(0)
    model = SimpleLmStack(
        args.num_layers,
        args.num_heads,
        args.head_dim,
        args.h_in,
        args.mlp_hidden,
    ).to("cpu")
    model.set_batch_seq(args.batch, args.seq)
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)
    x = torch.randn(args.batch * args.seq, args.h_in, device="cpu")
    return model, x


def _profile_compile(
    label: str,
    model: SimpleLmStack,
    x: torch.Tensor,
    *,
    numa: bool,
    num_tp_shards: int,
    warmup: int,
    iters: int,
    top: int,
) -> None:
    patch = {
        "_numa_attention_tp": numa,
        "_numa_mlp_tp": numa,
    }
    if numa:
        patch.update(
            {
                "_numa_attention_tp_world_size": num_tp_shards,
                "_numa_attention_tp_compute_shard": 0,
                "_numa_mlp_tp_world_size": num_tp_shards,
                "_numa_mlp_tp_compute_shard": 0,
            }
        )

    with inductor_config.patch(patch):
        compiled = torch.compile(model)
        # Warmup (compile + cache).
        for _ in range(max(2, warmup)):
            compiled(x)

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(iters):
                compiled(x)

    print(f"\n===== {label} (iters={iters}) =====")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=top,
        )
    )


def main() -> None:
    args = _parse()

    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None:
        raise RuntimeError("torch._C._cpu missing — rebuild with USE_NUMA=1")
    old_numa = cpu_mod._is_numa_enabled()
    cpu_mod._set_numa_enabled(True)
    n_nodes = int(cpu_mod._num_numa_nodes())
    num_tp_shards = _auto_num_tp_shards(args.num_heads, n_nodes)

    try:
        model_std, x_std = _build_model(args)
        _profile_compile(
            "STANDARD (NUMA passes OFF)",
            model_std,
            x_std,
            numa=False,
            num_tp_shards=num_tp_shards,
            warmup=args.warmup,
            iters=args.iters,
            top=args.top,
        )

        # Fresh model so compile caches don't collide.
        model_numa, x_numa = _build_model(args)
        _profile_compile(
            f"NUMA TP (world_size={num_tp_shards})",
            model_numa,
            x_numa,
            numa=True,
            num_tp_shards=num_tp_shards,
            warmup=args.warmup,
            iters=args.iters,
            top=args.top,
        )
    finally:
        cpu_mod._set_numa_enabled(old_numa)


if __name__ == "__main__":
    main()
