#!/usr/bin/env python3
# mypy: ignore-errors
"""
Same Gemma decoder block (prefill) as ``simple_gemma_decoder_block_torch_compile_demo.py``
but with NUMA enabled and the Inductor ``_numa_attention_tp`` / ``_numa_mlp_tp`` passes
turned on — counterpart to ``simple_llm_numa_tp_torch_compile_demo.py``.

**Pattern caveat (read first).** The current passes match ``addmm`` (i.e. ``F.linear``
with bias) for the column/row linears, ``SDPA-MATH`` decomposition for attention, and
``addmm -> gelu -> addmm`` for MLP. Gemma-1B-style architecture uses:

  - Fused **single** ``qkv_proj`` linear (not three separate Wq/Wk/Wv addmms)
  - **No bias** on every linear (lowers to ``mm``, not ``addmm``)
  - **GeGLU** MLP (``gate * up`` then ``down``) instead of ``linear -> gelu -> linear``
  - **Manual** ``q @ k.T`` softmax (not ``F.scaled_dot_product_attention``)

So with the passes as they exist today, neither matcher fires on this graph and the
NUMA mode runs the standard kernels (the runtime is still NUMA-enabled, so generated
``clone(..., numa_node=...)`` paths from any matched non-Gemma graph would still work).
This script is the harness used to **drive future pass extensions** that handle:

  - fused ``qkv_proj`` (column-parallel along the concatenated Q/K/V output dim)
  - bias-free ``mm`` linears
  - GeGLU MLP (column-parallel ``gate_proj`` and ``up_proj``, row-parallel ``down_proj``)
  - manual-softmax attention (Gemma uses no SDPA call)

Until then, you can compare against the standard demo to verify that the runtime
overhead of NUMA-on without active passes is small.

Defaults match the Gemma-3 270M shapes (``hidden=1152``, ``intermediate=6912``,
``num_heads=4``, ``num_kv_heads=1``, ``head_dim=256``).

Example::

  python scripts/inductor/simple_gemma_decoder_block_numa_tp_torch_compile_demo.py \\
      --num-layers 1 --batch 1 --seq 512 \\
      --h-in 1152 --num-heads 4 --num-kv-heads 1 --head-dim 256 \\
      --mlp-hidden 6912 --warmup 5 --iters 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from simple_gemma_decoder_block_torch_compile_demo import (  # noqa: E402
    SimpleGemmaStack,
    _build_prefill_inputs,
    _mean_forward_seconds,
)

inductor_config.max_autotune_gemm_backends = "TRITON"


def _auto_num_tp_shards(num_heads: int, num_numa_nodes: int) -> int:
    cap = max(1, int(num_numa_nodes))
    for d in range(min(cap, num_heads), 0, -1):
        if num_heads % d == 0:
            return d
    return 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=512, help="Prefill sequence length")
    p.add_argument("--h-in", type=int, default=1152)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--mlp-hidden", type=int, default=6912)
    p.add_argument(
        "--num-tp-shards",
        type=int,
        default=None,
        help="NUMA TP world size; default auto from heads vs NUMA node count",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cpu")

    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        raise RuntimeError(
            "NUMA runtime hooks missing (torch._C._cpu._set_numa_enabled)."
        )
    if not hasattr(cpu_mod, "_is_numa_enabled") or not hasattr(cpu_mod, "_num_numa_nodes"):
        raise RuntimeError("NUMA query hooks missing (_is_numa_enabled, _num_numa_nodes).")

    old_numa = cpu_mod._is_numa_enabled()
    cpu_mod._set_numa_enabled(True)
    if not cpu_mod._is_numa_enabled():
        raise RuntimeError("Failed to enable NUMA via _set_numa_enabled(True)")
    n_nodes = int(cpu_mod._num_numa_nodes())

    if args.num_tp_shards is None:
        num_tp_shards = _auto_num_tp_shards(args.num_heads, n_nodes)
        tp_src = "auto"
    else:
        num_tp_shards = args.num_tp_shards
        tp_src = "cli"
        if n_nodes < num_tp_shards:
            raise RuntimeError(
                f"Need at least {num_tp_shards} NUMA nodes, got {n_nodes}."
            )

    H_qkv = args.num_heads * args.head_dim
    if args.num_heads % num_tp_shards != 0:
        raise ValueError(f"num_heads={args.num_heads} not divisible by shards={num_tp_shards}")
    if H_qkv % num_tp_shards != 0:
        raise ValueError(f"H_qkv={H_qkv} not divisible by shards={num_tp_shards}")
    if args.mlp_hidden % num_tp_shards != 0:
        raise ValueError(f"mlp_hidden={args.mlp_hidden} not divisible by shards={num_tp_shards}")

    model = SimpleGemmaStack(
        num_layers=args.num_layers,
        hidden_size=args.h_in,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.mlp_hidden,
    ).to(device).eval()
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)

    x, freqs_cis, kv_write_indices, kv_caches, mask = _build_prefill_inputs(
        args.batch, args.seq, args.h_in, args.num_kv_heads, args.head_dim,
        args.num_layers, device,
    )

    try:
        with inductor_config.patch(
            {
                "_numa_attention_tp": True,
                "_numa_attention_tp_world_size": num_tp_shards,
                "_numa_attention_tp_compute_shard": 0,
                "_numa_mlp_tp": True,
                "_numa_mlp_tp_world_size": num_tp_shards,
                "_numa_mlp_tp_compute_shard": 0,
            }
        ), torch.no_grad():
            compiled = torch.compile(model)
            y = compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)
            y = compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)

            def run() -> None:
                nonlocal y
                y = compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)

            mean_s = _mean_forward_seconds(run, warmup=args.warmup, iters=args.iters)
    finally:
        cpu_mod._set_numa_enabled(old_numa)

    print(f"SimpleGemmaStack(layers={args.num_layers}) output shape: {tuple(y.shape)}")
    print(
        f"Mean forward (warmup={args.warmup}, iters={args.iters}): "
        f"{mean_s * 1e3:.4f} ms/iter"
    )
    print(
        "Inductor: _numa_attention_tp and _numa_mlp_tp enabled, "
        f"world_size={num_tp_shards}, compute_shard=0"
    )
    if tp_src == "auto":
        print(f"num_tp_shards={num_tp_shards} (auto: divisor of num_heads={args.num_heads}, N<={n_nodes})")
    else:
        print(f"num_tp_shards={num_tp_shards} (from --num-tp-shards)")
    print(f"NUMA: _num_numa_nodes()={n_nodes}")
    print(
        f"Dims: B={args.batch} S={args.seq} H_in={args.h_in} "
        f"H={args.num_heads} KV_H={args.num_kv_heads} E={args.head_dim} "
        f"H_qkv={H_qkv} mlp_hidden={args.mlp_hidden}"
    )


if __name__ == "__main__":
    main()
