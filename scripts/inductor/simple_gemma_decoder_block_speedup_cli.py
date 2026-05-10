#!/usr/bin/env python3
# mypy: ignore-errors
r"""
Single-config Gemma decoder-block (prefill) speedup CLI: runs **standard** then
**NUMA-TP** for one ``(B, S, L)`` configuration and prints only the speedup on stdout::

    speedup = mean_latency_ms_standard / mean_latency_ms_numa

This is the Gemma-architecture counterpart to ``simple_llm_decoder_block_speedup_cli.py``.
NUMA must succeed; errors go to stderr with ``SystemExit(1)``.

Self-contained: model + run_one_mode are inlined (no separate paper-bench module like
the GPT-2 path). The decoder block is imported from
``simple_gemma_decoder_block_torch_compile_demo`` to keep one source of truth.

Example::

  python scripts/inductor/simple_gemma_decoder_block_speedup_cli.py \\
      --batch-size 1 --seq-length 128 --num-layers 1 --warmup 10 --iters 50
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import List, Literal, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch
import torch._dynamo
import torch.nn as nn
from torch._inductor import config as inductor_config

from simple_gemma_decoder_block_torch_compile_demo import (  # noqa: E402
    SimpleGemmaStack,
    _build_prefill_inputs,
)

inductor_config.max_autotune_gemm_backends = "TRITON"

BenchMode = Literal["standard", "numa_tp"]


class NumaRuntimeError(RuntimeError):
    pass


def _auto_num_tp_shards(num_heads: int, num_numa_nodes: int) -> int:
    cap = max(1, int(num_numa_nodes))
    for d in range(min(cap, num_heads), 0, -1):
        if num_heads % d == 0:
            return d
    return 1


def _time_per_iter(run, *, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        run()
    out: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def run_one_mode(
    mode: BenchMode,
    *,
    num_layers: int,
    B: int,
    S: int,
    h_in: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mlp_hidden: int,
    num_tp_shards: Optional[int],
    warmup: int,
    iters: int,
) -> Tuple[float, int, int]:
    """Returns ``(mean_ms, shards_used, n_nodes)``."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    H_qkv = num_heads * head_dim

    cpu_mod = getattr(torch._C, "_cpu", None)
    old_numa: Optional[bool] = None
    n_nodes = 1
    if cpu_mod is not None and hasattr(cpu_mod, "_num_numa_nodes"):
        n_nodes = int(cpu_mod._num_numa_nodes())

    if mode == "numa_tp":
        if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
            raise NumaRuntimeError("NUMA runtime hooks missing.")
        if not hasattr(cpu_mod, "_is_numa_enabled") or not hasattr(cpu_mod, "_num_numa_nodes"):
            raise NumaRuntimeError("NUMA query hooks missing.")
        old_numa = cpu_mod._is_numa_enabled()
        cpu_mod._set_numa_enabled(True)
        if not cpu_mod._is_numa_enabled():
            cpu_mod._set_numa_enabled(old_numa)
            raise NumaRuntimeError("Failed to enable NUMA via _set_numa_enabled(True).")
        n_nodes = int(cpu_mod._num_numa_nodes())
        shards = num_tp_shards if num_tp_shards is not None else _auto_num_tp_shards(num_heads, n_nodes)
        if n_nodes < shards:
            raise RuntimeError(f"num_tp_shards={shards} but only {n_nodes} NUMA node(s)")
        if num_heads % shards != 0:
            raise ValueError(f"num_heads % shards ({shards}) != 0")
        if H_qkv % shards != 0:
            raise ValueError(f"H_qkv % shards ({shards}) != 0")
        if mlp_hidden % shards != 0:
            raise ValueError(f"mlp_hidden % shards ({shards}) != 0")
    else:
        shards = 1

    try:
        model = SimpleGemmaStack(
            num_layers=num_layers,
            hidden_size=h_in,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=mlp_hidden,
        ).to(device).eval()
        for p in model.parameters():
            nn.init.normal_(p, std=0.02)

        x, freqs_cis, kv_write_indices, kv_caches, mask = _build_prefill_inputs(
            B, S, h_in, num_kv_heads, head_dim, num_layers, device,
        )

        if mode == "standard":
            ctx = inductor_config.patch(
                {"_numa_attention_tp": False, "_numa_mlp_tp": False, "fx_graph_cache": False}
            )
        else:
            ctx = inductor_config.patch(
                {
                    "_numa_attention_tp": True,
                    "_numa_attention_tp_world_size": shards,
                    "_numa_attention_tp_compute_shard": 0,
                    "_numa_mlp_tp": True,
                    "_numa_mlp_tp_world_size": shards,
                    "_numa_mlp_tp_compute_shard": 0,
                    "fx_graph_cache": False,
                }
            )

        with ctx, torch.no_grad():
            torch._dynamo.reset()
            compiled = torch.compile(model)

            def run() -> None:
                compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)

            lats = _time_per_iter(run, warmup=warmup, iters=iters)

        return statistics.fmean(lats), shards, n_nodes
    finally:
        if mode == "numa_tp" and cpu_mod is not None and old_numa is not None:
            cpu_mod._set_numa_enabled(old_numa)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Print NUMA vs standard mean-latency speedup for one (B, S, L) Gemma "
            "decoder-block prefill config (stdout only; errors on stderr)."
        )
    )
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--seq-length", type=int, required=True)
    p.add_argument("--num-layers", type=int, required=True)
    p.add_argument("--h-in", type=int, default=1152)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--mlp-hidden", type=int, default=6912)
    p.add_argument("--num-tp-shards", type=int, default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    kw = dict(
        num_layers=args.num_layers,
        B=args.batch_size,
        S=args.seq_length,
        h_in=args.h_in,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        mlp_hidden=args.mlp_hidden,
        num_tp_shards=args.num_tp_shards,
        warmup=args.warmup,
        iters=args.iters,
    )
    try:
        std_mean, _, _ = run_one_mode("standard", **kw)
        numa_mean, _, _ = run_one_mode("numa_tp", **kw)
    except NumaRuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e
    except Exception as e:
        print(
            f"benchmark failed L={args.num_layers} B={args.batch_size} S={args.seq_length}: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    if not (std_mean == std_mean and std_mean > 0):
        print("invalid standard mean_ms", file=sys.stderr)
        raise SystemExit(1)
    if not (numa_mean == numa_mean and numa_mean > 0):
        print("invalid numa mean_ms", file=sys.stderr)
        raise SystemExit(1)

    speedup = std_mean / numa_mean
    if not (speedup == speedup):
        print("speedup is nan", file=sys.stderr)
        raise SystemExit(1)
    print(speedup)


if __name__ == "__main__":
    main()
