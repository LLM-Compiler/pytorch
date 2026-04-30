#!/usr/bin/env python3
# mypy: ignore-errors
r"""
Single-configuration decoder-block benchmark: runs **standard** then **NUMA-TP**
(same stack as ``simple_llm_decoder_block_paper_bench.py``) and prints **only** the
speedup on stdout:

    speedup = mean_latency_ms_standard / mean_latency_ms_numa

NUMA must succeed (same checks as the paper bench). Compile-debug, CSV/JSON, and
progress lines are omitted so stdout is suitable for scripting.

Example::

  python scripts/inductor/simple_llm_decoder_block_speedup_cli.py \\
      --batch-size 1 --seq-length 128 --num-layers 1 --warmup 10 --iters 50
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import simple_llm_decoder_block_paper_bench as paper  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Print NUMA vs standard mean-latency speedup for one (B, S, L) config "
            "(stdout only; errors on stderr)."
        )
    )
    p.add_argument("--batch-size", type=int, required=True, help="Batch size B")
    p.add_argument("--seq-length", type=int, required=True, help="Sequence length S")
    p.add_argument("--num-layers", type=int, required=True, help="Decoder depth L")
    p.add_argument("--h-in", type=int, default=1152)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--mlp-hidden", type=int, default=6912)
    p.add_argument(
        "--num-tp-shards",
        type=int,
        default=None,
        help="NUMA TP world size; default auto from heads vs NUMA node count",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    B, S, L = args.batch_size, args.seq_length, args.num_layers
    kw = dict(
        num_layers=L,
        B=B,
        S=S,
        h_in=args.h_in,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        mlp_hidden=args.mlp_hidden,
        num_tp_shards=args.num_tp_shards,
        warmup=args.warmup,
        iters=args.iters,
    )
    try:
        std_summary, _, _, _, _ = paper.run_one_mode("standard", **kw)
        numa_summary, _, _, _, _ = paper.run_one_mode("numa_tp", **kw)
    except paper.NumaRuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e
    except Exception as e:
        print(f"benchmark failed L={L} B={B} S={S}: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    base = std_summary["mean_ms"]
    numa_mean = numa_summary["mean_ms"]
    if not (base == base and base > 0):
        print("invalid standard mean_ms", file=sys.stderr)
        raise SystemExit(1)
    if not (numa_mean == numa_mean and numa_mean > 0):
        print("invalid numa mean_ms", file=sys.stderr)
        raise SystemExit(1)

    speedup = base / numa_mean
    if not (speedup == speedup):
        print("speedup is nan", file=sys.stderr)
        raise SystemExit(1)

    print(speedup)


if __name__ == "__main__":
    main()
