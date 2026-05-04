#!/usr/bin/env python3
# mypy: ignore-errors
r"""
Sweep GPT-2 Pre-LN presets (117M–1542M) over (batch, sequence) and capture
**per-configuration** Torch compile debug trees (standard vs NUMA-TP).

Mirrors ``simple_llm_decoder_block_paper_bench.py`` artifact layout:

  * Result root: ``--result-dir`` (default: ``./gpt2_compile_debug_sweep``).
  * Under ``<result_dir>/torch_compile_debug_runs/`` each run uses a unique folder:
    ``M<model>_B<batch>_S<seq>_<standard|numa_tp>/``
  * With ``TORCH_COMPILE_DEBUG=1``, Dynamo writes:
    ``<that_folder>/torch_compile_debug/run_<timestamp>-pid_*/

Environment is restored after each compile path (same pattern as the paper bench).

``sweep_manifest.jsonl`` rows (after the meta line) include ``latencies_ms``: one
float per timed iteration (milliseconds), in **chronological** order (length
``--iters`` after warmup). Pass ``--omit-per-iter-latencies`` to drop that array.

Default iteration space::

    models:     117m, 345m, 762m, 1542m
    batch:      1, 2, 4, 8
    seq:        128, 256, 512, 1024, 2048, 4096, 8192

With ``--tiered-max-seq-by-batch`` (and default ``--tiered-seq-base 8192``), for each
batch `B` only sequence lengths `S <= 8192 // B` are run (e.g. B=2 up to 4096,
B=4 up to 2048, B=8 up to 1024, B=1 up to 8192). Use ``--batch-sizes 2 4 8`` to
omit batch 1.

Example::

    python scripts/inductor/gpt2_multi_size_compile_debug_sweep.py \\
        --result-dir ./my_gpt2_debug_artifacts --warmup 1 --iters 1

Example (tiered caps, new folder — does not touch prior ``./gpt2_compile_debug_artifacts``)::

    python scripts/inductor/gpt2_multi_size_compile_debug_sweep.py \\
        --result-dir ./gpt2_compile_debug_artifacts_tiered_seq8192divB_bs2_4_8_w2 \\
        --batch-sizes 2 4 8 --tiered-max-seq-by-batch --world-size 2 \\
        --warmup 5 --iters 50

See ``torch_compile_debug_runs/README.txt`` under the result dir.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Load sibling module without requiring ``scripts.inductor`` to be a package.
_HERE = Path(__file__).resolve().parent
_g2_path = _HERE / "gpt2_multi_size_speedup_cli.py"
_spec = importlib.util.spec_from_file_location("gpt2_multi_size_speedup_cli", _g2_path)
assert _spec and _spec.loader
_g2 = importlib.util.module_from_spec(_spec)
# Required for @dataclass on Python 3.9: dataclasses looks up cls.__module__ in sys.modules.
sys.modules[_spec.name] = _g2
_spec.loader.exec_module(_g2)

MODEL_PRESETS = _g2.MODEL_PRESETS
GPT2PreLN = _g2.GPT2PreLN
DEF_VOCAB = _g2.DEF_VOCAB
DEF_CONTEXT = _g2.DEF_CONTEXT
_enable_numa = _g2._enable_numa
_run_path = _g2._run_path

_TORCH_DEBUG_RUNS_ROOT = "torch_compile_debug_runs"
_MANIFEST_NAME = "sweep_manifest.jsonl"

_TORCH_DEBUG_RUNS_README = """# Per-configuration Torch compile debug (GPT-2 multi-size sweep)
# Each subfolder name encodes: M<model>_B<batch>_S<seq>_<standard|numa_tp>
# With TORCH_COMPILE_DEBUG=1, Dynamo writes to:
#   <that_subfolder>/torch_compile_debug/run_YYYY_mm_dd_...-pid_*/
# The parent script restores env after each config.
""".lstrip()


def _ensure_debug_runs_readme(result_dir: str) -> None:
    parent = os.path.join(result_dir, _TORCH_DEBUG_RUNS_ROOT)
    os.makedirs(parent, exist_ok=True)
    readme = os.path.join(parent, "README.txt")
    if not os.path.isfile(readme):
        with open(readme, "w") as fh:
            fh.write(_TORCH_DEBUG_RUNS_README)


def _per_config_relpath(
    model_key: str, batch: int, seq: int, mode: str
) -> str:
    safe_model = model_key.replace("/", "_")
    return os.path.join(
        _TORCH_DEBUG_RUNS_ROOT,
        f"M{safe_model}_B{batch}_S{seq}_{mode}",
    )


def _append_manifest(result_dir: str, record: Dict[str, Any]) -> None:
    path = os.path.join(result_dir, _MANIFEST_NAME)
    with open(path, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _arch_for_preset(model_key: str) -> Dict[str, int]:
    p = MODEL_PRESETS[model_key]
    return {
        "num_layers": p.num_layers,
        "d_model": p.d_model,
        "num_heads": p.num_heads,
        "head_dim": p.head_dim,
        "d_ffn": p.d_ffn,
    }


def _batch_seq_pairs(
    batch_sizes: List[int],
    seq_lens: List[int],
    *,
    tiered_max_seq_by_batch: bool,
    tiered_seq_base: int,
) -> List[Tuple[int, int]]:
    """(B, S) pairs in nested-loop order: outer B, inner S (filtered if tiered)."""
    pairs: List[Tuple[int, int]] = []
    for B in batch_sizes:
        if tiered_max_seq_by_batch:
            cap = tiered_seq_base // B
        else:
            cap = max(seq_lens) if seq_lens else 0
        for S in seq_lens:
            if S <= cap:
                pairs.append((B, S))
    return pairs


def _parse_args() -> argparse.Namespace:
    default_models = ["117m", "345m", "762m", "1542m"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Root for torch_compile_debug_runs/ and sweep_manifest.jsonl. "
        "Default: <cwd>/gpt2_compile_debug_sweep",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model presets to sweep (default: all four).",
    )
    p.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Batch sizes B (default: 1 2 4 8).",
    )
    p.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[128, 256, 512, 1024, 2048, 4096, 8192],
        help="Sequence lengths S (default: 128 ... 8192).",
    )
    p.add_argument(
        "--tiered-max-seq-by-batch",
        action="store_true",
        help=(
            "For each batch B, only run S with S <= (tiered_seq_base // B) "
            "(default base 8192: B=2→4096, B=4→2048, B=8→1024, B=1→8192)."
        ),
    )
    p.add_argument(
        "--tiered-seq-base",
        type=int,
        default=8192,
        help="Numerator for tiered cap S <= base // B (default: 8192).",
    )
    p.add_argument("--vocab-size", type=int, default=DEF_VOCAB)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup forward passes per path (default: 1; use higher if needed).",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Timed iterations per path (default: 1; artifacts only need one compile).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tp-lm-head",
        action="store_true",
        help="Pass through to NUMA-TP Inductor LM-head flag (model has no LM head).",
    )
    p.add_argument(
        "--max-autotune",
        action="store_true",
        help="Enable Inductor max_autotune for matmuls in both paths.",
    )
    p.add_argument(
        "--skip-numa-tp",
        action="store_true",
        help="Only run standard path (still creates standard/ debug dirs only).",
    )
    p.add_argument(
        "--skip-standard",
        action="store_true",
        help="Only run NUMA-TP path.",
    )
    p.add_argument(
        "--omit-per-iter-latencies",
        action="store_true",
        help="Do not record latencies_ms (one float per --iters) in sweep_manifest.jsonl.",
    )
    p.add_argument(
        "--no-compile-debug",
        action="store_true",
        help=(
            "Do not set TORCH_COMPILE_DEBUG=1 or write per-config torch_compile_debug/ trees "
            "(faster; use when only timing / crash reproduction matters)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.skip_numa_tp and args.skip_standard:
        print("Both --skip-numa-tp and --skip-standard; nothing to do.", file=sys.stderr)
        return 2

    result_dir = os.path.abspath(
        args.result_dir or os.path.join(os.getcwd(), "gpt2_compile_debug_sweep")
    )
    os.makedirs(result_dir, exist_ok=True)
    _ensure_debug_runs_readme(result_dir)

    batch_seq_pairs = _batch_seq_pairs(
        list(args.batch_sizes),
        list(args.seq_lens),
        tiered_max_seq_by_batch=args.tiered_max_seq_by_batch,
        tiered_seq_base=args.tiered_seq_base,
    )
    if not batch_seq_pairs:
        print(
            "No (batch, seq) pairs after filtering. Check --batch-sizes, --seq-lens, "
            "--tiered-max-seq-by-batch, and --tiered-seq-base.",
            file=sys.stderr,
        )
        return 2

    # Meta line (overwrite manifest each run)
    manifest_path = os.path.join(result_dir, _MANIFEST_NAME)
    with open(manifest_path, "w") as fh:
        fh.write(
            json.dumps(
                {
                    "_record": "meta",
                    "format": "gpt2_compile_debug_sweep_v1",
                    "hostname": platform.node(),
                    "torch": torch.__version__,
                    "result_dir": result_dir,
                    "models": args.models,
                    "batch_sizes": args.batch_sizes,
                    "seq_lens": args.seq_lens,
                    "tiered_max_seq_by_batch": args.tiered_max_seq_by_batch,
                    "tiered_seq_base": (
                        args.tiered_seq_base if args.tiered_max_seq_by_batch else None
                    ),
                    "batch_seq_pairs": batch_seq_pairs,
                    "world_size": args.world_size,
                }
            )
            + "\n"
        )
        fh.flush()
        os.fsync(fh.fileno())

    _enable_numa()
    torch.manual_seed(args.seed)

    modes: List[str] = []
    if not args.skip_standard:
        modes.append("standard")
    if not args.skip_numa_tp:
        modes.append("numa_tp")

    t0_all = time.perf_counter()
    ok = 0
    fail = 0

    for model_key in args.models:
        arch = _arch_for_preset(model_key)
        H_qkv = arch["num_heads"] * arch["head_dim"]
        if H_qkv % args.world_size != 0:
            msg = (
                f"H_qkv={H_qkv} not divisible by world_size={args.world_size}"
            )
            print(f"[skip all {model_key}] {msg}", file=sys.stderr)
            _append_manifest(
                result_dir,
                {
                    "ok": False,
                    "model": model_key,
                    "batch": None,
                    "seq": None,
                    "mode": None,
                    "error": msg,
                    "skipped": "whole_model",
                },
            )
            fail += 1
            continue
        if arch["d_ffn"] % args.world_size != 0:
            msg = (
                f"d_ffn={arch['d_ffn']} not divisible by "
                f"world_size={args.world_size}"
            )
            print(f"[skip all {model_key}] {msg}", file=sys.stderr)
            _append_manifest(
                result_dir,
                {
                    "ok": False,
                    "model": model_key,
                    "batch": None,
                    "seq": None,
                    "mode": None,
                    "error": msg,
                    "skipped": "whole_model",
                },
            )
            fail += 1
            continue

        for B, S in batch_seq_pairs:
            context_size = max(S, DEF_CONTEXT)
            model = GPT2PreLN(
                vocab_size=args.vocab_size,
                context_size=context_size,
                d_model=arch["d_model"],
                num_layers=arch["num_layers"],
                num_heads=arch["num_heads"],
                head_dim=arch["head_dim"],
                d_ffn=arch["d_ffn"],
            ).eval()
            input_ids = torch.randint(
                0, args.vocab_size, (B, S), dtype=torch.long
            )

            for mode in modes:
                numa = mode == "numa_tp"
                rel = _per_config_relpath(model_key, B, S, mode)
                staging = os.path.join(result_dir, rel)
                label = f"{model_key} {mode} B={B} S={S}"
                print(f"[run] {label} -> {staging}", file=sys.stderr, flush=True)
                t_cfg = time.perf_counter()
                try:
                    use_cdbg = not args.no_compile_debug
                    timing, _out, info = _run_path(
                        model,
                        input_ids,
                        numa=numa,
                        world_size=args.world_size,
                        warmup=args.warmup,
                        iters=args.iters,
                        lm_head=args.tp_lm_head,
                        profile=False,
                        max_autotune=args.max_autotune,
                        compile_debug=use_cdbg,
                        compile_debug_staging=staging if use_cdbg else None,
                        record_iteration_latencies=not args.omit_per_iter_latencies,
                    )
                    elapsed = time.perf_counter() - t_cfg
                    ok += 1
                    row = {
                        "ok": True,
                        "model": model_key,
                        "batch": B,
                        "seq": S,
                        "mode": mode,
                        "mean_ms": timing.mean_ms,
                        "median_ms": timing.median_ms,
                        "min_ms": timing.min_ms,
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "staging_dir": staging,
                        "numa_calls": info.get("numa_calls"),
                        "cache_dir": info.get("cache_dir"),
                        "wall_s": elapsed,
                    }
                    if not args.omit_per_iter_latencies:
                        row["latencies_ms"] = info.get("latencies_ms", [])
                    _append_manifest(result_dir, row)
                    print(
                        f"  OK mean_ms={timing.mean_ms:.3f} "
                        f"numa_calls={info.get('numa_calls')}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as e:
                    fail += 1
                    print(f"  FAIL {e}", file=sys.stderr, flush=True)
                    _append_manifest(
                        result_dir,
                        {
                            "ok": False,
                            "model": model_key,
                            "batch": B,
                            "seq": S,
                            "mode": mode,
                            "error": str(e),
                            "staging_dir": staging,
                        },
                    )

    total_s = time.perf_counter() - t0_all
    print(
        f"Done. ok={ok} fail={fail} wall_s={total_s:.1f}\n"
        f"Manifest: {manifest_path}\n"
        f"Debug trees: {os.path.join(result_dir, _TORCH_DEBUG_RUNS_ROOT, '')}",
        file=sys.stderr,
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
