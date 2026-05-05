#!/usr/bin/env python3
# mypy: ignore-errors
r"""
Sweep BERT-Base / BERT-Large presets over (batch, sequence) and capture
**per-configuration** Torch compile debug trees (standard vs NUMA-TP).

Mirrors ``gpt2_multi_size_compile_debug_sweep.py`` artifact layout:

  * Result root: ``--result-dir`` (default: ``./bert_compile_debug_sweep``).
  * Under ``<result_dir>/torch_compile_debug_runs/`` each run uses a unique folder:
    ``M<model>_B<batch>_S<seq>_<standard|numa_tp>/``
  * With ``TORCH_COMPILE_DEBUG=1``, Dynamo writes:
    ``<that_folder>/torch_compile_debug/run_<timestamp>-pid_*/

Presets (``bert_multi_size_speedup_cli.py``):

  * ``base``  — 12 layers, d_model 768, 12 heads, FFN 3072 (~110M params)
  * ``large`` — 24 layers, d_model 1024, 16 heads, FFN 4096 (~340M params)

``sweep_manifest.jsonl`` rows match the GPT-2 sweep format; pass
``--omit-per-iter-latencies`` to drop ``latencies_ms``.

After **both** standard and NUMA-TP runs succeed for a given
(model, batch, sequence) configuration, the script appends one row to
``speedup_by_config.csv`` under ``--result-dir`` (unless disabled). Columns
are ``batch``, ``seq_length``, ``speedup`` (mean_standard_ms /
mean_numa_ms), ``mean_standard_ms``, ``mean_numa_ms``. If more than one
``--models`` entry is swept, a leading ``model`` column is added so rows
stay unique. The file is truncated at the start of each sweep; rows are
flushed after each config.

Optional ``--git-commit-push-each-speedup-row`` (see
``bert_speedup_csv_git_commit_push.py``) commits and pushes **only** that CSV
after each new row (CSV path must be inside your git clone and not ignored).

Default iteration space::

    models:     base, large
    batch:      1, 2, 4, 8
    seq:        128, 256, 512, 1024, 2048, 4096, 8192

Example::

    python scripts/inductor/bert_multi_size_compile_debug_sweep.py \\
        --result-dir ./my_bert_debug_artifacts

Example (tiered caps)::

    python scripts/inductor/bert_multi_size_compile_debug_sweep.py \\
        --result-dir ./bert_compile_debug_artifacts_tiered \\
        --batch-sizes 2 4 8 --tiered-max-seq-by-batch --world-size 2

See ``torch_compile_debug_runs/README.txt`` under the result dir.
"""
from __future__ import annotations

import argparse
import csv
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
_bert_path = _HERE / "bert_multi_size_speedup_cli.py"
_spec = importlib.util.spec_from_file_location(
    "bert_multi_size_speedup_cli", _bert_path
)
assert _spec and _spec.loader
_bert = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _bert
_spec.loader.exec_module(_bert)

MODEL_PRESETS = _bert.MODEL_PRESETS
BERTSizePreset = _bert.BERTSizePreset
BertEncoderModel = _bert.BertEncoderModel
DEF_VOCAB = _bert.DEF_VOCAB
DEF_CONTEXT = _bert.DEF_CONTEXT
_enable_numa = _bert._enable_numa
_run_path = _bert._run_path

_git_path = _HERE / "bert_speedup_csv_git_commit_push.py"
_spec_git = importlib.util.spec_from_file_location(
    "bert_speedup_csv_git_commit_push", _git_path
)
assert _spec_git and _spec_git.loader
_git_mod = importlib.util.module_from_spec(_spec_git)
sys.modules[_spec_git.name] = _git_mod
_spec_git.loader.exec_module(_git_mod)
commit_push_speedup_csv_only = _git_mod.commit_push_speedup_csv_only

_TORCH_DEBUG_RUNS_ROOT = "torch_compile_debug_runs"
_MANIFEST_NAME = "sweep_manifest.jsonl"
_SPEEDUP_CSV_NAME = "speedup_by_config.csv"


def _speedup_csv_header(include_model: bool) -> List[str]:
    base = [
        "batch",
        "seq_length",
        "speedup",
        "mean_standard_ms",
        "mean_numa_ms",
    ]
    return (["model"] + base) if include_model else base

_TORCH_DEBUG_RUNS_README = """# Per-configuration Torch compile debug (BERT multi-size sweep)
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


def _init_speedup_csv(path: str, *, include_model: bool) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_speedup_csv_header(include_model))
        fh.flush()
        os.fsync(fh.fileno())


def _append_speedup_csv(
    path: str,
    *,
    include_model: bool,
    model_key: str,
    batch: int,
    seq_length: int,
    mean_standard_ms: float,
    mean_numa_ms: float,
) -> None:
    if mean_numa_ms <= 0.0:
        raise ValueError("mean_numa_ms must be positive for speedup")
    speedup = mean_standard_ms / mean_numa_ms
    row: List[Any] = [
        batch,
        seq_length,
        speedup,
        mean_standard_ms,
        mean_numa_ms,
    ]
    if include_model:
        row = [model_key] + row
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def _arch_for_preset(model_key: str) -> Dict[str, int]:
    p: BERTSizePreset = MODEL_PRESETS[model_key]
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
    default_models = ["base", "large"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Root for torch_compile_debug_runs/ and sweep_manifest.jsonl. "
        "Default: <cwd>/bert_compile_debug_sweep",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model presets to sweep (default: base large).",
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
    p.add_argument("--type-vocab-size", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=DEF_VOCAB)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup forward passes per path (default: 5; use higher if needed).",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations per path (default: 50; use 1 for minimal timing / artifacts).",
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
    p.add_argument(
        "--speedup-csv",
        type=str,
        default=None,
        help=(
            "Path for incremental speedup CSV (default: "
            f"<result-dir>/{_SPEEDUP_CSV_NAME}). "
            "Appends one row per (model, B, S) after both standard and NUMA-TP succeed."
        ),
    )
    p.add_argument(
        "--no-speedup-csv",
        action="store_true",
        help="Do not write speedup_by_config.csv (or --speedup-csv).",
    )
    p.add_argument(
        "--git-commit-push-each-speedup-row",
        action="store_true",
        help=(
            "After each new speedup row: git add that CSV only, commit, push to "
            "--git-remote / --git-branch. Requires repo clone; file must not be gitignored."
        ),
    )
    p.add_argument(
        "--git-repo-root",
        type=str,
        default=None,
        help="Git repository root (default: infer via git rev-parse from CSV directory).",
    )
    p.add_argument(
        "--git-remote",
        type=str,
        default="origin",
        help="Remote name for git push (default: origin).",
    )
    p.add_argument(
        "--git-branch",
        type=str,
        default=None,
        help="Branch to push (default: current branch).",
    )
    p.add_argument(
        "--git-continue-on-error",
        action="store_true",
        help="If git commit/push fails, log and continue (default: exit immediately).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.skip_numa_tp and args.skip_standard:
        print("Both --skip-numa-tp and --skip-standard; nothing to do.", file=sys.stderr)
        return 2
    if args.git_commit_push_each_speedup_row and args.no_speedup_csv:
        print(
            "--git-commit-push-each-speedup-row requires a speedup CSV; "
            "omit --no-speedup-csv.",
            file=sys.stderr,
        )
        return 2

    result_dir = os.path.abspath(
        args.result_dir or os.path.join(os.getcwd(), "bert_compile_debug_sweep")
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

    speedup_csv_include_model = len(args.models) > 1

    manifest_path = os.path.join(result_dir, _MANIFEST_NAME)
    with open(manifest_path, "w") as fh:
        fh.write(
            json.dumps(
                {
                    "_record": "meta",
                    "format": "bert_compile_debug_sweep_v1",
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
                    "speedup_csv": (
                        os.path.abspath(
                            args.speedup_csv
                            or os.path.join(result_dir, _SPEEDUP_CSV_NAME)
                        )
                        if not args.no_speedup_csv
                        else None
                    ),
                    "speedup_csv_include_model_column": speedup_csv_include_model,
                    "git_commit_push_each_speedup_row": args.git_commit_push_each_speedup_row,
                    "git_repo_root": (
                        os.path.abspath(args.git_repo_root)
                        if args.git_repo_root
                        else None
                    ),
                    "git_remote": args.git_remote,
                    "git_branch": args.git_branch,
                    "git_continue_on_error": args.git_continue_on_error,
                }
            )
            + "\n"
        )
        fh.flush()
        os.fsync(fh.fileno())

    speedup_csv_path: str | None = None
    if not args.no_speedup_csv:
        speedup_csv_path = os.path.abspath(
            args.speedup_csv or os.path.join(result_dir, _SPEEDUP_CSV_NAME)
        )
        _init_speedup_csv(speedup_csv_path, include_model=speedup_csv_include_model)

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
            model = BertEncoderModel(
                vocab_size=args.vocab_size,
                context_size=context_size,
                type_vocab_size=args.type_vocab_size,
                d_model=arch["d_model"],
                num_layers=arch["num_layers"],
                num_heads=arch["num_heads"],
                head_dim=arch["head_dim"],
                d_ffn=arch["d_ffn"],
            ).eval()
            input_ids = torch.randint(
                0, args.vocab_size, (B, S), dtype=torch.long
            )

            mean_standard_ms: float | None = None
            mean_numa_ms: float | None = None
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
                    if mode == "standard":
                        mean_standard_ms = timing.mean_ms
                    elif mode == "numa_tp":
                        mean_numa_ms = timing.mean_ms
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
                    if mode == "standard":
                        mean_standard_ms = None
                    elif mode == "numa_tp":
                        mean_numa_ms = None
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

            if (
                speedup_csv_path is not None
                and "standard" in modes
                and "numa_tp" in modes
                and mean_standard_ms is not None
                and mean_numa_ms is not None
                and mean_numa_ms > 0.0
            ):
                try:
                    _append_speedup_csv(
                        speedup_csv_path,
                        include_model=speedup_csv_include_model,
                        model_key=model_key,
                        batch=B,
                        seq_length=S,
                        mean_standard_ms=mean_standard_ms,
                        mean_numa_ms=mean_numa_ms,
                    )
                except Exception as e:
                    print(
                        f"  WARN speedup_csv append failed: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    sp = mean_standard_ms / mean_numa_ms
                    print(
                        f"  speedup_csv: {model_key} B={B} S={S} "
                        f"speedup={sp:.4f} (standard/numa ms)",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.git_commit_push_each_speedup_row:
                        try:
                            commit_push_speedup_csv_only(
                                speedup_csv_path,
                                repo_root=args.git_repo_root,
                                remote=args.git_remote,
                                branch=args.git_branch,
                                commit_message=(
                                    f"BERT sweep speedup: {model_key} B={B} S={S} "
                                    f"speedup={sp:.4f}"
                                ),
                            )
                            print(
                                "  git: committed and pushed speedup CSV only.",
                                file=sys.stderr,
                                flush=True,
                            )
                        except Exception as ge:
                            if args.git_continue_on_error:
                                print(
                                    f"  WARN git commit/push: {ge}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            else:
                                print(
                                    f"  FAIL git commit/push: {ge}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                raise SystemExit(1) from ge

    total_s = time.perf_counter() - t0_all
    tail = (
        f"Done. ok={ok} fail={fail} wall_s={total_s:.1f}\n"
        f"Manifest: {manifest_path}\n"
        f"Debug trees: {os.path.join(result_dir, _TORCH_DEBUG_RUNS_ROOT, '')}"
    )
    if speedup_csv_path is not None:
        tail += f"\nSpeedup CSV: {speedup_csv_path}"
    print(tail, file=sys.stderr)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
