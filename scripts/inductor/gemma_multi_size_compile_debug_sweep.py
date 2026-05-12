#!/usr/bin/env python3
# mypy: ignore-errors
r"""
Sweep Gemma decoder-block presets over (batch, sequence) and capture
**per-configuration** Torch compile debug trees (standard vs NUMA-TP).

Gemma counterpart to ``gpt2_multi_size_compile_debug_sweep.py``. Reuses the
``SimpleGemmaStack`` decoder + ``_build_prefill_inputs`` helper from
``simple_gemma_decoder_block_torch_compile_demo.py`` (one source of truth for
the Gemma reference graph).

Artifact layout (matches the GPT-2 sweep)::

  * Result root: ``--result-dir`` (default: ``./gemma_compile_debug_sweep``).
  * Per-config debug staging: ``<result_dir>/torch_compile_debug_runs/
    M<model>_B<batch>_S<seq>_<standard|numa_tp>/``
  * With ``TORCH_COMPILE_DEBUG=1``, Dynamo writes:
    ``<that_folder>/torch_compile_debug/run_<timestamp>-pid_*/``
  * Manifest ``sweep_manifest.jsonl`` — meta line first, then one row per
    (model, batch, seq, mode). Rows include ``latencies_ms`` (one float per
    timed iteration, chronological) unless ``--omit-per-iter-latencies``.

Default iteration space::

    models:     270m, 1b
    batch:      1, 2, 4, 8
    seq:        128, 256, 512, 1024, 2048, 4096, 8192

With ``--tiered-max-seq-by-batch`` (default base 8192), for each batch B only
``S <= 8192 // B`` runs (B=2→4096, B=4→2048, B=8→1024, B=1→8192).

Larger presets (``4b``, ``12b``, ``27b``) are available via ``--models`` but
not in the default list — they have ``num_kv_heads > 2`` and will run with K/V
replicated under the current Gemma attention NUMA-TP pass (the
cache-placeholder sharding required for genuine column-parallel K/V is not yet
wired up).

Example::

    python scripts/inductor/gemma_multi_size_compile_debug_sweep.py \\
        --result-dir ./my_gemma_debug_artifacts

    python scripts/inductor/gemma_multi_size_compile_debug_sweep.py \\
        --result-dir ./gemma_debug_tiered_w2 \\
        --batch-sizes 1 2 4 8 --tiered-max-seq-by-batch --world-size 2

See ``torch_compile_debug_runs/README.txt`` under the result dir.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import os
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch._dynamo
import torch.nn as nn
from torch._inductor import config as inductor_config

inductor_config.max_autotune_gemm_backends = "TRITON"

# Load sibling Gemma demo module without requiring scripts.inductor as a package.
_HERE = Path(__file__).resolve().parent
_demo_path = _HERE / "simple_gemma_decoder_block_torch_compile_demo.py"
_spec = importlib.util.spec_from_file_location(
    "simple_gemma_decoder_block_torch_compile_demo", _demo_path
)
assert _spec and _spec.loader
_demo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _demo
_spec.loader.exec_module(_demo)

SimpleGemmaStack = _demo.SimpleGemmaStack
_build_prefill_inputs = _demo._build_prefill_inputs

_TORCH_DEBUG_RUNS_ROOT = "torch_compile_debug_runs"
_MANIFEST_NAME = "sweep_manifest.jsonl"

_TORCH_DEBUG_RUNS_README = """# Per-configuration Torch compile debug (Gemma multi-size sweep)
# Each subfolder name encodes: M<model>_B<batch>_S<seq>_<standard|numa_tp>
# With TORCH_COMPILE_DEBUG=1, Dynamo writes to:
#   <that_subfolder>/torch_compile_debug/run_YYYY_mm_dd_...-pid_*/
# The parent script restores env after each config.
""".lstrip()


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GemmaSizePreset:
    """Architectural hyperparameters for one Gemma-shaped decoder."""

    label: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int


# Gemma-3 family approximate shapes. 270m exactly matches the SimpleGemmaStack
# defaults used elsewhere in the repo; larger presets follow the publicly
# reported architectures (override any dim with the corresponding CLI flag if
# they drift from your reference).
MODEL_PRESETS: Dict[str, GemmaSizePreset] = {
    "270m": GemmaSizePreset(
        label="Gemma-3 270M",
        num_layers=18,
        hidden_size=1152,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        intermediate_size=6912,
    ),
    "1b": GemmaSizePreset(
        label="Gemma-3 1B",
        num_layers=26,
        hidden_size=1152,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        intermediate_size=6912,
    ),
    "4b": GemmaSizePreset(
        label="Gemma-3 4B",
        num_layers=34,
        hidden_size=2560,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        intermediate_size=10240,
    ),
    "12b": GemmaSizePreset(
        label="Gemma-3 12B",
        num_layers=48,
        hidden_size=3840,
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        intermediate_size=15360,
    ),
    "27b": GemmaSizePreset(
        label="Gemma-3 27B",
        num_layers=62,
        hidden_size=5376,
        num_heads=32,
        num_kv_heads=16,
        head_dim=128,
        intermediate_size=21504,
    ),
}


# ---------------------------------------------------------------------------
# Sweep helpers (manifest, debug staging, etc.)
# ---------------------------------------------------------------------------
def _ensure_debug_runs_readme(result_dir: str) -> None:
    parent = os.path.join(result_dir, _TORCH_DEBUG_RUNS_ROOT)
    os.makedirs(parent, exist_ok=True)
    readme = os.path.join(parent, "README.txt")
    if not os.path.isfile(readme):
        with open(readme, "w") as fh:
            fh.write(_TORCH_DEBUG_RUNS_README)


def _per_config_relpath(model_key: str, batch: int, seq: int, mode: str) -> str:
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
        "hidden_size": p.hidden_size,
        "num_heads": p.num_heads,
        "num_kv_heads": p.num_kv_heads,
        "head_dim": p.head_dim,
        "intermediate_size": p.intermediate_size,
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


# ---------------------------------------------------------------------------
# NUMA / compile-debug plumbing
# ---------------------------------------------------------------------------
def _enable_numa() -> None:
    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        raise RuntimeError("NUMA runtime hooks unavailable in this build.")
    cpu_mod._set_numa_enabled(True)
    print(
        f"NUMA enabled={cpu_mod._is_numa_enabled()} "
        f"nodes={cpu_mod._num_numa_nodes()}",
        file=sys.stderr,
        flush=True,
    )


def _dynamo_clear_debug_dir_cache() -> None:
    try:
        from torch._dynamo.utils import _get_debug_dir

        c = getattr(_get_debug_dir, "cache_clear", None)
        if c is not None:
            c()
    except Exception:
        pass


def _enter_compile_debug(staging_dir: str) -> Dict[str, Any]:
    from torch._dynamo import config as dynamo_config

    artifact_staging = os.path.abspath(staging_dir)
    os.makedirs(artifact_staging, exist_ok=True)

    dynamo_logger = logging.getLogger("torch._dynamo")
    state: Dict[str, Any] = {
        "prev_tcd": os.environ.get("TORCH_COMPILE_DEBUG"),
        "prev_tcdir": os.environ.get("TORCH_COMPILE_DEBUG_DIR"),
        "prev_dynamo_debug_root": dynamo_config.debug_dir_root,
        "prev_dynamo_log_level": dynamo_logger.level,
        "prev_trace_enabled": inductor_config.trace.enabled,
        "artifact_staging": artifact_staging,
    }
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = artifact_staging
    dynamo_config.debug_dir_root = dynamo_config.default_debug_dir_root()
    dynamo_logger.setLevel(logging.DEBUG)
    inductor_config.trace.enabled = True
    _dynamo_clear_debug_dir_cache()
    return state


def _exit_compile_debug(state: Dict[str, Any]) -> None:
    from torch._dynamo import config as dynamo_config

    p = state["prev_tcd"]
    if p is None:
        os.environ.pop("TORCH_COMPILE_DEBUG", None)
    else:
        os.environ["TORCH_COMPILE_DEBUG"] = p
    p2 = state["prev_tcdir"]
    if p2 is None:
        os.environ.pop("TORCH_COMPILE_DEBUG_DIR", None)
    else:
        os.environ["TORCH_COMPILE_DEBUG_DIR"] = p2
    dynamo_config.debug_dir_root = state["prev_dynamo_debug_root"]
    inductor_config.trace.enabled = state["prev_trace_enabled"]
    logging.getLogger("torch._dynamo").setLevel(state["prev_dynamo_log_level"])
    _dynamo_clear_debug_dir_cache()


def _flags(numa: bool, world_size: int) -> dict:
    return {
        "_numa_attention_tp": numa,
        "_numa_mlp_tp": numa,
        "_numa_attention_tp_world_size": world_size if numa else 0,
        "_numa_attention_tp_compute_shard": 0 if numa else 0,
        "_numa_mlp_tp_world_size": world_size if numa else 0,
        "_numa_mlp_tp_compute_shard": 0 if numa else 0,
        "fx_graph_cache": False,
    }


@dataclass
class TimingSummary:
    mean_ms: float
    median_ms: float
    min_ms: float
    iters: int


def _scan_output_code(cache_dir: str, since: float = 0.0) -> dict:
    candidate_dirs = [cache_dir]
    default = os.path.join(
        tempfile.gettempdir(), f"torchinductor_{os.environ.get('USER', '')}"
    )
    if default not in candidate_dirs:
        candidate_dirs.append(default)
    files: list[str] = []
    for d in candidate_dirs:
        if d and os.path.isdir(d):
            files.extend(
                glob.glob(os.path.join(d, "**", "output_code.py"), recursive=True)
            )
    if since > 0:
        files = [f for f in files if os.path.getmtime(f) >= since]
    numa_calls = 0
    triton_kernels = 0
    extern_mm = 0
    reassemble = 0
    largest_file = None
    largest_size = 0
    for f in files:
        try:
            with open(f) as fh:
                txt = fh.read()
        except OSError:
            continue
        if len(txt) > largest_size:
            largest_size = len(txt)
            largest_file = f
        numa_calls += txt.count("numa_args=")
        triton_kernels += txt.count("def triton_")
        extern_mm += txt.count("extern_kernels.mm(") + txt.count(
            "extern_kernels.addmm("
        )
        reassemble += txt.count("# numa reassemble") + txt.count("_numa_reassemble")
    return {
        "files": len(files),
        "numa_calls": numa_calls,
        "triton_kernels": triton_kernels,
        "extern_mm": extern_mm,
        "reassemble": reassemble,
        "largest_file": largest_file,
    }


def _run_gemma_path(
    *,
    arch: Dict[str, int],
    B: int,
    S: int,
    numa: bool,
    world_size: int,
    warmup: int,
    iters: int,
    compile_debug: bool,
    compile_debug_staging: Optional[str],
    record_iteration_latencies: bool,
) -> Tuple[TimingSummary, dict]:
    """Build a fresh Gemma model + inputs, compile, warm up, and time."""
    info: dict = {}
    cache_dir = tempfile.mkdtemp(prefix="gemma_sweep_inductor_cache_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    compile_started = time.time()

    if compile_debug:
        if not compile_debug_staging:
            raise RuntimeError(
                "compile_debug_staging is required when compile_debug is True"
            )
        cdbg_state = _enter_compile_debug(compile_debug_staging)
    else:
        cdbg_state = None

    try:
        torch.manual_seed(0)
        device = torch.device("cpu")
        model = (
            SimpleGemmaStack(
                num_layers=arch["num_layers"],
                hidden_size=arch["hidden_size"],
                num_heads=arch["num_heads"],
                num_kv_heads=arch["num_kv_heads"],
                head_dim=arch["head_dim"],
                intermediate_size=arch["intermediate_size"],
            )
            .to(device)
            .eval()
        )
        for p in model.parameters():
            nn.init.normal_(p, std=0.02)

        x, freqs_cis, kv_write_indices, kv_caches, mask = _build_prefill_inputs(
            B, S, arch["hidden_size"], arch["num_kv_heads"], arch["head_dim"],
            arch["num_layers"], device,
        )

        torch._dynamo.reset()
        with inductor_config.patch(_flags(numa, world_size)), torch.no_grad():
            compiled = torch.compile(model)

            def run() -> None:
                compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)

            for _ in range(warmup):
                run()

            info.update(_scan_output_code(cache_dir, since=compile_started))
            info["cache_dir"] = cache_dir

            ordered: List[float] = []
            for _ in range(iters):
                t0 = time.perf_counter()
                run()
                ordered.append((time.perf_counter() - t0) * 1000.0)

        s = sorted(ordered)
        if record_iteration_latencies:
            info["latencies_ms"] = list(ordered)
        if not ordered:
            timing = TimingSummary(mean_ms=0.0, median_ms=0.0, min_ms=0.0, iters=0)
        else:
            timing = TimingSummary(
                mean_ms=statistics.fmean(ordered),
                median_ms=s[len(s) // 2],
                min_ms=s[0],
                iters=iters,
            )
    finally:
        if cdbg_state is not None:
            _exit_compile_debug(cdbg_state)

    return timing, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    default_models = ["270m", "1b"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Root for torch_compile_debug_runs/ and sweep_manifest.jsonl. "
        "Default: <cwd>/gemma_compile_debug_sweep",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model presets to sweep (default: 270m 1b; larger sizes available).",
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
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup forward passes per path (default: 5).",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations per path (default: 50).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--skip-numa-tp",
        action="store_true",
        help="Only run standard path.",
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
            "Do not set TORCH_COMPILE_DEBUG=1 or write per-config torch_compile_debug/ "
            "trees (faster; use when only timing matters)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.skip_numa_tp and args.skip_standard:
        print("Both --skip-numa-tp and --skip-standard; nothing to do.", file=sys.stderr)
        return 2

    result_dir = os.path.abspath(
        args.result_dir or os.path.join(os.getcwd(), "gemma_compile_debug_sweep")
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

    manifest_path = os.path.join(result_dir, _MANIFEST_NAME)
    with open(manifest_path, "w") as fh:
        fh.write(
            json.dumps(
                {
                    "_record": "meta",
                    "format": "gemma_compile_debug_sweep_v1",
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
        problems: List[str] = []
        if H_qkv % args.world_size != 0:
            problems.append(
                f"H_qkv={H_qkv} not divisible by world_size={args.world_size}"
            )
        if arch["num_heads"] % args.world_size != 0:
            problems.append(
                f"num_heads={arch['num_heads']} not divisible by "
                f"world_size={args.world_size}"
            )
        if arch["intermediate_size"] % args.world_size != 0:
            problems.append(
                f"intermediate_size={arch['intermediate_size']} not divisible by "
                f"world_size={args.world_size}"
            )
        if arch["num_heads"] % arch["num_kv_heads"] != 0:
            problems.append(
                f"num_heads={arch['num_heads']} not divisible by "
                f"num_kv_heads={arch['num_kv_heads']}"
            )
        if problems:
            msg = "; ".join(problems)
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
            for mode in modes:
                numa = mode == "numa_tp"
                rel = _per_config_relpath(model_key, B, S, mode)
                staging = os.path.join(result_dir, rel)
                label = f"{model_key} {mode} B={B} S={S}"
                print(f"[run] {label} -> {staging}", file=sys.stderr, flush=True)
                t_cfg = time.perf_counter()
                try:
                    use_cdbg = not args.no_compile_debug
                    timing, info = _run_gemma_path(
                        arch=arch,
                        B=B,
                        S=S,
                        numa=numa,
                        world_size=args.world_size,
                        warmup=args.warmup,
                        iters=args.iters,
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
