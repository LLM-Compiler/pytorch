#!/usr/bin/env python3
# mypy: ignore-errors
r"""
Research-oriented benchmark: **decoder block only** (``SimpleLmStack`` / ``TransformerLayer``)
comparing the **standard** Inductor path vs the **NUMA tensor-parallel** compiled path.

This mirrors the methodology of ``gemma_bench_orchestrator.py`` in spirit (warmup, many timed
iters, per-iter latencies, CSV/JSON manifest) but targets the minimal stacks in
``simple_llm_torch_compile_demo.py`` and ``simple_llm_numa_tp_torch_compile_demo.py`` — not
full Gemma.

**Iteration space (what to sweep for a paper)**

- **(batch, seq)** — prefill is large S; *decode-style* is ``seq=1`` with batch as parallel
  streams (same as Gemma’s prefill vs decode labeling).
- **num_layers** — single-block (1) for kernel-focused claims; >1 for depth scaling.
- **mode** — ``standard`` (``_numa_*`` off) vs ``numa_tp`` (NUMA enabled + both passes on).
- **num_tp_shards** (NUMA only) — ablation: often match ``min(num_numa_nodes, num_heads)`` or sweep 2/4.

**Timing methodology**

- **Warmup** (separate, not timed): enough iterations for ``torch.compile`` + autotune to settle
  (default 10; increase for noisy first runs).
- **Measure**: default 50 timed forwards; use ``time.perf_counter()`` **per iteration** (not
  only total/iters) so you can report mean, stdev, p50, p95, p99 (standard in systems papers).
- **Fairness**: fixed ``torch.manual_seed(0)`` and the same init per config.
- **Order**: for each (B,S,L), run both modes in a fixed order.
- **Environment**: log ``OMP_NUM_THREADS`` and host; optional ``numactl`` policy like Gemma
  (this script does not force it — set externally for reproducible affinity).

**Output (crash-tolerant):** Each successful configuration is **appended** immediately to
``--out-csv`` and to ``--out-json`` (line-delimited JSON, **JSONL**): a meta line, then
one JSON object per run (with ``latencies_ms``). After each line the file is **flushed and
``fsync``'d** so a server crash or ``kill -9`` still leaves all completed runs on disk.

- **``TORCH_COMPILE_DEBUG`` (one folder per config):** Before each ``run_one_mode`` the
  script sets ``TORCH_COMPILE_DEBUG=1`` and a **unique** ``TORCH_COMPILE_DEBUG_DIR`` under
  ``<result_dir>/torch_compile_debug_runs/<L_B_S_mode>/``; Dynamo/Inductor then write
  ``.../torch_compile_debug/run_<timestamp>-pid_*/`` inside that staging dir, and the
  environment is **restored** after the config. Use ``--no-compile-debug`` to skip
  (faster, large sweeps). See ``torch_compile_debug_runs/README.txt``.

**Example**

  python scripts/inductor/simple_llm_decoder_block_paper_bench.py \
      --batch-sizes 1 4 --seq-lens 1 128 512 --num-layers 1 \
      --warmup 10 --iters 50 --out-csv results.csv --out-json results.jsonl

"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


class NumaRuntimeError(RuntimeError):
    """NUMA hooks missing or `_set_numa_enabled(True)` did not stick — abort whole run."""

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor_config
from torch.nn.attention import sdpa_kernel, SDPBackend

inductor_config.max_autotune_gemm_backends = "TRITON"
#inductor_config.max_autotune = True

BenchMode = Literal["standard", "numa_tp"]

# PyTorch: TORCH_COMPILE_DEBUG_DIR is joined with "torch_compile_debug" for the
# run_* folders (see torch._dynamo.config.default_debug_dir_root).
_TORCH_DEBUG_RUNS_ROOT = "torch_compile_debug_runs"
_TORCH_DEBUG_RUNS_README = """# Per-configuration Torch compile debug
# Each subfolder name encodes: L<layers>_B<batch>_S<seq>_<mode>
#   mode is `standard` or `numa_tp`
# With TORCH_COMPILE_DEBUG=1, Dynamo writes to:
#   <that_subfolder>/torch_compile_debug/run_YYYY_mm_dd_...-pid_*/
# (see torch._dynamo.utils.get_debug_dir). The parent script restores env after each config.
""".lstrip()


def _dynamo_clear_debug_dir_cache() -> None:
    try:
        from torch._dynamo.utils import _get_debug_dir

        c = getattr(_get_debug_dir, "cache_clear", None)
        if c is not None:
            c()
    except Exception:
        pass


def _result_dir_for_outputs(out_csv: str) -> str:
    return os.path.dirname(os.path.abspath(out_csv))


def _per_config_torch_debug_dirname(
    n_layers: int, batch: int, seq: int, mode: str
) -> str:
    """Unique, filesystem-safe folder name for one (L,B,S,mode) config."""
    return f"L{n_layers}_B{batch}_S{seq}_{mode}"


def _ensure_debug_runs_readme(result_dir: str) -> None:
    parent = os.path.join(result_dir, _TORCH_DEBUG_RUNS_ROOT)
    os.makedirs(parent, exist_ok=True)
    readme = os.path.join(parent, "README.txt")
    if not os.path.isfile(readme):
        with open(readme, "w") as f:
            f.write(_TORCH_DEBUG_RUNS_README)


def _enter_compile_debug_for_config(
    result_dir: str, per_config_rel: str
) -> Dict[str, Any]:
    """Set TORCH_COMPILE_DEBUG=1; staging dir: result_dir / per_config_rel (unique per config)."""
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config_mod

    artifact_staging = os.path.join(result_dir, per_config_rel)
    os.makedirs(artifact_staging, exist_ok=True)

    dynamo_logger = logging.getLogger("torch._dynamo")
    state: Dict[str, Any] = {
        "prev_tcd": os.environ.get("TORCH_COMPILE_DEBUG"),
        "prev_tcdir": os.environ.get("TORCH_COMPILE_DEBUG_DIR"),
        "prev_dynamo_debug_root": dynamo_config.debug_dir_root,
        "prev_dynamo_log_level": dynamo_logger.level,
        # config.trace.enabled is read once at import time from the env var,
        # so setting the env var alone has no effect — must patch directly.
        "prev_trace_enabled": inductor_config_mod.trace.enabled,
        "artifact_staging": artifact_staging,
    }
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = artifact_staging
    dynamo_config.debug_dir_root = dynamo_config.default_debug_dir_root()
    # Enable content in the dynamo log file (default logger level is WARNING).
    dynamo_logger.setLevel(logging.DEBUG)
    # This is the master switch that gates DebugContext writing fx_graph,
    # ir_pre/post_fusion, output_code, etc. — env var is already too late.
    inductor_config_mod.trace.enabled = True
    _dynamo_clear_debug_dir_cache()
    return state


def _exit_compile_debug(state: Dict[str, Any]) -> None:
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config_mod

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
    inductor_config_mod.trace.enabled = state["prev_trace_enabled"]
    logging.getLogger("torch._dynamo").setLevel(state["prev_dynamo_log_level"])
    _dynamo_clear_debug_dir_cache()


class TransformerLayer(nn.Module):
    """Identical to the two ``simple_llm_*_torch_compile_demo.py`` blocks."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.H = num_heads
        self.E = head_dim
        self.H_in = hidden_dim
        self.H_qkv = num_heads * head_dim

        self.Wq = nn.Parameter(torch.empty(hidden_dim, self.H_qkv))
        self.Wk = nn.Parameter(torch.empty(hidden_dim, self.H_qkv))
        self.Wv = nn.Parameter(torch.empty(hidden_dim, self.H_qkv))
        self.bq = nn.Parameter(torch.empty(self.H_qkv))
        self.bk = nn.Parameter(torch.empty(self.H_qkv))
        self.bv = nn.Parameter(torch.empty(self.H_qkv))
        self.Wo = nn.Parameter(torch.empty(self.H_qkv, hidden_dim))
        self.bo = nn.Parameter(torch.empty(hidden_dim))

        self.mlp_fc1_w = nn.Parameter(torch.empty(hidden_dim, mlp_hidden_dim))
        self.mlp_fc1_b = nn.Parameter(torch.empty(mlp_hidden_dim))
        self.mlp_fc2_w = nn.Parameter(torch.empty(mlp_hidden_dim, hidden_dim))
        self.mlp_fc2_b = nn.Parameter(torch.empty(hidden_dim))

        self._batch: int = 0
        self._seq: int = 0

    def set_batch_seq(self, batch: int, seq: int) -> None:
        self._batch = batch
        self._seq = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H, E = self._batch, self._seq, self.H, self.E
        q = torch.addmm(self.bq, x, self.Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(self.bk, x, self.Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(self.bv, x, self.Wv).view(B, S, H, E).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        attn_out = torch.addmm(self.bo, y, self.Wo)
        h = attn_out + x

        u = torch.addmm(self.mlp_fc1_b, h, self.mlp_fc1_w)
        u = F.gelu(u)
        mlp_out = torch.addmm(self.mlp_fc2_b, u, self.mlp_fc2_w)
        return mlp_out + h


class SimpleLmStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerLayer(num_heads, head_dim, hidden_dim, mlp_hidden_dim)
            for _ in range(num_layers)
        )

    def set_batch_seq(self, batch: int, seq: int) -> None:
        for layer in self.layers:
            layer.set_batch_seq(batch, seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _auto_num_tp_shards(num_heads: int, num_numa_nodes: int) -> int:
    cap = max(1, int(num_numa_nodes))
    for d in range(min(cap, num_heads), 0, -1):
        if num_heads % d == 0:
            return d
    return 1


def _percentile_nearest(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def time_forward_per_iter(
    run: Any,
    *,
    warmup: int,
    iters: int,
) -> Tuple[List[float], float]:
    """
    Returns (latencies_ms_per_iter, compile_proxy_s).

    *compile_proxy_s* = wall time of the first *warmup* iteration only (rough compile+
    first-run effect); first iter is often an outlier — report separately in tables.
    """
    compile_proxy = 0.0
    for i in range(warmup):
        t0 = time.perf_counter()
        run()
        dt = time.perf_counter() - t0
        if i == 0:
            compile_proxy = dt

    latencies_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return latencies_ms, compile_proxy


def _summarize_latencies_ms(latencies: Sequence[float]) -> Dict[str, float]:
    s = sorted(latencies)
    n = len(s)
    if n == 0:
        return {
            "mean_ms": float("nan"),
            "stdev_ms": float("nan"),
            "min_ms": float("nan"),
            "max_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
        }
    mean = statistics.fmean(s)
    stdev = statistics.pstdev(s) if n > 1 else 0.0
    return {
        "mean_ms": mean,
        "stdev_ms": stdev,
        "min_ms": s[0],
        "max_ms": s[-1],
        "p50_ms": _percentile_nearest(s, 50),
        "p95_ms": _percentile_nearest(s, 95),
        "p99_ms": _percentile_nearest(s, 99),
    }


@dataclass
class BenchRow:
    hostname: str
    mode: str
    num_layers: int
    batch: int
    seq: int
    h_in: int
    num_heads: int
    head_dim: int
    mlp_hidden: int
    num_tp_shards: int
    num_numa_nodes: int
    warmup: int
    iters: int
    omp_num_threads: str
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    compile_proxy_s: float
    speedup_vs_standard_mean: float
    note: str


_CSV_FIELDNAMES = [f.name for f in fields(BenchRow)]


def _init_output_files(
    out_csv: str, out_json: Optional[str], host: str, omp: str, torch_ver: str
) -> None:
    """Create CSV (header only) and optional JSONL (meta line). Overwrites on each start."""
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())

    if out_json is not None:
        with open(out_json, "w") as f:
            f.write(
                json.dumps(
                    {
                        "_record": "meta",
                        "format": "simple_llm_decoder_bench_v1",
                        "hostname": host,
                        "omp_num_threads": str(omp),
                        "torch": torch_ver,
                    }
                )
                + "\n"
            )
            f.flush()
            os.fsync(f.fileno())


def _append_csv_row(out_csv: str, row: BenchRow) -> None:
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        w.writerow(asdict(row))
        f.flush()
        os.fsync(f.fileno())


def _append_jsonl_run(out_json: str, row: BenchRow, latencies_ms: List[float]) -> None:
    rec = {**asdict(row), "latencies_ms": latencies_ms}
    with open(out_json, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_one_mode(
    mode: BenchMode,
    *,
    num_layers: int,
    B: int,
    S: int,
    h_in: int,
    num_heads: int,
    head_dim: int,
    mlp_hidden: int,
    num_tp_shards: Optional[int],
    warmup: int,
    iters: int,
) -> Tuple[Dict[str, float], List[float], float, int, int]:
    """
    Returns summary dict, raw latencies ms, compile_proxy_s, num_tp_shards used, n_nodes.
    """
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
            raise NumaRuntimeError(
                "NUMA runtime hooks missing (torch._C._cpu._set_numa_enabled)."
            )
        if not hasattr(cpu_mod, "_is_numa_enabled") or not hasattr(
            cpu_mod, "_num_numa_nodes"
        ):
            raise NumaRuntimeError(
                "NUMA query hooks missing (_is_numa_enabled, _num_numa_nodes)."
            )
        old_numa = cpu_mod._is_numa_enabled()
        cpu_mod._set_numa_enabled(True)
        if not cpu_mod._is_numa_enabled():
            cpu_mod._set_numa_enabled(old_numa)
            raise NumaRuntimeError(
                "Failed to enable NUMA via _set_numa_enabled(True)."
            )
        n_nodes = int(cpu_mod._num_numa_nodes())
        if num_tp_shards is None:
            shards = _auto_num_tp_shards(num_heads, n_nodes)
        else:
            shards = num_tp_shards
        if n_nodes < shards:
            raise RuntimeError(
                f"num_tp_shards={shards} but only {n_nodes} NUMA node(s)"
            )
        for cond, msg in (
            (num_heads % shards == 0, f"num_heads % shards ({shards}) != 0"),
            (H_qkv % shards == 0, f"H_qkv % shards ({shards}) != 0"),
            (mlp_hidden % shards == 0, f"mlp_hidden % shards ({shards}) != 0"),
        ):
            if not cond:
                raise ValueError(msg)
    else:
        shards = 1

    try:
        model = SimpleLmStack(
            num_layers, num_heads, head_dim, h_in, mlp_hidden,
        ).to(device)
        model.set_batch_seq(B, S)
        for p in model.parameters():
            nn.init.normal_(p, std=0.02)

        x = torch.randn(B * S, h_in, device=device)
        y = x  # sink

        if mode == "standard":
            ctx = inductor_config.patch(
                {
                    "_numa_attention_tp": False,
                    "_numa_mlp_tp": False,
                    # fx_graph_cache uses save_config_portable() for its key, which
                    # skips all keys starting with "_". The NUMA flags are all "_"-prefixed,
                    # so standard and numa_tp share the same cache key. Disabling the cache
                    # forces a fresh compilation for each mode so NUMA passes actually run.
                    "fx_graph_cache": False,
                }
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

        with ctx:
            # Clear all Dynamo compile caches so each mode gets a fresh
            # compilation. Without this, the standard-mode cached graph is
            # served as a cache hit for the numa_tp run — the inductor passes
            # never fire and the NUMA path silently executes standard code.
            torch._dynamo.reset()
            compiled = torch.compile(model)

            def run() -> None:
                nonlocal y
                y = compiled(x)

            latencies, compile_proxy = time_forward_per_iter(
                run, warmup=warmup, iters=iters
            )

        summary = _summarize_latencies_ms(latencies)
        summary["compile_proxy_s"] = compile_proxy
        return summary, latencies, compile_proxy, shards, n_nodes
    finally:
        if mode == "numa_tp" and cpu_mod is not None and old_numa is not None:
            cpu_mod._set_numa_enabled(old_numa)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper benchmark: standard vs NUMA-TP decoder block (SimpleLmStack)"
    )
    p.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1],
        help="Batch sizes B (default: 1)",
    )
    p.add_argument(
        "--seq-lens", nargs="+", type=int, default=[1, 128, 512],
        help="Sequence lengths; use 1 for decode-style, >1 for prefill (default: 1 128 512)",
    )
    p.add_argument(
        "--num-layers", nargs="+", type=int, default=[1],
        help="Decoder depth(s) to test (default: 1)",
    )
    p.add_argument("--h-in", type=int, default=1152)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--mlp-hidden", type=int, default=6912)
    p.add_argument(
        "--num-tp-shards",
        type=int,
        default=None,
        help="NUMA TP world size; default auto from heads vs NUMA count",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["standard", "numa_tp"],
        choices=["standard", "numa_tp"],
        help="Runs are ordered: standard first, then numa_tp (so speedup is well-defined).",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out-csv", type=str, default="decoder_block_paper_bench.csv")
    p.add_argument(
        "--out-json",
        type=str,
        default="decoder_block_paper_bench.jsonl",
        help="Line-delimited JSON: meta line, then one run object per line (appended, fsynced)",
    )
    p.add_argument(
        "--skip-json",
        action="store_true",
        help="Do not write JSON (smaller, CSV only)",
    )
    p.add_argument(
        "--no-compile-debug",
        action="store_true",
        help="Do not set TORCH_COMPILE_DEBUG=1 per config (faster; no per-config debug dirs)",
    )
    return p.parse_args()


def _mode_run_order(modes: Sequence[str]) -> List[str]:
    """Always run *standard* before *numa_tp* so in-memory speedup is well-defined."""
    order = ("standard", "numa_tp")
    return [m for m in order if m in modes]


def main() -> None:
    args = _parse_args()
    run_modes = _mode_run_order(args.modes)
    if not run_modes:
        print("No modes selected.")
        return

    host = platform.node()
    omp = os.environ.get("OMP_NUM_THREADS", "(unset)")

    _init_output_files(
        args.out_csv,
        None if args.skip_json else args.out_json,
        host,
        omp,
        torch.__version__,
    )

    all_rows: List[BenchRow] = []

    # Baseline means per (L,B,S) for speedup
    standard_mean: Dict[Tuple[int, int, int], float] = {}

    result_dir = _result_dir_for_outputs(args.out_csv)
    if not args.no_compile_debug:
        _ensure_debug_runs_readme(result_dir)
    for n_layers in args.num_layers:
        for B in args.batch_sizes:
            for S in args.seq_lens:
                for mode in run_modes:
                    try:
                        cdbg_state: Optional[Dict[str, Any]] = None
                        if not args.no_compile_debug:
                            rel = os.path.join(
                                _TORCH_DEBUG_RUNS_ROOT,
                                _per_config_torch_debug_dirname(
                                    n_layers, B, S, mode
                                ),
                            )
                            cdbg_state = _enter_compile_debug_for_config(
                                result_dir, rel
                            )
                            print(
                                f"[info] TORCH_COMPILE_DEBUG=1; dir="
                                f"{cdbg_state['artifact_staging']!r}",
                                file=sys.stderr,
                            )
                        try:
                            summary, lats, compile_proxy, shards, n_nodes = (
                                run_one_mode(
                                    mode,  # type: ignore[arg-type]
                                    num_layers=n_layers,
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
                            )
                        finally:
                            if cdbg_state is not None:
                                _exit_compile_debug(cdbg_state)
                    except NumaRuntimeError as e:
                        print(
                            f"[FATAL] {e}",
                            file=sys.stderr,
                        )
                        raise SystemExit(1) from e
                    except Exception as e:
                        print(
                            f"[FAIL] mode={mode} L={n_layers} B={B} S={S}: {e}"
                        )
                        continue

                    key = (n_layers, B, S)
                    if mode == "standard":
                        standard_mean[key] = summary["mean_ms"]
                        speedup = 1.0
                    else:
                        base = standard_mean.get(key, float("nan"))
                        if base and base == base and base > 0:
                            speedup = base / summary["mean_ms"]
                        else:
                            speedup = float("nan")

                    row = BenchRow(
                        hostname=host,
                        mode=mode,
                        num_layers=n_layers,
                        batch=B,
                        seq=S,
                        h_in=args.h_in,
                        num_heads=args.num_heads,
                        head_dim=args.head_dim,
                        mlp_hidden=args.mlp_hidden,
                        num_tp_shards=shards,
                        num_numa_nodes=n_nodes,
                        warmup=args.warmup,
                        iters=args.iters,
                        omp_num_threads=str(omp),
                        mean_ms=summary["mean_ms"],
                        stdev_ms=summary["stdev_ms"],
                        min_ms=summary["min_ms"],
                        max_ms=summary["max_ms"],
                        p50_ms=summary["p50_ms"],
                        p95_ms=summary["p95_ms"],
                        p99_ms=summary["p99_ms"],
                        compile_proxy_s=compile_proxy,
                        speedup_vs_standard_mean=speedup,
                        note="seq=1 => decode-style; seq>1 => prefill",
                    )
                    all_rows.append(row)
                    _append_csv_row(args.out_csv, row)
                    if not args.skip_json:
                        _append_jsonl_run(args.out_json, row, lats)
                    print(
                        f"OK {mode:10} L={n_layers} B={B} S={S} "
                        f"mean={summary['mean_ms']:.3f} ms"
                    )
                    if mode == "numa_tp" and speedup == speedup:
                        print(f"     speedup vs standard (mean): {speedup:.3f}x")

    n_ok = len(all_rows)
    if n_ok == 0:
        print(
            "No successful benchmark rows; only CSV header and JSONL meta (if not "
            "--skip-json) were written."
        )
    else:
        print(
            f"Finished. Appended {n_ok} data row(s) to {args.out_csv} "
            + (
                f"and {n_ok} run line(s) to {args.out_json}."
                if not args.skip_json
                else "(JSONL skipped; --skip-json)."
            )
        )
        print("Each row was written with flush+fsync as it completed (crash-tolerant).")


if __name__ == "__main__":
    main()
