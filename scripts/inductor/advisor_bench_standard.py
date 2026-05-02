#!/usr/bin/env python3
# mypy: ignore-errors
"""
Intel Advisor wrapper — standard (non-NUMA) compilation path.

Designed to be run under `advixe-cl --start-paused`.  Advisor begins paused,
so torch.compile, Triton kernel JIT, gcc invocations, and all warmup iterations
execute *outside* the recording region.  The measurement loop calls
__itt_resume() / __itt_pause() so the resulting roofline reflects only
steady-state Triton-CPU kernel execution.

Usage (called by run_advisor_roofline.sh):
    advixe-cl --collect=survey --start-paused --project-dir=./advi_standard \
              -- python advisor_bench_standard.py
    advixe-cl --collect=tripcounts --enable-cache-simulation --flop \
              --start-paused --project-dir=./advi_standard \
              -- python advisor_bench_standard.py
"""
from __future__ import annotations

import ctypes
import os
import sys

# Allow importing the model class from the bench script in the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

from simple_llm_decoder_block_paper_bench import SimpleLmStack

inductor_config.compile_threads = 1
inductor_config.max_autotune_gemm_backends = "TRITON"

# ── Intel Advisor ITT pause/resume ────────────────────────────────────────────
# Survey and tripcounts use DIFFERENT collectors: survey loads
# libittnotify_collector.so directly, tripcounts is driven by Pin which loads
# its own ittnotify lib.  If we load a second copy via an explicit path our
# __itt_resume call updates *that* copy's state and never reaches Pin's
# tripcounts collector → tripcounts stays paused for the whole measurement →
# GFLOPS=0.  Fix: bind to whichever libittnotify*.so the active wrapper has
# already mapped into this process.  Discover it via /proc/self/maps.
# Outside Advisor, symbol lookup fails and the calls become no-ops.
_ADVISOR_HOME = os.environ.get("ADVISOR_HOME", "/opt/intel/oneapi/advisor/2023.1.0")

def _itt_lib_candidates():
    paths = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libittnotify" in line and ".so" in line:
                    parts = line.split()
                    if len(parts) >= 6 and parts[-1] not in paths:
                        paths.append(parts[-1])
    except OSError:
        pass
    paths.append(os.path.join(_ADVISOR_HOME, "lib64", "runtime", "libittnotify.so"))
    paths.append(os.path.join(_ADVISOR_HOME, "lib64", "runtime", "libittnotify_collector.so"))
    return paths

def _load_itt():
    for path in _itt_lib_candidates():
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return getattr(lib, "__itt_resume"), getattr(lib, "__itt_pause"), path
        except (OSError, AttributeError):
            continue
    return None, None, None

_itt_resume_fn, _itt_pause_fn, _itt_lib_used = _load_itt()
print(f"[itt] bound to: {_itt_lib_used}", flush=True)

def itt_resume():
    if _itt_resume_fn is not None:
        _itt_resume_fn()

def itt_pause():
    if _itt_pause_fn is not None:
        _itt_pause_fn()

# ── Config ────────────────────────────────────────────────────────────────────
B, S        = int(os.environ.get("ADVISOR_BENCH_B", 1)), \
              int(os.environ.get("ADVISOR_BENCH_S", 512))
H_IN        = int(os.environ.get("ADVISOR_BENCH_H_IN",       1152))
NUM_HEADS   = int(os.environ.get("ADVISOR_BENCH_NUM_HEADS",      4))
HEAD_DIM    = int(os.environ.get("ADVISOR_BENCH_HEAD_DIM",     256))
MLP_HIDDEN  = int(os.environ.get("ADVISOR_BENCH_MLP_HIDDEN", 6912))
# Defaults match run_advisor_roofline.sh (overridable via env).
WARMUP      = int(os.environ.get("ADVISOR_BENCH_WARMUP",        2))
ITERS       = int(os.environ.get("ADVISOR_BENCH_ITERS",        20))
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(0)
model = SimpleLmStack(1, NUM_HEADS, HEAD_DIM, H_IN, MLP_HIDDEN)
model.set_batch_seq(B, S)
for p in model.parameters():
    nn.init.normal_(p, std=0.02)

x = torch.randn(B, S, H_IN)

with inductor_config.patch({
    "_numa_attention_tp": False,
    "_numa_mlp_tp":       False,
    "fx_graph_cache":     False,
}):
    torch._dynamo.reset()
    compiled = torch.compile(model)

    print(f"[standard] warmup ({WARMUP} iters) — Advisor PAUSED, "
          f"compile + Triton JIT happens here …", flush=True)
    for _ in range(WARMUP):
        _ = compiled(x)

    print(f"[standard] measurement loop ({ITERS} iters) — resuming Advisor …",
          flush=True)
    itt_resume()
    try:
        for _ in range(ITERS):
            _ = compiled(x)
    finally:
        itt_pause()

    print("[standard] done.", flush=True)
