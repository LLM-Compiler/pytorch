#!/usr/bin/env python3
# mypy: ignore-errors
"""
Intel Advisor wrapper — NUMA tensor-parallel compilation path.

Designed to be run under `advixe-cl --start-paused`.  Advisor begins paused,
so torch.compile, Triton kernel JIT, gcc invocations, and all warmup iterations
execute *outside* the recording region.  The measurement loop calls
__itt_resume() / __itt_pause() so the resulting roofline reflects only
steady-state Triton-CPU kernel execution with NUMA tensor-parallel sharding.

Usage (called by run_advisor_roofline.sh):
    advixe-cl --collect=survey --start-paused --project-dir=./advi_numa \
              -- python advisor_bench_numa.py
    advixe-cl --collect=tripcounts --enable-cache-simulation --flop \
              --start-paused --project-dir=./advi_numa \
              -- python advisor_bench_numa.py
"""
from __future__ import annotations

import ctypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn as nn
from torch._inductor import config as inductor_config

from simple_llm_decoder_block_paper_bench import SimpleLmStack

inductor_config.compile_threads = 1
inductor_config.max_autotune_gemm_backends = "TRITON"

# ── Intel Advisor ITT pause/resume ────────────────────────────────────────────
# libittnotify_collector.so exports __itt_pause / __itt_resume (verified via nm).
# When the process runs under `advixe-cl --start-paused`, the collector starts
# paused; we resume it only around the measurement loop.  Outside Advisor, the
# dlopen fails and these become no-ops.
_ADVISOR_HOME = os.environ.get("ADVISOR_HOME", "/opt/intel/oneapi/advisor/2023.1.0")
_ITT_LIB_PATH = os.path.join(_ADVISOR_HOME, "lib64", "runtime", "libittnotify_collector.so")

def _load_itt():
    if not os.path.exists(_ITT_LIB_PATH):
        return None, None
    try:
        lib = ctypes.CDLL(_ITT_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
        return getattr(lib, "__itt_resume"), getattr(lib, "__itt_pause")
    except (OSError, AttributeError):
        return None, None

_itt_resume_fn, _itt_pause_fn = _load_itt()

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
WARMUP      = int(os.environ.get("ADVISOR_BENCH_WARMUP",        1))
ITERS       = int(os.environ.get("ADVISOR_BENCH_ITERS",        1))
TP_SHARDS   = int(os.environ.get("ADVISOR_BENCH_TP_SHARDS",     4))  # match NUMA nodes
# ─────────────────────────────────────────────────────────────────────────────

cpu_mod = torch._C._cpu
cpu_mod._set_numa_enabled(True)
if not cpu_mod._is_numa_enabled():
    raise RuntimeError("Failed to enable NUMA via _set_numa_enabled(True).")
print(f"[numa] NUMA enabled, world_size={TP_SHARDS}", flush=True)

torch.manual_seed(0)
model = SimpleLmStack(1, NUM_HEADS, HEAD_DIM, H_IN, MLP_HIDDEN)
model.set_batch_seq(B, S)
for p in model.parameters():
    nn.init.normal_(p, std=0.02)

x = torch.randn(B * S, H_IN)

with inductor_config.patch({
    "_numa_attention_tp":             True,
    "_numa_attention_tp_world_size":  TP_SHARDS,
    "_numa_attention_tp_compute_shard": 0,
    "_numa_mlp_tp":                   True,
    "_numa_mlp_tp_world_size":        TP_SHARDS,
    "_numa_mlp_tp_compute_shard":     0,
    "fx_graph_cache":                 False,
}):
    torch._dynamo.reset()
    compiled = torch.compile(model)

    print(f"[numa] warmup ({WARMUP} iters) — Advisor PAUSED, "
          f"compile + Triton JIT happens here …", flush=True)
    for _ in range(WARMUP):
        _ = compiled(x)

    print(f"[numa] measurement loop ({ITERS} iters) — resuming Advisor …",
          flush=True)
    itt_resume()
    try:
        for _ in range(ITERS):
            _ = compiled(x)
    finally:
        itt_pause()

    print("[numa] done.", flush=True)
