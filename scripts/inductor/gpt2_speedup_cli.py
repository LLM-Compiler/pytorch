#!/usr/bin/env python3
# mypy: ignore-errors
r"""
End-to-end GPT-2 117M (Pre-LN) prefill speedup CLI.

Defines a minimal GPT-2 117M model with token + positional embeddings,
12 Pre-LN transformer blocks (LN -> attn -> +residual -> LN -> MLP ->
+residual), a final LayerNorm, and a tied LM head. Runs prefill (no KV
cache) and reports:

    speedup = mean_latency_ms_standard / mean_latency_ms_numa_tp

Optionally also runs a correctness check (max-abs / max-rel diff vs the
standard-compiled reference).

NUMA passes enabled in the NUMA-TP path:
  * ``_numa_layer_norm_tp``
  * ``_numa_attention_tp``
  * ``_numa_mlp_tp``
  * ``_numa_residual_tp``

The token + positional embedding lookups and the LM head matmul are NOT
TP'd by this iteration — they run as ordinary kernels. Reassembly at the
final block boundary is therefore expected (output of the last residual
add feeds the final LN, which is TP'd; the final LN's reassembled output
then feeds the LM head, which is non-TP).

Example::

    python scripts/inductor/gpt2_speedup_cli.py \
        --batch-size 1 --seq-length 1024 --warmup 5 --iters 30 \
        --world-size 2

    # With correctness check first:
    python scripts/inductor/gpt2_speedup_cli.py \
        --batch-size 1 --seq-length 256 --warmup 2 --iters 5 \
        --world-size 2 --check-correctness
"""
from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch._inductor import config as inductor_config

inductor_config.max_autotune_gemm_backends = "TRITON"
#inductor_config.max_autotune = True

# GPT-2 117M default hyperparameters.
DEF_VOCAB = 50257
DEF_CONTEXT = 1024
DEF_D_MODEL = 768
DEF_NUM_LAYERS = 12
DEF_NUM_HEADS = 12
DEF_HEAD_DIM = 64  # D_MODEL // NUM_HEADS
DEF_D_FFN = 3072  # 4 * D_MODEL


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GPT2TransformerBlock(torch.nn.Module):
    """Pre-LN GPT-2 block: ``x + attn(LN1(x))`` then ``x + mlp(LN2(x))``."""

    def __init__(self, h_in: int, num_heads: int, head_dim: int, d_ffn: int) -> None:
        super().__init__()
        self.H = num_heads
        self.E = head_dim
        self.H_in = h_in
        h_qkv = num_heads * head_dim

        self.ln1 = torch.nn.LayerNorm(h_in)
        self.Wq = torch.nn.Parameter(torch.empty(h_in, h_qkv))
        self.Wk = torch.nn.Parameter(torch.empty(h_in, h_qkv))
        self.Wv = torch.nn.Parameter(torch.empty(h_in, h_qkv))
        self.bq = torch.nn.Parameter(torch.empty(h_qkv))
        self.bk = torch.nn.Parameter(torch.empty(h_qkv))
        self.bv = torch.nn.Parameter(torch.empty(h_qkv))
        self.Wo = torch.nn.Parameter(torch.empty(h_qkv, h_in))
        self.bo = torch.nn.Parameter(torch.empty(h_in))

        self.ln2 = torch.nn.LayerNorm(h_in)
        self.mlp_fc1_w = torch.nn.Parameter(torch.empty(h_in, d_ffn))
        self.mlp_fc1_b = torch.nn.Parameter(torch.empty(d_ffn))
        self.mlp_fc2_w = torch.nn.Parameter(torch.empty(d_ffn, h_in))
        self.mlp_fc2_b = torch.nn.Parameter(torch.empty(h_in))

    def _attn(self, xn: torch.Tensor, B: int, S: int) -> torch.Tensor:
        H, E = self.H, self.E
        x_flat = xn.reshape(B * S, self.H_in)
        q = torch.addmm(self.bq, x_flat, self.Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(self.bk, x_flat, self.Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(self.bv, x_flat, self.Wv).view(B, S, H, E).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        return torch.addmm(self.bo, y, self.Wo).view(B, S, self.H_in)

    def _mlp(self, xn: torch.Tensor, B: int, S: int) -> torch.Tensor:
        x_flat = xn.reshape(B * S, self.H_in)
        h = torch.addmm(self.mlp_fc1_b, x_flat, self.mlp_fc1_w)
        h = F.gelu(h)
        return torch.addmm(self.mlp_fc2_b, h, self.mlp_fc2_w).view(B, S, self.H_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        x = x + self._attn(self.ln1(x), B, S)
        x = x + self._mlp(self.ln2(x), B, S)
        return x


class GPT2_117M(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = DEF_VOCAB,
        context_size: int = DEF_CONTEXT,
        d_model: int = DEF_D_MODEL,
        num_layers: int = DEF_NUM_LAYERS,
        num_heads: int = DEF_NUM_HEADS,
        head_dim: int = DEF_HEAD_DIM,
        d_ffn: int = DEF_D_FFN,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model

        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(context_size, d_model)
        self.blocks = torch.nn.ModuleList(
            GPT2TransformerBlock(d_model, num_heads, head_dim, d_ffn)
            for _ in range(num_layers)
        )
        self.final_ln = torch.nn.LayerNorm(d_model)
        self._init_weights(num_residual_layers=2 * num_layers)

    def _init_weights(self, num_residual_layers: int) -> None:
        std = 0.02
        residual_std = std / math.sqrt(num_residual_layers)
        with torch.no_grad():
            torch.nn.init.normal_(self.token_embedding.weight, std=std)
            torch.nn.init.normal_(self.position_embedding.weight, std=std)
            for blk in self.blocks:
                # QKV projections + MLP fc1: standard std.
                for p in (blk.Wq, blk.Wk, blk.Wv, blk.mlp_fc1_w):
                    torch.nn.init.normal_(p, std=std)
                # Output projections (residual layers) get scaled std.
                for p in (blk.Wo, blk.mlp_fc2_w):
                    torch.nn.init.normal_(p, std=residual_std)
                for p in (
                    blk.bq, blk.bk, blk.bv, blk.bo,
                    blk.mlp_fc1_b, blk.mlp_fc2_b,
                ):
                    torch.nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, S]
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        return x


# ---------------------------------------------------------------------------
# Compile + run helpers
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


def _flags(
    numa: bool, world_size: int, *, lm_head: bool = False, max_autotune: bool = False
) -> dict:
    base = {
        "_numa_layer_norm_tp": numa,
        "_numa_attention_tp": numa,
        "_numa_mlp_tp": numa,
        "_numa_residual_tp": numa,
        "_numa_lm_head_tp": numa and lm_head,
        "_numa_layer_norm_tp_world_size": world_size if numa else 0,
        "_numa_attention_tp_world_size": world_size if numa else 0,
        "_numa_mlp_tp_world_size": world_size if numa else 0,
        "_numa_lm_head_tp_world_size": world_size if (numa and lm_head) else 0,
    }
    # if max_autotune:
    #     base.update({
    #         "max_autotune": True,
    #         "max_autotune_gemm": True,
    #         "max_autotune_pointwise": False,
    #     })
    return base


def _compile(model: torch.nn.Module, numa: bool, world_size: int):
    return torch.compile(model, mode="default", fullgraph=False)


@dataclass
class TimingSummary:
    mean_ms: float
    median_ms: float
    min_ms: float
    iters: int


def _time_forward(
    compiled,
    input_ids: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> TimingSummary:
    with torch.no_grad():
        for _ in range(warmup):
            compiled(input_ids)
        latencies = []
        for _ in range(iters):
            t0 = time.perf_counter()
            compiled(input_ids)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies.sort()
    return TimingSummary(
        mean_ms=sum(latencies) / len(latencies),
        median_ms=latencies[len(latencies) // 2],
        min_ms=latencies[0],
        iters=iters,
    )


def _scan_output_code(cache_dir: str, since: float = 0.0) -> dict:
    """Grep generated Inductor output_code.py for NUMA tagging signals.

    Inductor caches its cache-dir resolution at import time, so the env-var
    override we set in ``_run_path`` is often ignored and artifacts land in
    the default ``/tmp/torchinductor_<user>/`` tree. Scan both, and (if
    ``since`` is given) restrict to files modified after that timestamp so
    we only count artifacts produced by the current compile.
    """
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
    # The Inductor wrapper file is large but not always called output_code.py
    # in newer caches — also pick up the largest .py per dir.
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
        extern_mm += txt.count("extern_kernels.mm(") + txt.count("extern_kernels.addmm(")
        reassemble += txt.count("# numa reassemble") + txt.count("_numa_reassemble")
    return {
        "files": len(files),
        "numa_calls": numa_calls,
        "triton_kernels": triton_kernels,
        "extern_mm": extern_mm,
        "reassemble": reassemble,
        "largest_file": largest_file,
    }


def _profile_forward(compiled, input_ids: torch.Tensor, *, iters: int):
    from torch.profiler import profile, ProfilerActivity, record_function

    with torch.no_grad():
        # warm
        for _ in range(2):
            compiled(input_ids)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            for _ in range(iters):
                with record_function("gpt2_forward"):
                    compiled(input_ids)
    return prof


def _print_profile(label: str, prof, *, top_n: int = 25) -> None:
    print(f"\n===== {label}: top {top_n} ops by self CPU time =====", file=sys.stderr)
    table = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=top_n
    )
    print(table, file=sys.stderr, flush=True)


def _run_path(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    numa: bool,
    world_size: int,
    warmup: int,
    iters: int,
    lm_head: bool = False,
    profile: bool = False,
    max_autotune: bool = False,
    compile_debug: bool = False,
    compile_debug_staging: str | None = None,
) -> tuple[TimingSummary, torch.Tensor, dict]:
    info: dict = {}
    cache_dir = tempfile.mkdtemp(prefix="gpt2_speedup_inductor_cache_")
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
        # Reset dynamo+inductor so the patched flags take effect on this compile
        # rather than a previously cached graph.
        torch._dynamo.reset()
        with inductor_config.patch(
            _flags(numa, world_size, lm_head=lm_head, max_autotune=max_autotune)
        ):
            compiled = _compile(model, numa=numa, world_size=world_size)
            # Run once to capture output for the correctness check (also primes
            # the cache before timing). Counts toward warmup.
            with torch.no_grad():
                ref_out = compiled(input_ids).detach().clone()
            info.update(_scan_output_code(cache_dir, since=compile_started))
            info["cache_dir"] = cache_dir
            remaining_warmup = max(0, warmup - 1)
            timing = _time_forward(
                compiled, input_ids, warmup=remaining_warmup, iters=iters
            )
            if profile:
                info["prof"] = _profile_forward(
                    compiled, input_ids, iters=min(iters, 10)
                )
    finally:
        if cdbg_state is not None:
            _exit_compile_debug(cdbg_state)
    return timing, ref_out, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-length", type=int, default=DEF_CONTEXT)
    p.add_argument("--num-layers", type=int, default=DEF_NUM_LAYERS)
    p.add_argument("--num-heads", type=int, default=DEF_NUM_HEADS)
    p.add_argument("--head-dim", type=int, default=DEF_HEAD_DIM)
    p.add_argument("--d-model", type=int, default=DEF_D_MODEL)
    p.add_argument("--d-ffn", type=int, default=DEF_D_FFN)
    p.add_argument("--vocab-size", type=int, default=DEF_VOCAB)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--check-correctness",
        action="store_true",
        help="Compare standard vs NUMA-TP output before timing; exit non-zero on FAIL.",
    )
    p.add_argument(
        "--tp-lm-head",
        action="store_true",
        help=(
            "Also TP the LM head matmul (replicated weight, M-split output). "
            "Increases per-socket memory by V*D*sizeof(dtype) but typically "
            "needed to break past the LM-head latency floor in GPT-style models."
        ),
    )
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument(
        "--max-autotune",
        action="store_true",
        help=(
            "Enable Inductor max_autotune for matmuls. Without this the "
            "addmm/bmm calls fall through to extern BLAS (OpenBLAS) which "
            "does not honor numa_args, capping NUMA TP speedup to a small "
            "fraction of the per-block ceiling."
        ),
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run torch.profiler for both standard and NUMA-TP paths and print "
            "top CPU ops by self time. Also scans the generated Inductor "
            "output_code.py to verify NUMA-TP tagging actually fired."
        ),
    )
    p.add_argument(
        "--compile-debug",
        action="store_true",
        help=(
            "Set TORCH_COMPILE_DEBUG=1 for both standard and NUMA-TP compiles, "
            "with separate staging dirs under --compile-debug-root (artifacts "
            "under <staging>/torch_compile_debug/run_*)."
        ),
    )
    p.add_argument(
        "--compile-debug-root",
        type=str,
        default=None,
        help=(
            "Base directory for compile debug staging; subfolders standard/ and "
            "numa_tp/ are created. Default: <cwd>/gpt2_speedup_torch_compile_debug"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.seq_length > DEF_CONTEXT:
        print(
            f"warning: seq_length={args.seq_length} > context={DEF_CONTEXT}; "
            "increase --vocab-size if you change context_size",
            file=sys.stderr,
        )
    H_qkv = args.num_heads * args.head_dim
    if H_qkv % args.world_size != 0:
        raise SystemExit(
            f"H_qkv={H_qkv} not divisible by world_size={args.world_size}"
        )
    if args.d_ffn % args.world_size != 0:
        raise SystemExit(
            f"d_ffn={args.d_ffn} not divisible by world_size={args.world_size}"
        )

    _enable_numa()
    torch.manual_seed(args.seed)

    model = GPT2_117M(
        vocab_size=args.vocab_size,
        context_size=max(args.seq_length, DEF_CONTEXT),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        d_ffn=args.d_ffn,
    ).eval()
    input_ids = torch.randint(
        0, args.vocab_size, (args.batch_size, args.seq_length), dtype=torch.long
    )

    print(
        f"GPT-2 prefill: B={args.batch_size} S={args.seq_length} "
        f"L={args.num_layers} H={args.num_heads} E={args.head_dim} "
        f"D={args.d_model} FFN={args.d_ffn} world_size={args.world_size}",
        file=sys.stderr,
    )

    compile_debug_root: str | None = None
    std_staging: str | None = None
    numa_staging: str | None = None
    if args.compile_debug:
        compile_debug_root = os.path.abspath(
            args.compile_debug_root
            or os.path.join(os.getcwd(), "gpt2_speedup_torch_compile_debug")
        )
        os.makedirs(compile_debug_root, exist_ok=True)
        std_staging = os.path.join(compile_debug_root, "standard")
        numa_staging = os.path.join(compile_debug_root, "numa_tp")
        print(
            "TORCH_COMPILE_DEBUG=1 for both paths; artifacts under "
            "<staging>/torch_compile_debug/run_*",
            file=sys.stderr,
        )
        print(f"  standard staging: {std_staging}", file=sys.stderr)
        print(f"  numa_tp staging:  {numa_staging}", file=sys.stderr)

    print("Running standard ...", file=sys.stderr, flush=True)
    std_timing, std_out, std_info = _run_path(
        model, input_ids,
        numa=False, world_size=args.world_size,
        warmup=args.warmup, iters=args.iters,
        profile=args.profile,
        max_autotune=args.max_autotune,
        compile_debug=args.compile_debug,
        compile_debug_staging=std_staging,
    )

    print(
        f"Running NUMA-TP (tp_lm_head={args.tp_lm_head}) ...",
        file=sys.stderr, flush=True,
    )
    numa_timing, numa_out, numa_info = _run_path(
        model, input_ids,
        numa=True, world_size=args.world_size,
        warmup=args.warmup, iters=args.iters,
        lm_head=args.tp_lm_head,
        profile=args.profile,
        compile_debug=args.compile_debug,
        compile_debug_staging=numa_staging,
    )

    if args.check_correctness:
        if std_out.shape != numa_out.shape:
            raise SystemExit(
                f"shape mismatch: std={std_out.shape} numa={numa_out.shape}"
            )
        if not torch.isfinite(numa_out).all():
            nan = int((~torch.isfinite(numa_out)).sum())
            raise SystemExit(f"NUMA-TP output has {nan} non-finite elements")
        abs_diff = (std_out - numa_out).abs()
        max_abs = abs_diff.max().item()
        rel_diff = abs_diff / std_out.abs().clamp_min(1e-8)
        max_rel = rel_diff.max().item()
        ok = torch.allclose(std_out, numa_out, atol=args.atol, rtol=args.rtol)
        print(
            f"correctness: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
            f"{'PASS' if ok else 'FAIL'} (atol={args.atol} rtol={args.rtol})",
            file=sys.stderr,
        )
        if not ok:
            return 1

    print(
        f"standard mean_ms={std_timing.mean_ms:.3f} median={std_timing.median_ms:.3f} "
        f"min={std_timing.min_ms:.3f}",
        file=sys.stderr,
    )
    print(
        f"numa_tp mean_ms={numa_timing.mean_ms:.3f} median={numa_timing.median_ms:.3f} "
        f"min={numa_timing.min_ms:.3f}",
        file=sys.stderr,
    )

    if args.profile:
        print("\n===== TP verification (output_code.py scan) =====", file=sys.stderr)
        for label, info in (("standard", std_info), ("numa_tp", numa_info)):
            print(
                f"  {label}: files={info['files']} triton_kernels={info['triton_kernels']} "
                f"extern_mm={info['extern_mm']} numa_args_calls={info['numa_calls']} "
                f"reassemble={info['reassemble']}",
                file=sys.stderr,
            )
            print(f"    cache_dir={info['cache_dir']}", file=sys.stderr)
            print(f"    largest_output_code={info['largest_file']}", file=sys.stderr)
        if numa_info["numa_calls"] == 0:
            print(
                "  WARNING: numa_args_calls=0 in NUMA-TP path — TP did NOT fire. "
                "The NUMA path is running the standard kernels.",
                file=sys.stderr,
            )
        else:
            print(
                f"  OK: NUMA-TP path emitted {numa_info['numa_calls']} numa-tagged "
                f"kernel calls (vs {std_info['numa_calls']} in standard, expected 0).",
                file=sys.stderr,
            )
        if "prof" in std_info:
            _print_profile("STANDARD", std_info["prof"])
        if "prof" in numa_info:
            _print_profile("NUMA-TP", numa_info["prof"])

    speedup = std_timing.mean_ms / numa_timing.mean_ms
    # stdout: just the speedup number, like the existing CLI.
    print(speedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
