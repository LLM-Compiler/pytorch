#!/usr/bin/env python3
# mypy: ignore-errors
r"""
GPT-2 Pre-LN prefill speedup CLI (multi-size presets).

Same benchmark as ``gpt2_speedup_cli.py`` (standard ``torch.compile`` vs
NUMA-TP Inductor path), with a ``--model`` preset for GPT-2–shaped sizes:

  * ``117m``  — 12 layers, d_model 768
  * ``345m``  — 24 layers, d_model 1024
  * ``762m``  — 36 layers, d_model 1280
  * ``1542m`` — 48 layers, d_model 1600

Head count follows OpenAI GPT-2 (head_dim 64, FFN = 4 * d_model). Override
any dimension with ``--num-layers``, ``--d-model``, etc. when set.

Example::

    python scripts/inductor/gpt2_multi_size_speedup_cli.py \\
        --model 345m --batch-size 1 --seq-length 512 --world-size 2

    python scripts/inductor/gpt2_multi_size_speedup_cli.py \\
        --model 117m --check-correctness --world-size 2
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
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch._inductor import config as inductor_config

inductor_config.max_autotune_gemm_backends = "TRITON"

DEF_VOCAB = 50257
DEF_CONTEXT = 1024


@dataclass(frozen=True)
class GPT2SizePreset:
    """Architectural hyperparameters for one GPT-2–style width."""

    label: str
    num_layers: int
    d_model: int
    num_heads: int
    head_dim: int
    d_ffn: int


# d_model / num_heads == head_dim (64) for all; d_ffn == 4 * d_model
MODEL_PRESETS: Dict[str, GPT2SizePreset] = {
    "117m": GPT2SizePreset(
        label="GPT-2 117M",
        num_layers=12,
        d_model=768,
        num_heads=12,
        head_dim=64,
        d_ffn=3072,
    ),
    "345m": GPT2SizePreset(
        label="GPT-2 345M",
        num_layers=24,
        d_model=1024,
        num_heads=16,
        head_dim=64,
        d_ffn=4096,
    ),
    "762m": GPT2SizePreset(
        label="GPT-2 762M",
        num_layers=36,
        d_model=1280,
        num_heads=20,
        head_dim=64,
        d_ffn=5120,
    ),
    "1542m": GPT2SizePreset(
        label="GPT-2 1542M",
        num_layers=48,
        d_model=1600,
        num_heads=25,
        head_dim=64,
        d_ffn=6400,
    ),
}


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


class GPT2PreLN(torch.nn.Module):
    """Minimal Pre-LN GPT-2–shaped model (random init; prefill through final LN)."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        d_ffn: int,
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
                for p in (blk.Wq, blk.Wk, blk.Wv, blk.mlp_fc1_w):
                    torch.nn.init.normal_(p, std=std)
                for p in (blk.Wo, blk.mlp_fc2_w):
                    torch.nn.init.normal_(p, std=residual_std)
                for p in (
                    blk.bq, blk.bk, blk.bv, blk.bo,
                    blk.mlp_fc1_b, blk.mlp_fc2_b,
                ):
                    torch.nn.init.zeros_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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
    return {
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
    latencies_ms_out: Optional[List[float]] = None,
) -> TimingSummary:
    """If ``latencies_ms_out`` is a list, append one float per timed iteration (chronological)."""
    with torch.no_grad():
        for _ in range(warmup):
            compiled(input_ids)
        ordered: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            compiled(input_ids)
            ordered.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(ordered)
    if latencies_ms_out is not None:
        latencies_ms_out.clear()
        latencies_ms_out.extend(ordered)
    if not ordered:
        return TimingSummary(mean_ms=0.0, median_ms=0.0, min_ms=0.0, iters=0)
    return TimingSummary(
        mean_ms=sum(ordered) / len(ordered),
        median_ms=s[len(s) // 2],
        min_ms=s[0],
        iters=iters,
    )


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
    record_iteration_latencies: bool = False,
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
        torch._dynamo.reset()
        with inductor_config.patch(
            _flags(numa, world_size, lm_head=lm_head, max_autotune=max_autotune)
        ):
            compiled = _compile(model, numa=numa, world_size=world_size)
            with torch.no_grad():
                ref_out = compiled(input_ids).detach().clone()
            info.update(_scan_output_code(cache_dir, since=compile_started))
            info["cache_dir"] = cache_dir
            remaining_warmup = max(0, warmup - 1)
            iter_lat: list[float] | None = [] if record_iteration_latencies else None
            timing = _time_forward(
                compiled,
                input_ids,
                warmup=remaining_warmup,
                iters=iters,
                latencies_ms_out=iter_lat,
            )
            if record_iteration_latencies and iter_lat is not None:
                info["latencies_ms"] = list(iter_lat)
            if profile:
                info["prof"] = _profile_forward(
                    compiled, input_ids, iters=min(iters, 10)
                )
    finally:
        if cdbg_state is not None:
            _exit_compile_debug(cdbg_state)
    return timing, ref_out, info


def _resolve_arch(args: argparse.Namespace, preset: GPT2SizePreset) -> dict:
    """Apply CLI overrides on top of ``--model`` preset (None = use preset)."""

    def pick(name: str, preset_val: int) -> int:
        v = getattr(args, name)
        return preset_val if v is None else v

    num_layers = pick("num_layers", preset.num_layers)
    d_model = pick("d_model", preset.d_model)
    num_heads = pick("num_heads", preset.num_heads)
    head_dim = pick("head_dim", preset.head_dim)
    d_ffn = pick("d_ffn", preset.d_ffn)
    if num_heads * head_dim != d_model:
        raise SystemExit(
            f"Architecture inconsistency: num_heads*head_dim={num_heads * head_dim} "
            f"!= d_model={d_model}"
        )
    return {
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "d_ffn": d_ffn,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    model_choices = sorted(MODEL_PRESETS.keys())
    preset_help = "; ".join(
        f"{k}: L={v.num_layers} D={v.d_model} H={v.num_heads} FFN={v.d_ffn}"
        for k, v in sorted(MODEL_PRESETS.items(), key=lambda kv: kv[0])
    )
    p = argparse.ArgumentParser(
        description=(
            "GPT-2 Pre-LN prefill benchmark (standard vs NUMA-TP). "
            "Select architecture with --model; optional --num-layers etc. override the preset."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="117m",
        choices=model_choices,
        metavar="SIZE",
        help=f"Model preset. Presets: {preset_help}",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-length", type=int, default=DEF_CONTEXT)
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override layer count (default: from --model preset).",
    )
    p.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Override attention heads (default: from --model preset).",
    )
    p.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Override per-head dimension (default: from --model preset).",
    )
    p.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Override hidden size (default: from --model preset).",
    )
    p.add_argument(
        "--d-ffn",
        type=int,
        default=None,
        help="Override MLP intermediate size (default: from --model preset).",
    )
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
            "Also TP the LM head matmul (Inductor NUMA flag only; this model has no LM head)."
        ),
    )
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument(
        "--max-autotune",
        action="store_true",
        help="Enable Inductor max_autotune for matmuls.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Run torch.profiler for both paths and scan Inductor output for NUMA tags.",
    )
    p.add_argument(
        "--compile-debug",
        action="store_true",
        help="TORCH_COMPILE_DEBUG=1 with staging under --compile-debug-root.",
    )
    p.add_argument(
        "--compile-debug-root",
        type=str,
        default=None,
        help=(
            "Base directory for compile debug; subfolders standard/ and numa_tp/. "
            "Default: <cwd>/gpt2_multi_size_torch_compile_debug"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    preset = MODEL_PRESETS[args.model]
    arch = _resolve_arch(args, preset)

    if args.seq_length > DEF_CONTEXT:
        print(
            f"warning: seq_length={args.seq_length} > default context={DEF_CONTEXT}; "
            f"position table size uses max(seq_length, {DEF_CONTEXT}).",
            file=sys.stderr,
        )
    H_qkv = arch["num_heads"] * arch["head_dim"]
    if H_qkv % args.world_size != 0:
        raise SystemExit(
            f"H_qkv={H_qkv} not divisible by world_size={args.world_size}"
        )
    if arch["d_ffn"] % args.world_size != 0:
        raise SystemExit(
            f"d_ffn={arch['d_ffn']} not divisible by world_size={args.world_size}"
        )

    _enable_numa()
    torch.manual_seed(args.seed)

    model = GPT2PreLN(
        vocab_size=args.vocab_size,
        context_size=max(args.seq_length, DEF_CONTEXT),
        d_model=arch["d_model"],
        num_layers=arch["num_layers"],
        num_heads=arch["num_heads"],
        head_dim=arch["head_dim"],
        d_ffn=arch["d_ffn"],
    ).eval()
    input_ids = torch.randint(
        0, args.vocab_size, (args.batch_size, args.seq_length), dtype=torch.long
    )

    print(
        f"{preset.label} ({args.model}) prefill: B={args.batch_size} S={args.seq_length} "
        f"L={arch['num_layers']} H={arch['num_heads']} E={arch['head_dim']} "
        f"D={arch['d_model']} FFN={arch['d_ffn']} world_size={args.world_size}",
        file=sys.stderr,
    )

    compile_debug_root: str | None = None
    std_staging: str | None = None
    numa_staging: str | None = None
    if args.compile_debug:
        compile_debug_root = os.path.abspath(
            args.compile_debug_root
            or os.path.join(os.getcwd(), "gpt2_multi_size_torch_compile_debug")
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
    print(speedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
