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
import math
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch._inductor import config as inductor_config


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
        # LM head with tied weights to the token embedding (no bias).
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
        # LM head: tied to token embedding -> matmul with embedding.weight.T.
        logits = F.linear(x, self.token_embedding.weight)
        return logits


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


def _flags(numa: bool, world_size: int) -> dict:
    return {
        "_numa_layer_norm_tp": numa,
        "_numa_attention_tp": numa,
        "_numa_mlp_tp": numa,
        "_numa_residual_tp": numa,
        "_numa_layer_norm_tp_world_size": world_size if numa else 0,
        "_numa_attention_tp_world_size": world_size if numa else 0,
        "_numa_mlp_tp_world_size": world_size if numa else 0,
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


def _run_path(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    numa: bool,
    world_size: int,
    warmup: int,
    iters: int,
) -> tuple[TimingSummary, torch.Tensor]:
    with inductor_config.patch(_flags(numa, world_size)):
        compiled = _compile(model, numa=numa, world_size=world_size)
        # Run once to capture output for the correctness check (also primes
        # the cache before timing). Counts toward warmup.
        with torch.no_grad():
            ref_out = compiled(input_ids).detach().clone()
        remaining_warmup = max(0, warmup - 1)
        timing = _time_forward(
            compiled, input_ids, warmup=remaining_warmup, iters=iters
        )
    return timing, ref_out


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
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
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

    print("Running standard ...", file=sys.stderr, flush=True)
    std_timing, std_out = _run_path(
        model, input_ids,
        numa=False, world_size=args.world_size,
        warmup=args.warmup, iters=args.iters,
    )

    print("Running NUMA-TP ...", file=sys.stderr, flush=True)
    numa_timing, numa_out = _run_path(
        model, input_ids,
        numa=True, world_size=args.world_size,
        warmup=args.warmup, iters=args.iters,
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

    speedup = std_timing.mean_ms / numa_timing.mean_ms
    # stdout: just the speedup number, like the existing CLI.
    print(speedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
