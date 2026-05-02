#!/usr/bin/env python3
"""
End-to-end correctness check for a full Pre-LN transformer block compiled
under all NUMA tensor-parallel passes (LN + attention + residual + MLP).

Block under test (GPT-2 style Pre-LN):

    def forward(x):
        x = x + attn(LN1(x))
        x = x + mlp(LN2(x))
        return x

Compiles the same block twice:
  1. **Reference**: ``torch.compile`` with all NUMA passes disabled.
  2. **NUMA-TP**: ``torch.compile`` with LN + attn + MLP + residual NUMA TP enabled.

Compares outputs element-wise and reports max abs / rel diff.

Usage:
    python scripts/inductor/check_numa_preLN_block_correctness.py \
        --batch 1 --seq 1024 --h-in 768 --num-heads 12 --head-dim 64 \
        --mlp-hidden 3072 --world-size 2
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch._inductor import config as inductor_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--h-in", type=int, default=768)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--mlp-hidden", type=int, default=3072)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    return p.parse_args()


class PreLNBlock(torch.nn.Module):
    def __init__(self, h_in, num_heads, head_dim, mlp_hidden):
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
        self.mlp_fc1_w = torch.nn.Parameter(torch.empty(h_in, mlp_hidden))
        self.mlp_fc1_b = torch.nn.Parameter(torch.empty(mlp_hidden))
        self.mlp_fc2_w = torch.nn.Parameter(torch.empty(mlp_hidden, h_in))
        self.mlp_fc2_b = torch.nn.Parameter(torch.empty(h_in))

        for p in (self.Wq, self.Wk, self.Wv, self.Wo, self.mlp_fc1_w, self.mlp_fc2_w):
            torch.nn.init.normal_(p, std=0.02)
        for p in (self.bq, self.bk, self.bv, self.bo, self.mlp_fc1_b, self.mlp_fc2_b):
            torch.nn.init.zeros_(p)

    def _attn(self, xn, B, S):
        H, E = self.H, self.E
        x_flat = xn.reshape(B * S, self.H_in)
        q = torch.addmm(self.bq, x_flat, self.Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(self.bk, x_flat, self.Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(self.bv, x_flat, self.Wv).view(B, S, H, E).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        return torch.addmm(self.bo, y, self.Wo).view(B, S, self.H_in)

    def _mlp(self, xn, B, S):
        x_flat = xn.reshape(B * S, self.H_in)
        h = torch.addmm(self.mlp_fc1_b, x_flat, self.mlp_fc1_w)
        h = F.gelu(h)
        return torch.addmm(self.mlp_fc2_b, h, self.mlp_fc2_w).view(B, S, self.H_in)

    def forward(self, x):
        B, S, _ = x.shape
        x = x + self._attn(self.ln1(x), B, S)
        x = x + self._mlp(self.ln2(x), B, S)
        return x


def _enable_numa() -> None:
    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        raise RuntimeError("NUMA runtime hooks unavailable in this build.")
    cpu_mod._set_numa_enabled(True)
    print(
        f"NUMA enabled={cpu_mod._is_numa_enabled()} "
        f"nodes={cpu_mod._num_numa_nodes()}",
        flush=True,
    )


def _compile_with(model, x, *, numa: bool, world_size: int):
    flags = {
        "_numa_layer_norm_tp": numa,
        "_numa_attention_tp": numa,
        "_numa_mlp_tp": numa,
        "_numa_residual_tp": numa,
        "_numa_layer_norm_tp_world_size": world_size if numa else 0,
        "_numa_attention_tp_world_size": world_size if numa else 0,
        "_numa_mlp_tp_world_size": world_size if numa else 0,
    }
    with inductor_config.patch(flags):
        compiled = torch.compile(model, mode="default", fullgraph=True)
        with torch.no_grad():
            out = compiled(x)
    return out


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)

    if (args.num_heads * args.head_dim) % args.world_size != 0:
        raise SystemExit(
            f"H_qkv={args.num_heads * args.head_dim} not divisible by "
            f"world_size={args.world_size}"
        )
    if args.mlp_hidden % args.world_size != 0:
        raise SystemExit(
            f"mlp_hidden={args.mlp_hidden} not divisible by world_size={args.world_size}"
        )

    _enable_numa()

    model = PreLNBlock(
        args.h_in, args.num_heads, args.head_dim, args.mlp_hidden
    ).eval()
    x = torch.randn(args.batch, args.seq, args.h_in)

    print("Compiling reference (NUMA off) ...", flush=True)
    ref = _compile_with(model, x, numa=False, world_size=args.world_size)

    print("Compiling NUMA-TP path ...", flush=True)
    got = _compile_with(model, x, numa=True, world_size=args.world_size)

    if ref.shape != got.shape:
        raise SystemExit(f"shape mismatch: ref={ref.shape} got={got.shape}")
    if not torch.isfinite(got).all():
        nan = int((~torch.isfinite(got)).sum())
        raise SystemExit(f"NUMA-TP output has {nan} non-finite elements")

    abs_diff = (ref - got).abs()
    rel_diff = abs_diff / (ref.abs().clamp_min(1e-8))
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    print(f"max abs diff = {max_abs:.3e}")
    print(f"max rel diff = {max_rel:.3e}")

    ok = torch.allclose(ref, got, atol=args.atol, rtol=args.rtol)
    if ok:
        print(f"PASS (atol={args.atol} rtol={args.rtol})")
        return 0
    print(f"FAIL (atol={args.atol} rtol={args.rtol})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
