#!/usr/bin/env python3
# mypy: ignore-errors
"""
Minimal Gemma decoder block (prefill phase only) compiled with ``torch.compile``,
**without** the NUMA tensor-parallel path (counterpart to the GPT-2-style script
``simple_llm_torch_compile_demo.py``).

Architecture mirrors ``gemma_pytorch/gemma/model.py`` Gemma2/3 decoder layer:

  - RMSNorm pre-/post-attention and (optional) pre-/post-FFW norms
  - Fused ``qkv_proj`` linear (no bias) producing concatenated Q,K,V with GQA
    (``num_kv_heads <= num_heads``); split with ``torch.split``
  - Optional QK-norm (per-head RMSNorm on Q and K, Gemma-3 path)
  - Rotary position embeddings on Q and K (``apply_rotary_emb``)
  - ``index_copy_`` write into preallocated K/V caches; for prefill all positions
    ``[0..S-1]`` are written in one call
  - Manual ``q.mul_(scale) @ k.T`` -> ``+ mask`` -> ``softmax`` -> ``@ v``
    (``torch.matmul``-based, not SDPA, to match the Gemma reference exactly)
  - ``o_proj`` linear (no bias) and residual
  - GeGLU MLP: ``gate_proj`` and ``up_proj`` (parallel, no bias) -> ``gelu(gate, "tanh") * up``
    -> ``down_proj`` (no bias)

Defaults are Gemma-3 270M shapes (``hidden=1152``, ``intermediate=6912``,
``num_heads=4``, ``num_kv_heads=1``, ``head_dim=256``).

Note on the existing NUMA-TP Inductor passes: they match ``addmm`` (linear with bias) +
SDPA-MATH for attention and ``addmm -> gelu -> addmm`` for MLP. Gemma's no-bias
``F.linear`` lowers to ``mm`` (not ``addmm``), and the MLP is GeGLU (not 2-linear+GELU),
so neither pass matches this graph today. This script therefore intentionally only
runs the standard path; pair it with ``simple_gemma_decoder_block_numa_tp_torch_compile_demo.py``
once the passes are extended to cover Gemma.

**Cache note:** see ``simple_llm_torch_compile_demo.py`` — same caveat about
``TORCHINDUCTOR_FX_GRAPH_CACHE``.

Example (defaults shown explicitly)::

  python scripts/inductor/simple_gemma_decoder_block_torch_compile_demo.py \\
      --num-layers 1 --batch 1 --seq 512 \\
      --h-in 1152 --num-heads 4 --num-kv-heads 1 --head-dim 256 \\
      --mlp-hidden 6912 --warmup 5 --iters 20
"""

from __future__ import annotations

import os

if os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE") is None:
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor_config

inductor_config.max_autotune_gemm_backends = "TRITON"


def precompute_freqs_cis(dim: int, end: int, theta: float = 10_000.0) -> torch.Tensor:
    """Mirror of ``gemma.model.precompute_freqs_cis`` (no rope scaling)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Mirror of ``gemma.model.apply_rotary_emb``."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1)
    )
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out


class RMSNorm(nn.Module):
    """Mirror of ``gemma.model.RMSNorm`` (with ``add_unit_offset=True`` like Gemma)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1 + self.weight.float())
        return out.type_as(x)


class GemmaAttention(nn.Module):
    """Prefill-only Gemma attention with split Q/K/V projections, GQA, RoPE,
    KV cache, optional QK-norm.

    The Gemma reference uses a single fused ``qkv_proj``; here we split it into
    three separate ``Wq``/``Wk``/``Wv`` (no-bias ``mm``) so the existing NUMA-TP
    attention pass — which matches three column-parallel linears sharing the
    same activation — has a chance to fire on this graph.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        use_qk_norm: bool,
        attn_logit_softcapping: float | None,
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5
        self.attn_logit_softcapping = attn_logit_softcapping

        # Split Q/K/V weights (no bias). Shapes follow F.linear's (out, in)
        # convention: Wq is (q_size, hidden), Wk/Wv are (kv_size, hidden).
        self.Wq = nn.Parameter(torch.empty(self.q_size, hidden_size))
        self.Wk = nn.Parameter(torch.empty(self.kv_size, hidden_size))
        self.Wv = nn.Parameter(torch.empty(self.kv_size, hidden_size))
        self.o_proj = nn.Parameter(torch.empty(hidden_size, num_heads * head_dim))
        self.query_norm = RMSNorm(head_dim, eps=rms_norm_eps) if use_qk_norm else None
        self.key_norm = RMSNorm(head_dim, eps=rms_norm_eps) if use_qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        xq = F.linear(hidden_states, self.Wq)
        xk = F.linear(hidden_states, self.Wk)
        xv = F.linear(hidden_states, self.Wv)
        xq = xq.view(B, S, self.num_heads, self.head_dim)
        xk = xk.view(B, S, self.num_kv_heads, self.head_dim)
        xv = xv.view(B, S, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)
        key, value = k_cache, v_cache
        if self.num_kv_heads != self.num_heads:
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        q = xq.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        q = q * self.scaling
        scores = torch.matmul(q, k.transpose(2, 3))
        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = F.linear(out, self.o_proj)
        return out


class GemmaMLP(nn.Module):
    """GeGLU: ``gelu(gate_proj(x), "tanh") * up_proj(x)`` -> ``down_proj``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(hidden_size, intermediate_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.gate_proj)
        gate = F.gelu(gate, approximate="tanh")
        up = F.linear(x, self.up_proj)
        fuse = gate * up
        return F.linear(fuse, self.down_proj)


class GemmaDecoderLayer(nn.Module):
    """Gemma2/3-style block with pre/post-attn and (optional) pre/post-FFW norms."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        use_qk_norm: bool,
        use_pre_ffw_norm: bool,
        use_post_ffw_norm: bool,
        attn_logit_softcapping: float | None,
    ) -> None:
        super().__init__()
        self.self_attn = GemmaAttention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            rms_norm_eps, use_qk_norm, attn_logit_softcapping,
        )
        self.mlp = GemmaMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(hidden_size, eps=rms_norm_eps) if use_pre_ffw_norm else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(hidden_size, eps=rms_norm_eps) if use_post_ffw_norm else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, freqs_cis, kv_write_indices, kv_cache, mask)
        h = self.post_attention_layernorm(h)
        h = residual + h

        residual = h
        if self.pre_feedforward_layernorm is not None:
            h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        if self.post_feedforward_layernorm is not None:
            h = self.post_feedforward_layernorm(h)
        return residual + h


class SimpleGemmaStack(nn.Module):
    """Fixed-shape, prefill-only stack of ``GemmaDecoderLayer`` modules."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        use_qk_norm: bool = True,
        use_pre_ffw_norm: bool = True,
        use_post_ffw_norm: bool = True,
        attn_logit_softcapping: float | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            GemmaDecoderLayer(
                hidden_size, num_heads, num_kv_heads, head_dim,
                intermediate_size, rms_norm_eps, use_qk_norm,
                use_pre_ffw_norm, use_post_ffw_norm, attn_logit_softcapping,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer, kv in zip(self.layers, kv_caches):
            hidden_states = layer(hidden_states, freqs_cis, kv_write_indices, kv, mask)
        return hidden_states


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=512, help="Prefill sequence length")
    p.add_argument("--h-in", type=int, default=1152)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-kv-heads", type=int, default=1, help="GQA: 1 in Gemma-3 270M")
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--mlp-hidden", type=int, default=6912)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    return p.parse_args()


def _mean_forward_seconds(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    if iters <= 0:
        return float("nan")
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def _build_prefill_inputs(
    B: int, S: int, H_in: int, num_kv_heads: int, head_dim: int,
    num_layers: int, device: torch.device,
):
    x = torch.randn(B, S, H_in, device=device)
    freqs_cis = precompute_freqs_cis(head_dim, S).to(device)
    kv_write_indices = torch.arange(S, device=device)
    kv_caches = [
        (
            torch.zeros(B, S, num_kv_heads, head_dim, device=device),
            torch.zeros(B, S, num_kv_heads, head_dim, device=device),
        )
        for _ in range(num_layers)
    ]
    mask = torch.full((1, 1, S, S), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return x, freqs_cis, kv_write_indices, kv_caches, mask


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cpu")

    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    model = SimpleGemmaStack(
        num_layers=args.num_layers,
        hidden_size=args.h_in,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.mlp_hidden,
    ).to(device).eval()
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)

    x, freqs_cis, kv_write_indices, kv_caches, mask = _build_prefill_inputs(
        args.batch, args.seq, args.h_in, args.num_kv_heads, args.head_dim,
        args.num_layers, device,
    )

    with inductor_config.patch(
        {"_numa_attention_tp": False, "_numa_mlp_tp": False}
    ), torch.no_grad():
        compiled = torch.compile(model)
        y = compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)
        y = compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)

        def run() -> None:
            nonlocal y
            y = compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)

        mean_s = _mean_forward_seconds(run, warmup=args.warmup, iters=args.iters)

    print(f"SimpleGemmaStack(layers={args.num_layers}) output shape: {tuple(y.shape)}")
    print(
        f"Mean forward (warmup={args.warmup}, iters={args.iters}): "
        f"{mean_s * 1e3:.4f} ms/iter"
    )
    print("Inductor: _numa_attention_tp=False, _numa_mlp_tp=False (standard codegen path)")
    print(
        f"Dims: B={args.batch} S={args.seq} H_in={args.h_in} "
        f"H={args.num_heads} KV_H={args.num_kv_heads} E={args.head_dim} "
        f"mlp_hidden={args.mlp_hidden}"
    )


if __name__ == "__main__":
    main()
