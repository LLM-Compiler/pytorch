#!/usr/bin/env python3
# mypy: ignore-errors
"""
Minimal multi-layer "LLM" stack (attention + residual + MLP + residual) compiled with
``torch.compile`` so both NUMA tensor-parallel Inductor passes can run on one graph.

Pattern requirements (see ``torch/_inductor/fx_passes/numa_attention_tp.py`` and
``numa_mlp_tp.py``):

**Attention** — three column-parallel linears sharing the same activation ``x`` as
``mat1``, then MATH-backend SDPA (decomposes to primitive ops for Triton), then one
row-parallel output linear:

  ``addmm(bq,x,Wq)``, ``addmm(bk,x,Wk)``, ``addmm(bv,x,Wv)``
  -> view/transpose -> ``F.scaled_dot_product_attention`` under ``SDPBackend.MATH``
  -> ``addmm(bo, y, Wo)``

**MLP** — two linear layers with GELU between (``F.gelu`` or decomposed GELU):

  ``addmm(b1, x, A)`` -> ``F.gelu`` -> ``addmm(b2, y, B)``

Constraints for shard count ``N`` (``_numa_*_world_size``):

  - ``num_heads % N == 0`` and ``(num_heads * head_dim) % N == 0`` (attention)
  - ``mlp_hidden_dim % N == 0`` (MLP)

``post_grad.py`` runs ``numa_mlp_tp_pass`` then ``numa_attention_tp_pass`` when both
config flags are set.

**Defaults** are Gemma-1B–scale (``H_in=1152``, ``seq=512``, ``H_qkv=1024``, ``mlp_hidden=6912``)
so matmul work dominates NUMA clone/launcher overhead; use smaller flags only for fast
debug.

Example (explicit flags = current defaults):

  python scripts/inductor/simple_llm_numa_tp_torch_compile_demo.py \\
      --num-layers 1 --batch 1 --seq 512 --h-in 1152 --num-heads 4 --head-dim 256 \\
      --mlp-hidden 6912 --warmup 5 --iters 20

Small / quick smoke test:

  python scripts/inductor/simple_llm_numa_tp_torch_compile_demo.py \\
      --seq 8 --h-in 64 --head-dim 16 --mlp-hidden 128 --warmup 1 --iters 2
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor_config
from torch.nn.attention import sdpa_kernel, SDPBackend

inductor_config.max_autotune_gemm_backends = "TRITON"
inductor_config.max_autotune = True


def _auto_num_tp_shards(num_heads: int, num_numa_nodes: int) -> int:
    cap = max(1, int(num_numa_nodes))
    for d in range(min(cap, num_heads), 0, -1):
        if num_heads % d == 0:
            return d
    return 1


class TransformerLayer(nn.Module):
    """One decoder-style block: pre-norm-free residual attention + residual MLP.

    Activations are ``[B*S, H_in]``. Attention uses the same hidden size for
    input and output; MLP maps ``H_in -> mlp_hidden -> H_in``.
    """

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
        # --- Attention (matches numa_attention_tp_pass) ---
        q = torch.addmm(self.bq, x, self.Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(self.bk, x, self.Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(self.bv, x, self.Wv).view(B, S, H, E).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        attn_out = torch.addmm(self.bo, y, self.Wo)
        h = attn_out + x

        # --- MLP (matches numa_mlp_tp_pass: addmm -> gelu -> addmm) ---
        u = torch.addmm(self.mlp_fc1_b, h, self.mlp_fc1_w)
        u = F.gelu(u)
        mlp_out = torch.addmm(self.mlp_fc2_b, u, self.mlp_fc2_w)
        return mlp_out + h


class SimpleLmStack(nn.Module):
    """``num_layers`` stacked ``TransformerLayer`` modules."""

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num-layers", type=int, default=1, help="Number of transformer blocks")
    p.add_argument("--batch", type=int, default=1, help="Batch size B")
    p.add_argument("--seq", type=int, default=512, help="Sequence length S")
    p.add_argument("--h-in", type=int, default=1152, help="Hidden size H_in")
    p.add_argument("--num-heads", type=int, default=4, help="Attention heads H")
    p.add_argument("--head-dim", type=int, default=256, help="Per-head dim E (H_qkv = H*E)")
    p.add_argument(
        "--mlp-hidden",
        type=int,
        default=6912,
        help="MLP intermediate dim (must be divisible by num_tp_shards)",
    )
    p.add_argument(
        "--num-tp-shards",
        type=int,
        default=None,
        help="NUMA TP world size for both passes; default: auto from heads vs NUMA count",
    )
    p.add_argument("--warmup", type=int, default=5, help="Warmup forwards before timed iters")
    p.add_argument("--iters", type=int, default=20, help="Timed forwards (mean ms/iter)")
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


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, S = args.batch, args.seq
    H_in = args.h_in
    H = args.num_heads
    E = args.head_dim
    H_qkv = H * E
    d_mlp = args.mlp_hidden

    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        raise RuntimeError(
            "NUMA runtime hooks missing (torch._C._cpu._set_numa_enabled)."
        )
    if not hasattr(cpu_mod, "_is_numa_enabled") or not hasattr(
        cpu_mod, "_num_numa_nodes"
    ):
        raise RuntimeError("NUMA query hooks missing (_is_numa_enabled, _num_numa_nodes).")

    old_numa = cpu_mod._is_numa_enabled()
    cpu_mod._set_numa_enabled(True)
    if not cpu_mod._is_numa_enabled():
        raise RuntimeError("Failed to enable NUMA via _set_numa_enabled(True)")
    n_nodes = int(cpu_mod._num_numa_nodes())

    if args.num_tp_shards is None:
        num_tp_shards = _auto_num_tp_shards(H, n_nodes)
        tp_src = "auto"
    else:
        num_tp_shards = args.num_tp_shards
        tp_src = "cli"
        if n_nodes < num_tp_shards:
            raise RuntimeError(
                f"Need at least {num_tp_shards} NUMA nodes, got {n_nodes}."
            )

    if H % num_tp_shards != 0:
        raise ValueError(
            f"num_heads={H} must be divisible by num_tp_shards={num_tp_shards}"
        )
    if H_qkv % num_tp_shards != 0:
        raise ValueError(
            f"H_qkv={H_qkv} must be divisible by num_tp_shards={num_tp_shards}"
        )
    if d_mlp % num_tp_shards != 0:
        raise ValueError(
            f"mlp_hidden={d_mlp} must be divisible by num_tp_shards={num_tp_shards}"
        )

    model = SimpleLmStack(
        args.num_layers, H, E, H_in, d_mlp,
    ).to(device)
    model.set_batch_seq(B, S)
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)

    x = torch.randn(B * S, H_in, device=device)

    try:
        with inductor_config.patch(
            {
                "_numa_attention_tp": True,
                "_numa_attention_tp_world_size": num_tp_shards,
                "_numa_attention_tp_compute_shard": 0,
                "_numa_mlp_tp": True,
                "_numa_mlp_tp_world_size": num_tp_shards,
                "_numa_mlp_tp_compute_shard": 0,
            }
        ):
            compiled = torch.compile(model)
            y = compiled(x)
            y = compiled(x)

            def run() -> None:
                nonlocal y
                y = compiled(x)

            mean_s = _mean_forward_seconds(run, warmup=args.warmup, iters=args.iters)
    finally:
        cpu_mod._set_numa_enabled(old_numa)

    print(
        f"SimpleLmStack(layers={args.num_layers}) output shape: {tuple(y.shape)}"
    )
    print(
        f"Mean forward (warmup={args.warmup}, iters={args.iters}): "
        f"{mean_s * 1e3:.4f} ms/iter"
    )
    print(
        "Inductor: _numa_attention_tp and _numa_mlp_tp enabled, "
        f"world_size={num_tp_shards}, compute_shard=0"
    )
    if tp_src == "auto":
        print(
            f"num_tp_shards={num_tp_shards} (auto: divisor of num_heads={H}, N<={n_nodes})"
        )
    else:
        print(f"num_tp_shards={num_tp_shards} (from --num-tp-shards)")
    print(f"NUMA: _num_numa_nodes()={n_nodes}")
    print(
        f"Dims: B={B} S={S} H_in={H_in} H={H} E={E} H_qkv={H_qkv} mlp_hidden={d_mlp}"
    )

    if os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1":
        try:
            from torch._dynamo.utils import get_debug_dir

            print(
                "TORCH_COMPILE_DEBUG=1: Dynamo log:",
                os.path.join(get_debug_dir(), "torchdynamo", "debug.log"),
            )
        except Exception as e:
            print("Could not resolve Dynamo debug log:", e)


if __name__ == "__main__":
    main()
