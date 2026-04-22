#!/usr/bin/env python3
# mypy: ignore-errors
"""
Same minimal LLM stack as ``simple_llm_numa_tp_torch_compile_demo.py`` (same default
dims and ``torch.compile`` + Triton autotune), but **without** the NUMA tensor-parallel
path: no ``torch._C._cpu._set_numa_enabled``, and Inductor ``_numa_attention_tp`` /
``_numa_mlp_tp`` are explicitly disabled.

Use this as a baseline to compare against the NUMA script.

**Cache note:** Inductor’s FX graph cache key uses ``save_config_portable()``, which omits
config keys starting with ``_``. So ``_numa_mlp_tp`` / ``_numa_attention_tp`` do **not**
change the hash. If you previously compiled the same graph with NUMA on, you can get a
cache hit that still emits ``clone(..., numa_node=...)``, which then fails unless NUMA
runtime is enabled. This script sets ``TORCHINDUCTOR_FX_GRAPH_CACHE=0`` by default (before
``import torch``) to avoid that collision; or clear the Inductor cache under ``~/.cache/torch``.

**Defaults** match ``simple_llm_numa_tp_torch_compile_demo.py`` (Gemma-1B–scale shapes).

Example (explicit flags = current defaults):

  python scripts/inductor/simple_llm_torch_compile_demo.py \\
      --num-layers 1 --batch 1 --seq 512 --h-in 1152 --num-heads 4 --head-dim 256 \\
      --mlp-hidden 6912 --warmup 5 --iters 20
"""

from __future__ import annotations

import os

# See docstring: FX graph cache does not hash `_numa_*` flags; avoid reusing a
# NUMA-tagged compilation of the same graph.
if os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE") is None:
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor_config
from torch.nn.attention import sdpa_kernel, SDPBackend

inductor_config.max_autotune_gemm_backends = "TRITON"
inductor_config.max_autotune = True


class TransformerLayer(nn.Module):
    """One decoder-style block: pre-norm-free residual attention + residual MLP."""

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
    p.add_argument("--mlp-hidden", type=int, default=6912, help="MLP intermediate dimension")
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

    model = SimpleLmStack(
        args.num_layers, H, E, H_in, d_mlp,
    ).to(device)
    model.set_batch_seq(B, S)
    for p in model.parameters():
        nn.init.normal_(p, std=0.02)

    x = torch.randn(B * S, H_in, device=device)

    with inductor_config.patch(
        {
            "_numa_attention_tp": False,
            "_numa_mlp_tp": False,
        }
    ):
        compiled = torch.compile(model)
        y = compiled(x)
        y = compiled(x)

        def run() -> None:
            nonlocal y
            y = compiled(x)

        mean_s = _mean_forward_seconds(run, warmup=args.warmup, iters=args.iters)

    print(
        f"SimpleLmStack(layers={args.num_layers}) output shape: {tuple(y.shape)}"
    )
    print(
        f"Mean forward (warmup={args.warmup}, iters={args.iters}): "
        f"{mean_s * 1e3:.4f} ms/iter"
    )
    print(
        "Inductor: _numa_attention_tp=False, _numa_mlp_tp=False (standard codegen path)"
    )
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
            print("Could not resolve Dynamo log path:", e)


if __name__ == "__main__":
    main()
