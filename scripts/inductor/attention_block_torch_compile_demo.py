#!/usr/bin/env python3
# mypy: ignore-errors
"""
Functional attention block (addmm Q/K/V -> MATH-backend SDPA -> addmm output)
compiled with ``torch.compile`` under the NUMA attention tensor-parallel pass.

This script enables ``torch._inductor.config._numa_attention_tp`` so Inductor
runs ``numa_attention_tp_pass`` during post-grad (see ``post_grad.py``). The
output dim ``H_qkv = num_heads * head_dim`` must be divisible by
``NUM_TP_SHARDS``, and ``num_heads`` must be divisible by ``NUM_TP_SHARDS``
for head-aligned sharding.

Architecture (mirrors the MLP NUMA TP pass):
  - FX pass rewrites only dimensions (slice nodes for Wq/Wk/Wv/Wo + biases)
    and annotates ``numa_shard_plan`` metadata. No clone nodes are inserted
    at the FX level.
  - Inductor wrapper codegen reads ``numa_shard_plan`` and emits per-NUMA-node
    ``aten.clone.default(..., numa_node=N)`` / ``empty_strided_cpu_numa`` calls
    for every input/output buffer of each kernel in the chain (Q/K/V linears,
    softmax, two attention matmuls, output linear).
  - Triton CPU launcher receives ``numa_args`` (per-node pointer tuples) and
    launches the same kernel grid N times with threads pinned to each NUMA
    socket via ``pthread_setaffinity_np``.
  - After the row-parallel output linear (``z_shard``), the wrapper emits an
    ``aten.add.Tensor`` all-reduce across per-node partial outputs.

Run with extra Dynamo debug logging:

  TORCH_COMPILE_DEBUG=1 python scripts/inductor/attention_block_torch_compile_demo.py

For Inductor graph/code logging:

  TORCH_LOGS="+graph_code" python scripts/inductor/attention_block_torch_compile_demo.py

Default (small, fast):
  python scripts/inductor/attention_block_torch_compile_demo.py

Gemma 3 1B attention: ``h-in 1152``, ``4`` heads, ``head_dim 256`` (``H_qkv=1024``).
  Optional ``--num-tp-shards`` must divide 4 heads (omit for auto vs NUMA count):
  python scripts/inductor/attention_block_torch_compile_demo.py \\
      --batch 1 --seq 512 --h-in 1152 --num-heads 4 --head-dim 256

Timing (mean compiled forward per iteration after warmup):
  python scripts/inductor/attention_block_torch_compile_demo.py --warmup 5 --iters 50
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor_config  # ensures submodule import
from torch.nn.attention import sdpa_kernel, SDPBackend

def _auto_num_tp_shards(num_heads: int, num_numa_nodes: int) -> int:
    """Largest divisor of ``num_heads`` that is <= max(1, num_numa_nodes).

    Tensor-parallel shard count must divide ``num_heads`` for head-aligned
    sharding and cannot exceed the number of NUMA nodes the runtime can pin to.
    """
    cap = max(1, int(num_numa_nodes))
    for d in range(min(cap, num_heads), 0, -1):
        if num_heads % d == 0:
            return d
    return 1


inductor_config.max_autotune_gemm_backends = "TRITON"
inductor_config.max_autotune = True


class AttentionBlock(nn.Module):
    """Functional MHA block aligned with ``numa_attention_tp_pass``'s matcher.

    Shapes: ``x`` [B*S, H_in], ``Wq/Wk/Wv`` [H_in, H*E], ``bq/bk/bv`` [H*E],
    ``Wo`` [H*E, H_in], ``bo`` [H_in] -> out [B*S, H_in].
    """

    def __init__(self, batch: int, seq: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.B = batch
        self.S = seq
        self.H = num_heads
        self.E = head_dim

    def forward(
        self,
        x: torch.Tensor,
        Wq: torch.Tensor,
        Wk: torch.Tensor,
        Wv: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        Wo: torch.Tensor,
        bo: torch.Tensor,
    ) -> torch.Tensor:
        B, S, H, E = self.B, self.S, self.H, self.E
        q = torch.addmm(bq, x, Wq).view(B, S, H, E).transpose(1, 2)
        k = torch.addmm(bk, x, Wk).view(B, S, H, E).transpose(1, 2)
        v = torch.addmm(bv, x, Wv).view(B, S, H, E).transpose(1, 2)
        # MATH backend forces the in-graph decomposition into primitive ops
        # (matmul + _safe_softmax + matmul) so every heavy op is a Triton
        # kernel and the numa_args pinning path works end-to-end.
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B * S, H * E)
        return torch.addmm(bo, y, Wo)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch size B")
    parser.add_argument("--seq", type=int, default=6, help="Sequence length S")
    parser.add_argument("--h-in", type=int, default=12, help="Input hidden dim H_in")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of heads H")
    parser.add_argument("--head-dim", type=int, default=8, help="Per-head dim E")
    parser.add_argument(
        "--num-tp-shards",
        type=int,
        default=None,
        metavar="N",
        help=(
            "NUMA TP world size; must divide num_heads and be <= detected NUMA "
            "node count. Default: largest divisor of --num-heads that is <= that "
            "count (falls back to 1 if none)."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        metavar="N",
        help="Warmup forward passes before timing (not counted).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        metavar="N",
        help="Timed forward passes; mean time is reported over this many runs.",
    )
    return parser.parse_args()


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

    B = args.batch
    S = args.seq
    H_in = args.h_in
    H = args.num_heads
    E = args.head_dim
    H_qkv = H * E

    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        raise RuntimeError(
            "NUMA runtime control hooks are unavailable in this build "
            "(missing torch._C._cpu._set_numa_enabled)."
        )
    if not hasattr(cpu_mod, "_is_numa_enabled") or not hasattr(
        cpu_mod, "_num_numa_nodes"
    ):
        raise RuntimeError(
            "NUMA runtime query hooks are unavailable in this build "
            "(need _is_numa_enabled and _num_numa_nodes)."
        )

    old_numa_enabled = cpu_mod._is_numa_enabled()
    cpu_mod._set_numa_enabled(True)
    if not cpu_mod._is_numa_enabled():
        raise RuntimeError(
            "Failed to enable NUMA runtime via torch._C._cpu._set_numa_enabled(True)"
        )
    n_nodes = int(cpu_mod._num_numa_nodes())

    if args.num_tp_shards is None:
        num_tp_shards = _auto_num_tp_shards(H, n_nodes)
        tp_shards_source = "auto"
    else:
        num_tp_shards = args.num_tp_shards
        tp_shards_source = "cli"
        if n_nodes < num_tp_shards:
            raise RuntimeError(
                f"Need at least {num_tp_shards} NUMA nodes, but runtime reports "
                f"{n_nodes}. Lower --num-tp-shards or run on a machine with more "
                "NUMA nodes."
            )

    assert H % num_tp_shards == 0, (
        f"num_heads={H} must be divisible by num_tp_shards={num_tp_shards} "
        "for head-aligned sharding."
    )
    assert H_qkv % num_tp_shards == 0, (
        f"H_qkv={H_qkv} must be divisible by num_tp_shards={num_tp_shards}."
    )

    x = torch.randn(B * S, H_in, device=device)
    Wq = torch.randn(H_in, H_qkv, device=device)
    Wk = torch.randn(H_in, H_qkv, device=device)
    Wv = torch.randn(H_in, H_qkv, device=device)
    bq = torch.randn(H_qkv, device=device)
    bk = torch.randn(H_qkv, device=device)
    bv = torch.randn(H_qkv, device=device)
    Wo = torch.randn(H_qkv, H_in, device=device)
    bo = torch.randn(H_in, device=device)

    model = AttentionBlock(B, S, H, E).to(device)

    try:
        with inductor_config.patch(
            {
                "_numa_attention_tp": True,
                "_numa_attention_tp_world_size": num_tp_shards,
                "_numa_attention_tp_compute_shard": 0,
            }
        ):
            compiled = torch.compile(model)
            y = compiled(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)
            y = compiled(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)

            def run_compiled() -> None:
                nonlocal y
                y = compiled(x, Wq, Wk, Wv, bq, bk, bv, Wo, bo)

            compiled_mean_s = _mean_forward_seconds(
                run_compiled, warmup=args.warmup, iters=args.iters
            )
    finally:
        cpu_mod._set_numa_enabled(old_numa_enabled)

    print(
        "AttentionBlock (addmm Q/K/V -> MATH SDPA -> addmm output) "
        f"output shape: {tuple(y.shape)}"
    )
    print(
        f"Mean forward time (warmup={args.warmup}, iters={args.iters}): "
        f"compiled {compiled_mean_s * 1e3:.4f} ms/iter"
    )
    print(
        "Inductor: _numa_attention_tp=True, "
        f"_numa_attention_tp_world_size={num_tp_shards}, "
        "_numa_attention_tp_compute_shard=0"
    )
    if tp_shards_source == "auto":
        print(
            f"num_tp_shards={num_tp_shards} (auto: largest divisor of "
            f"num_heads={H} with N<={n_nodes})"
        )
    else:
        print(f"num_tp_shards={num_tp_shards} (from --num-tp-shards)")
    print(f"NUMA runtime: enabled=True, _num_numa_nodes()={n_nodes}")
    print(f"Dims: B={B} S={S} H_in={H_in} H={H} E={E} H_qkv={H_qkv}")

    if os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1":
        try:
            from torch._dynamo.utils import get_debug_dir

            log_path = os.path.join(get_debug_dir(), "torchdynamo", "debug.log")
            print("TORCH_COMPILE_DEBUG=1 is set; Dynamo file log:", log_path)
        except Exception as e:
            print("Could not resolve Dynamo debug log path:", e)


if __name__ == "__main__":
    main()
