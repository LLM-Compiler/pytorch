#!/usr/bin/env python3
# mypy: ignore-errors
"""
Compare three ``SimpleLmStack`` forward paths on CPU:

1. **Eager** — same module code as ``simple_llm_numa_tp_torch_compile_demo.py`` (no compile).
2. **Standard Inductor** — ``torch.compile`` with ``_numa_attention_tp=False`` and
   ``_numa_mlp_tp=False``.
3. **NUMA Inductor** — ``torch.compile`` with both NUMA passes on, NUMA runtime enabled,
   and shard settings matching the NUMA demo.

Weights and input ``x`` are identical across paths (``deepcopy`` + shared ``state_dict``).
Inductor FX graph cache is disabled by default before ``import torch`` so NUMA vs non-NUMA
compilations cannot alias (``save_config_portable()`` omits ``_numa_*`` keys).

Checks (in order):

- ``y_ref`` vs ``y_std`` — standard Inductor matches eager (sanity baseline).
- ``y_std`` vs ``y_numa`` — NUMA-tagged compile matches standard compile (same math as
  non-sharded if TP is correct).
- ``y_ref`` vs ``y_numa`` — reported as a third line so you see **direct** agreement with
  eager without relying only on transitivity.

**Limits:** This does not prove the FX passes ran (inspect ``output_code.py`` / ``numa_args``
when ``num_tp_shards>=2``). A PASS with ``num_tp_shards==1`` is trivial equivalence. Very
small ``max_abs`` on the std vs NUMA line is expected; bit-identical ``0`` can happen when
schedules match, but is not guaranteed for all shapes.

Example:

  python scripts/inductor/check_simple_llm_numa_correctness.py

  python scripts/inductor/check_simple_llm_numa_correctness.py \\
      --num-layers 1 --batch 1 --seq 512 --h-in 1152 --num-heads 4 --head-dim 256 \\
      --mlp-hidden 6912 --rtol 1e-4 --atol 1e-4

**Interpreting results:** If ``num_tp_shards`` is 1, the NUMA FX passes do not shard; use
``num_tp_shards>=2`` on a host with at least that many NUMA nodes to stress TP.

**Node count:** ``torch._C._cpu._num_numa_nodes()`` wraps ``c10::GetNumNUMANodes()``, which
returns **-1** unless NUMA is enabled at runtime (``FLAGS_caffe2_cpu_numa_enabled``) *and*
libnuma is available (see ``c10/util/numa.cpp``). This script enables NUMA **before** reading
the count so a 2-node machine reports ``2``, not ``-1``.

Use ``--max-autotune`` to match the demo scripts (verbose AUTOTUNE logs, slower).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

if os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE") is None:
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor import config as inductor_config
from torch.nn.attention import sdpa_kernel, SDPBackend

inductor_config.max_autotune_gemm_backends = "TRITON"


def _auto_num_tp_shards(num_heads: int, num_numa_nodes: int) -> int:
    """Largest divisor of ``num_heads`` that is <= max(1, reported NUMA nodes)."""
    if num_numa_nodes <= 0:
        # Unknown/invalid topology (e.g. ``GetNumNUMANodes()`` returns -1) → cap at 1.
        cap = 1
    else:
        cap = max(1, int(num_numa_nodes))
    for d in range(min(cap, num_heads), 0, -1):
        if num_heads % d == 0:
            return d
    return 1


class TransformerLayer(nn.Module):
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--h-in", type=int, default=1152)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--mlp-hidden", type=int, default=6912)
    p.add_argument(
        "--num-tp-shards",
        type=int,
        default=None,
        help="NUMA TP world size; default: auto from heads vs NUMA node count.",
    )
    p.add_argument(
        "--rtol-ref-std",
        type=float,
        default=1e-4,
        help="torch.allclose rtol for eager vs standard compiled.",
    )
    p.add_argument(
        "--atol-ref-std",
        type=float,
        default=1e-5,
        help="torch.allclose atol for eager vs standard compiled.",
    )
    p.add_argument(
        "--rtol-std-numa",
        type=float,
        default=1e-4,
        help="torch.allclose rtol for standard compiled vs NUMA compiled.",
    )
    p.add_argument(
        "--atol-std-numa",
        type=float,
        default=1e-4,
        help="torch.allclose atol for standard compiled vs NUMA compiled.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Extra compiled forwards before collecting outputs for comparison.",
    )
    p.add_argument(
        "--skip-numa",
        action="store_true",
        help="Only compare eager vs standard compiled (no NUMA path).",
    )
    p.add_argument(
        "--max-autotune",
        action="store_true",
        help=(
            "Enable Inductor max_autotune (same as the demo scripts; noisy and slow). "
            "Default is off for faster, quieter correctness checks."
        ),
    )
    return p.parse_args()


def _max_err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a - b).abs()
    max_abs = float(d.max().item())
    denom = b.abs().clamp_min(1e-12)
    max_rel = float((d / denom).max().item())
    return max_abs, max_rel


def main() -> None:
    args = _parse_args()
    inductor_config.max_autotune = bool(args.max_autotune)

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    H_in = args.h_in
    H = args.num_heads
    E = args.head_dim
    H_qkv = H * E
    d_mlp = args.mlp_hidden

    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None or not hasattr(cpu_mod, "_set_numa_enabled"):
        print("ERROR: NUMA runtime hooks unavailable.", file=sys.stderr)
        sys.exit(1)

    # c10::GetNumNUMANodes() returns -1 when IsNUMAEnabled() is false (default flag off),
    # even if the machine has multiple NUMA nodes — enable first, then query.
    old_numa_enabled = bool(cpu_mod._is_numa_enabled())
    cpu_mod._set_numa_enabled(True)
    if not cpu_mod._is_numa_enabled():
        print(
            "ERROR: could not enable NUMA runtime (needs build with NUMA / libnuma). "
            "See c10/util/numa.cpp and FLAGS_caffe2_cpu_numa_enabled.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        n_nodes = int(cpu_mod._num_numa_nodes())
        if args.num_tp_shards is None:
            num_tp_shards = _auto_num_tp_shards(H, n_nodes)
        else:
            num_tp_shards = args.num_tp_shards
            if n_nodes >= 0 and n_nodes < num_tp_shards:
                print(
                    f"ERROR: need at least {num_tp_shards} NUMA nodes, got {n_nodes}.",
                    file=sys.stderr,
                )
                sys.exit(1)

        if (
            H % num_tp_shards != 0
            or H_qkv % num_tp_shards != 0
            or d_mlp % num_tp_shards != 0
        ):
            print(
                "ERROR: num_heads, H_qkv, and mlp_hidden must be divisible by "
                f"num_tp_shards={num_tp_shards}.",
                file=sys.stderr,
            )
            sys.exit(1)

        _run_checks(args, device, H, E, H_in, H_qkv, d_mlp, num_tp_shards, n_nodes)
    finally:
        cpu_mod._set_numa_enabled(old_numa_enabled)


def _run_checks(
    args: argparse.Namespace,
    device: torch.device,
    H: int,
    E: int,
    H_in: int,
    H_qkv: int,
    d_mlp: int,
    num_tp_shards: int,
    n_nodes: int,
) -> None:
    B, S = args.batch, args.seq

    base = SimpleLmStack(
        args.num_layers, H, E, H_in, d_mlp,
    ).to(device)
    base.set_batch_seq(B, S)
    for p in base.parameters():
        nn.init.normal_(p, std=0.02)

    x = torch.randn(B * S, H_in, device=device)

    with torch.no_grad():
        y_ref = base(x)

    m_std = copy.deepcopy(base)
    m_numa = copy.deepcopy(base)

    with torch.no_grad():
        with inductor_config.patch(
            {
                "_numa_attention_tp": False,
                "_numa_mlp_tp": False,
            }
        ):
            compiled_std = torch.compile(m_std)
            for _ in range(max(0, args.warmup)):
                _ = compiled_std(x)
            y_std = compiled_std(x)

    ok_ref_std = torch.allclose(
        y_ref, y_std, rtol=args.rtol_ref_std, atol=args.atol_ref_std
    )
    ma_rs, mr_rs = _max_err(y_ref, y_std)

    print(
        f"eager vs standard compiled: "
        f"{'PASS' if ok_ref_std else 'FAIL'} "
        f"(max_abs={ma_rs:.6g}, max_rel={mr_rs:.6g}, "
        f"rtol={args.rtol_ref_std}, atol={args.atol_ref_std})"
    )

    if not ok_ref_std:
        print("Stopping: fix eager vs standard compiled before NUMA comparison.", file=sys.stderr)
        sys.exit(1)

    if args.skip_numa:
        print("(--skip-numa) Skipping NUMA path.")
        sys.exit(0)

    with torch.no_grad():
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
            compiled_numa = torch.compile(m_numa)
            for _ in range(max(0, args.warmup)):
                _ = compiled_numa(x)
            y_numa = compiled_numa(x)

    ok_std_numa = torch.allclose(
        y_std, y_numa, rtol=args.rtol_std_numa, atol=args.atol_std_numa
    )
    ma_sn, mr_sn = _max_err(y_std, y_numa)

    print(
        f"standard compiled vs NUMA compiled: "
        f"{'PASS' if ok_std_numa else 'FAIL'} "
        f"(max_abs={ma_sn:.6g}, max_rel={mr_sn:.6g}, "
        f"rtol={args.rtol_std_numa}, atol={args.atol_std_numa})"
    )

    ok_ref_numa = torch.allclose(
        y_ref, y_numa, rtol=args.rtol_ref_std, atol=args.atol_ref_std
    )
    ma_rn, mr_rn = _max_err(y_ref, y_numa)
    print(
        f"eager vs NUMA compiled: "
        f"{'PASS' if ok_ref_numa else 'FAIL'} "
        f"(max_abs={ma_rn:.6g}, max_rel={mr_rn:.6g}, "
        f"rtol={args.rtol_ref_std}, atol={args.atol_ref_std})"
    )

    if not torch.isfinite(y_numa).all():
        print("WARNING: NUMA output has NaN/Inf.", file=sys.stderr)

    print(
        f"Config: num_tp_shards={num_tp_shards}, "
        f"_num_numa_nodes()={n_nodes}, "
        f"shape={tuple(y_ref.shape)}"
    )

    if n_nodes <= 0:
        print(
            "NOTE: _num_numa_nodes() is not a positive count (unknown topology in this "
            "environment). Auto shard count falls back to 1; use a NUMA-capable host or "
            "`--num-tp-shards N` with N>=2 to exercise the FX NUMA passes when supported.",
            file=sys.stderr,
        )
    if num_tp_shards < 2:
        print(
            "NOTE: num_tp_shards < 2 — the NUMA MLP/attention FX passes skip rewriting, "
            "and Inductor emits the same graph as the non-NUMA path. "
            "standard vs NUMA matching exactly (e.g. max_abs=0) is expected here and "
            "does not validate multi-socket NUMA TP codegen.",
            file=sys.stderr,
        )

    sys.exit(0 if (ok_std_numa and ok_ref_numa) else 1)


if __name__ == "__main__":
    main()
