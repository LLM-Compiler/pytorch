"""
NUMA tensor-parallel pass for an LM-head-style lone matmul.

Targets the final ``aten.mm`` / ``aten.addmm`` of a GPT-style decoder, which
is the model's largest matmul (``[B*S, D] @ [D, V]`` with V = vocab_size).
Without this pass the LM head runs single-threaded (or OMP) on the
reassembled hidden state and dominates end-to-end prefill time.

Strategy: **replicate the weight per socket, M-split the output.**

  - Each socket gets a full clone of the weight matrix (memory cost:
    ``V * D * dtype_size`` per socket — for GPT-2 117M fp32 that's
    ~154 MB per socket).
  - The launcher M-splits the kernel grid: socket ``i`` computes rows
    ``[i * M//W : (i+1) * M//W]`` of the output.
  - M-split reassembly fires at the end (the LM head output is the
    model output; no NUMA consumer downstream).

This is correctness-preserving and trivially-parallel (no communication
beyond reassembly). It does NOT require any launcher-side changes — the
existing M-split + replicated-weight path handles it via the same code
that drives e.g. layer_norm.

The pass is conservative: it only tags ``mm``/``addmm`` nodes that are
not already part of an attention/MLP/LN TP plan. After all the other
NUMA passes have run on a GPT-style graph, the only such node is the
LM head.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import torch

from .. import config
from .numa_tp_common import (
    get_compute_shard_id,
    get_numa_world_size,
    set_numa_meta,
)

aten = torch.ops.aten


def _debug_enabled() -> bool:
    return bool(getattr(config, "_numa_lm_head_tp_debug", False))


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[numa_lm_head_tp] {msg}", flush=True)


def _is_mm_or_addmm(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in (
        aten.mm.default,
        aten.addmm.default,
    )


def _extract_parts(
    node: torch.fx.Node,
) -> Optional[tuple]:
    """Return (bias_or_None, mat1, mat2) if the node is an mm/addmm with
    fx.Node operands, else None."""
    if node.target is aten.mm.default:
        if len(node.args) < 2:
            return None
        a, b = node.args[0], node.args[1]
        if not isinstance(a, torch.fx.Node) or not isinstance(b, torch.fx.Node):
            return None
        return None, a, b
    if node.target is aten.addmm.default:
        if len(node.args) < 3:
            return None
        bias, a, b = node.args[0], node.args[1], node.args[2]
        if (
            not isinstance(bias, torch.fx.Node)
            or not isinstance(a, torch.fx.Node)
            or not isinstance(b, torch.fx.Node)
        ):
            return None
        return bias, a, b
    return None


def _build_plan(
    *,
    plan_id: int,
    world_size: int,
    shard_id: int,
    x_node_name: str,
    weight_name: str,
    bias_name: str,
) -> Dict[str, Any]:
    replicated: List[str] = [x_node_name, weight_name]
    if bias_name:
        replicated.append(bias_name)
    return {
        "kind": "lm_head",
        "plan_id": plan_id,
        "world_size": world_size,
        "compute_shard": shard_id,
        "shard_d": 0,
        "x_node": x_node_name,
        "weight_node": weight_name,
        "bias_node": bias_name,
        "column_parallel_nodes": [],
        "row_parallel_nodes": [],
        "replicated_nodes": replicated,
        "zero_on_nonzero_nodes": [],
    }


def numa_lm_head_tp_pass(graph: torch.fx.Graph) -> None:
    """Tag any untagged ``mm``/``addmm`` nodes with an lm_head shard plan.

    Idempotent: skips nodes already carrying ``numa_shard_plan``.
    """
    if not bool(getattr(config, "_numa_lm_head_tp", False)):
        return

    world_size = get_numa_world_size("_numa_lm_head_tp_world_size")
    _debug(
        f"pass start: world_size={world_size} "
        f"(override={getattr(config, '_numa_lm_head_tp_world_size', 0)})"
    )
    if world_size < 2:
        _debug(f"skip: world_size={world_size} < 2")
        return

    shard_id = get_compute_shard_id(world_size, "_numa_lm_head_tp_compute_shard")
    plan_counter = itertools.count()
    matched = 0

    for node in graph.nodes:
        if not _is_mm_or_addmm(node):
            continue
        if node.meta.get("numa_shard_plan") is not None:
            continue
        parts = _extract_parts(node)
        if parts is None:
            continue
        bias, x_arg, w_arg = parts
        plan_id = next(plan_counter)
        plan = _build_plan(
            plan_id=plan_id,
            world_size=world_size,
            shard_id=shard_id,
            x_node_name=x_arg.name,
            weight_name=w_arg.name,
            bias_name=bias.name if bias is not None else "",
        )
        node.meta["numa_shard_plan"] = plan
        set_numa_meta(
            node,
            plan_id=plan_id,
            shard_id=shard_id,
            total_shards=world_size,
            role="z_shard",
            target_node=shard_id,
        )
        matched += 1
        _debug(
            f"matched lm_head: {node.name}(x={x_arg.name}, w={w_arg.name}, "
            f"bias={bias.name if bias else None}) plan_id={plan_id}"
        )

    if matched:
        _debug(f"pass done: tagged {matched} lm_head node(s)")
    else:
        _debug("pass done: no untagged mm/addmm nodes")
