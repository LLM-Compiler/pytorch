"""
NUMA tensor-parallel pass for the residual add that follows an attention or
MLP TP region.

In a Pre-LN or Post-LN transformer block the row-parallel out_linear is
immediately followed by ``aten.add(Z, X)`` where ``Z`` is the TP region's
output and ``X`` is the same activation that entered the region (the
``x_node`` of the upstream shard plan).

Without this pass the residual add runs *outside* the TP region as an
ordinary Inductor kernel on the reassembled shared buffer of ``Z``. That
forces:

  1. The launcher to reassemble per-socket ``Z`` shards into a shared buffer
     (the deferred ``_NumaReassembleLine`` lines are not elided because the
     non-NUMA add references the shared buffer).
  2. The next TP region (next block's attention/MLP) to re-clone the shared
     residual output back to per-socket buffers.

This pass tags the residual ``aten.add`` with a ``numa_shard_plan`` of kind
``residual_add`` and ``numa_role="z_shard"``. Wrapper codegen then:

  - Looks up per-socket ``Z`` and ``X`` from ``_numa_tensor_cache`` (already
    populated by the upstream attention/MLP and the upstream X-replicate).
  - Allocates a per-socket output via ``empty_strided_cpu_numa``.
  - Emits the kernel call with ``numa_args=`` so the launcher M-splits the
    add the same way it M-splits every other NUMA kernel.
  - Treats the upstream ``Z``'s reassembly as elidable because the only
    consumer is now a NUMA kernel call (line contains ``numa_args=``).

The plan carries empty ``column_parallel_nodes`` / ``row_parallel_nodes`` /
``replicated_nodes`` / ``zero_on_nonzero_nodes`` lists: both inputs are
satisfied entirely from ``_numa_tensor_cache`` so no slice/clone codegen
is needed.
"""
from __future__ import annotations

import operator
from typing import Any, Dict, Optional, Set

import torch

from .. import config
from .numa_tp_common import set_numa_meta

aten = torch.ops.aten


def _debug_enabled() -> bool:
    return bool(getattr(config, "_numa_residual_tp_debug", False))


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[numa_residual_tp] {msg}", flush=True)


def _is_add_tensor(n: torch.fx.Node) -> bool:
    return (
        n.op == "call_function"
        and n.target is aten.add.Tensor
    )


def _other_operand(add_node: torch.fx.Node, z_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Return the operand of ``add_node`` that is not ``z_node``, or None."""
    if len(add_node.args) < 2:
        return None
    a, b = add_node.args[0], add_node.args[1]
    if a is z_node and isinstance(b, torch.fx.Node):
        return b
    if b is z_node and isinstance(a, torch.fx.Node):
        return a
    return None


def _block_input_names(plan_x_node: torch.fx.Node) -> Set[str]:
    """Return the set of node names that may be the residual operand for a
    block whose attention/MLP plan input is ``plan_x_node``.

    Post-LN block: the residual operand IS ``plan_x_node`` itself.
    Pre-LN block:  ``plan_x_node`` is ``getitem(native_layer_norm(...), 0)``;
    the residual operand is the LayerNorm's input. Walk back through that
    one LN op so this pass works for both layouts.
    """
    candidates: Set[str] = {plan_x_node.name}
    # Pre-LN: plan_x_node = getitem(ln, 0) where ln = native_layer_norm.
    target = plan_x_node.target
    if (
        target is operator.getitem
        and len(plan_x_node.args) == 2
        and plan_x_node.args[1] == 0
        and isinstance(plan_x_node.args[0], torch.fx.Node)
    ):
        ln_node = plan_x_node.args[0]
        if (
            ln_node.op == "call_function"
            and ln_node.target is aten.native_layer_norm.default
            and len(ln_node.args) >= 1
            and isinstance(ln_node.args[0], torch.fx.Node)
        ):
            candidates.add(ln_node.args[0].name)
    return candidates


def _build_residual_plan(upstream_plan: Dict[str, Any], x_node_name: str) -> Dict[str, Any]:
    return {
        "kind": "residual_add",
        "plan_id": upstream_plan.get("plan_id", 0),
        "world_size": int(upstream_plan.get("world_size", 0)),
        "compute_shard": int(upstream_plan.get("compute_shard", 0)),
        "shard_d": int(upstream_plan.get("shard_d", 0)),
        "x_node": x_node_name,
        # Empty lists: every input is reused from _numa_tensor_cache; the
        # output is a per-socket fresh allocation. ensure_generalized_lists
        # treats this plan as already-normalized via the early-exit branch.
        "column_parallel_nodes": [],
        "row_parallel_nodes": [],
        "replicated_nodes": [],
        "zero_on_nonzero_nodes": [],
    }


def numa_residual_tp_pass(graph: torch.fx.Graph) -> None:
    """Tag residual ``aten.add(Z, X)`` adds that follow attention/MLP TP
    regions with a ``residual_add`` shard plan.

    Idempotent: an add already carrying ``numa_shard_plan`` is skipped.
    """
    if not bool(getattr(config, "_numa_residual_tp", False)):
        return
    _debug("pass start")

    # Collect candidate Z nodes (z_shard outputs of attention or MLP TP).
    # Iterate graph.nodes in topological order so we never depend on user-set
    # iteration order (per project convention).
    candidates = []
    for node in graph.nodes:
        plan = node.meta.get("numa_shard_plan")
        if plan is None:
            continue
        if node.meta.get("numa_role") != "z_shard":
            continue
        kind = plan.get("kind")
        if kind not in ("attention", "mlp"):
            continue
        candidates.append(node)

    if not candidates:
        _debug("pass done: no z_shard candidates")
        return

    # Build a name → node lookup so we can resolve the plan's stored
    # x_node name back to its FX node (needed to walk through Pre-LN).
    by_name = {n.name: n for n in graph.nodes}

    matched = 0
    for z_node in candidates:
        upstream_plan = z_node.meta["numa_shard_plan"]
        x_node_name = upstream_plan.get("x_node")
        if not isinstance(x_node_name, str):
            continue
        plan_x_node = by_name.get(x_node_name)
        if plan_x_node is None:
            continue
        accepted_names = _block_input_names(plan_x_node)
        # Scan z_node.users for an aten.add whose other operand is the block
        # input (or, in Pre-LN, the LayerNorm's input). Tolerate extra users
        # (e.g. AOTAutograd save-for-backward).
        for user in list(z_node.users):
            if not _is_add_tensor(user):
                continue
            if user.meta.get("numa_shard_plan") is not None:
                continue
            other = _other_operand(user, z_node)
            if other is None or other.name not in accepted_names:
                continue
            res_plan = _build_residual_plan(upstream_plan, other.name)
            user.meta["numa_shard_plan"] = res_plan
            set_numa_meta(
                user,
                plan_id=res_plan["plan_id"],
                shard_id=res_plan["compute_shard"],
                total_shards=res_plan["world_size"],
                role="z_shard",
                target_node=res_plan["compute_shard"],
            )
            matched += 1
            _debug(
                f"matched residual: {user.name} = {z_node.name} + {other.name} "
                f"(plan_id={res_plan['plan_id']}, kind={upstream_plan.get('kind')})"
            )
            break  # one residual per Z

    if matched:
        _debug(f"pass done: tagged {matched} residual add(s)")
    else:
        _debug("pass done: no matching residual adds")
