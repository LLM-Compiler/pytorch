"""
NUMA tensor-parallel pass for LayerNorm.

LayerNorm normalizes along the last (hidden) dim, so each row can be
processed independently. Per-row parallelism (M-split) is therefore
correctness-preserving for LN — no cross-socket communication is needed.

Two input geometries are accepted by this pass:

  (a) **Replicated input** — e.g. the LN at the start of the very first
      transformer block, fed by the embedding sum. Each socket gets a full
      clone of x and computes LN on its M-shard.
  (b) **Row-shard input** — e.g. the LN inside a Pre-LN block whose input is
      the output of a residual_add (cached per-socket from the upstream TP
      region). Each socket already has full hidden for its rows; LN runs
      directly on the cached row-shard.

The pass simply tags ``aten.native_layer_norm.default`` nodes with a
``numa_shard_plan`` of kind ``"layer_norm"`` and ``numa_role="z_shard"``.
Wrapper codegen then:

  - Looks up x from ``_numa_tensor_cache`` (case b) or replicates it
    (case a) via the existing ``x_node`` path.
  - Treats weight/bias as replicated (cached per-socket clones).
  - Allocates per-socket output / mean / rstd via ``empty_strided_cpu_numa``.
  - Emits the kernel call with ``numa_args=`` so the launcher M-splits.
  - Defers M-split reassembly for outputs (elides when the LN output's
    only consumer is another NUMA kernel — which is the case inside any
    Pre-LN transformer block).

The output stays per-socket (row-shard) in the cache; downstream NUMA
kernels (the next attention's column-parallel QKV) read their M-shard
directly. The whole transformer block can thus flow socket-locally with
zero reassembly across LN→attn→residual→LN→MLP→residual boundaries.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List

import torch

from .. import config
from .numa_tp_common import (
    get_compute_shard_id,
    get_numa_world_size,
    set_numa_meta,
)

aten = torch.ops.aten


def _debug_enabled() -> bool:
    return bool(getattr(config, "_numa_layer_norm_tp_debug", False))


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[numa_layer_norm_tp] {msg}", flush=True)


def _build_plan(
    *,
    plan_id: int,
    world_size: int,
    shard_id: int,
    x_node_name: str,
    weight_name: str,
    bias_name: str,
) -> Dict[str, Any]:
    replicated: List[str] = [x_node_name]
    if weight_name:
        replicated.append(weight_name)
    if bias_name:
        replicated.append(bias_name)
    return {
        "kind": "layer_norm",
        "plan_id": plan_id,
        "world_size": world_size,
        "compute_shard": shard_id,
        "shard_d": 0,  # not used by codegen for this kind
        "x_node": x_node_name,
        "weight_node": weight_name,
        "bias_node": bias_name,
        "column_parallel_nodes": [],
        "row_parallel_nodes": [],
        "replicated_nodes": replicated,
        "zero_on_nonzero_nodes": [],
    }


def numa_layer_norm_tp_pass(graph: torch.fx.Graph) -> None:
    """Tag ``aten.native_layer_norm`` nodes with a layer_norm shard plan.

    Idempotent: a node already carrying ``numa_shard_plan`` is skipped.
    """
    if not bool(getattr(config, "_numa_layer_norm_tp", False)):
        return

    world_size = get_numa_world_size("_numa_layer_norm_tp_world_size")
    _debug(
        f"pass start: world_size={world_size} "
        f"(override={getattr(config, '_numa_layer_norm_tp_world_size', 0)})"
    )
    if world_size < 2:
        _debug(f"skip: world_size={world_size} < 2")
        return

    shard_id = get_compute_shard_id(world_size, "_numa_layer_norm_tp_compute_shard")
    plan_counter = itertools.count()
    matched = 0

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is not aten.native_layer_norm.default:
            continue
        if node.meta.get("numa_shard_plan") is not None:
            continue
        # Args: (input, normalized_shape, weight, bias, eps)
        if len(node.args) < 5:
            continue
        x_arg = node.args[0]
        weight_arg = node.args[2]
        bias_arg = node.args[3]
        if not isinstance(x_arg, torch.fx.Node):
            continue
        weight_name = weight_arg.name if isinstance(weight_arg, torch.fx.Node) else ""
        bias_name = bias_arg.name if isinstance(bias_arg, torch.fx.Node) else ""

        plan_id = next(plan_counter)
        plan = _build_plan(
            plan_id=plan_id,
            world_size=world_size,
            shard_id=shard_id,
            x_node_name=x_arg.name,
            weight_name=weight_name,
            bias_name=bias_name,
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
            f"matched layer_norm: {node.name}(x={x_arg.name}, w={weight_name}, "
            f"b={bias_name}) plan_id={plan_id}"
        )

    if matched:
        _debug(f"pass done: tagged {matched} layer_norm node(s)")
    else:
        _debug("pass done: no layer_norm nodes")
