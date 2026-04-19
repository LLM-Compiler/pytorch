"""
Shared helpers for NUMA tensor-parallel FX passes (MLP and attention).

The passes produce ``numa_shard_plan`` dicts that the Inductor wrapper codegen
consumes to emit per-NUMA-node buffer clones and pass them as ``numa_args`` to
the Triton CPU launcher. The shared schema is a generalization of the original
MLP pass layout so both MLP and attention blocks can drive the same codegen
path in ``torch/_inductor/codegen/wrapper.py``.

Generalized shard plan keys (all optional; populated by the pass):
    - ``kind``: ``"mlp"`` | ``"attention"`` (informational)
    - ``plan_id``, ``world_size``, ``shard_d``, ``compute_shard``
    - ``x_node``: name of the replicated activation input tensor
    - ``column_parallel_nodes``: ``List[Tuple[str, int]]`` — ``(arg_name, slice_dim)``
      pairs for tensors sharded along the output dim (slice dim 1 for 2D weight,
      0 for 1D bias). The slice chunk size is always ``shard_d``.
    - ``row_parallel_nodes``: ``List[Tuple[str, int]]`` — ``(arg_name, slice_dim)``
      for tensors sharded along the input/reduction dim (dim 0 for 2D weight).
    - ``replicated_nodes``: ``List[str]`` — replicate the tensor on every socket.
    - ``zero_on_nonzero_nodes``: ``List[str]`` — for row-parallel biases: keep
      the full value on compute-shard 0, zero it on the rest, to avoid double-
      counting after the post-kernel all-reduce.

The original MLP plan also set ``a_node``/``b_node``/``b1_node``/``b2_node``
plus their slice dims; the wrapper keeps a back-compat path that synthesizes
the generalized lists from those legacy fields when the new lists are absent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .. import config

aten = torch.ops.aten


# ---------------------------------------------------------------------------
# World / shard helpers
# ---------------------------------------------------------------------------
def get_numa_world_size(
    world_size_cfg_key: str = "_numa_mlp_tp_world_size",
) -> int:
    """Return the current NUMA world size (socket / TP shard count).

    Reads ``config.<world_size_cfg_key>`` first (test/debug override), then
    falls back to the runtime NUMA hooks in ``torch._C._cpu``.
    """
    override = int(getattr(config, world_size_cfg_key, 0))
    if override > 0:
        return override

    cpu_mod = getattr(torch._C, "_cpu", None)
    if cpu_mod is None:
        return 1
    is_enabled = getattr(cpu_mod, "_is_numa_enabled", None)
    get_nodes = getattr(cpu_mod, "_num_numa_nodes", None)
    if callable(is_enabled) and callable(get_nodes) and is_enabled():
        nodes = int(get_nodes())
        return nodes if nodes > 0 else 1
    return 1


def get_compute_shard_id(
    world_size: int,
    compute_cfg_key: str = "_numa_mlp_tp_compute_shard",
) -> int:
    shard_id = int(getattr(config, compute_cfg_key, 0))
    if world_size <= 0:
        return 0
    if shard_id < 0:
        return 0
    if shard_id >= world_size:
        return world_size - 1
    return shard_id


# ---------------------------------------------------------------------------
# Metadata helpers shared by both passes
# ---------------------------------------------------------------------------
def set_numa_meta(
    node: torch.fx.Node,
    *,
    plan_id: int,
    shard_id: int,
    total_shards: int,
    role: str,
    target_node: int,
    isolated: bool = False,
) -> None:
    node.meta["numa_plan_id"] = plan_id
    node.meta["numa_shard_id"] = shard_id
    node.meta["numa_total_shards"] = total_shards
    node.meta["numa_role"] = role
    node.meta["numa_target_node"] = target_node
    node.meta["numa_isolated"] = isolated
    if role == "final_add":
        node.meta["numa_isolated_add"] = True
        node.meta["numa_reduce_role"] = "final_sum"


def clone_meta(dst: torch.fx.Node, src: torch.fx.Node) -> None:
    dst.meta.update(src.meta)


def _get_single_tensor_metadata(node: torch.fx.Node) -> Optional[Any]:
    tm = node.meta.get("tensor_meta")
    if tm is None:
        return None
    from torch.fx.passes.shape_prop import TensorMetadata

    return tm if isinstance(tm, TensorMetadata) else None


def resize_node_val_and_tensor_meta(
    node: torch.fx.Node, new_shape: Tuple[int, ...]
) -> None:
    """Rewrite ``node.meta['val']`` (FakeTensor) and ``tensor_meta`` to
    ``new_shape`` so downstream shape-prop / Inductor sees consistent shapes
    after the pass rewires operands."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        new_val = val.new_empty(torch.Size(new_shape))
    else:
        tm = _get_single_tensor_metadata(node)
        if tm is None:
            return
        new_val = torch.empty(torch.Size(new_shape), dtype=tm.dtype, device="cpu")
    node.meta["val"] = new_val
    try:
        from torch.fx.passes.shape_prop import _extract_tensor_metadata

        node.meta["tensor_meta"] = _extract_tensor_metadata(new_val)
    except Exception:
        pass


def shape_dim_to_int(x: Any) -> Optional[int]:
    if isinstance(x, int):
        return x
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def shape_2d(node: torch.fx.Node) -> Optional[Tuple[int, int]]:
    val = node.meta.get("val")
    shape: Optional[Tuple[Any, ...]] = None
    if isinstance(val, torch.Tensor) and val.dim() == 2:
        shape = tuple(val.shape)
    else:
        tm = _get_single_tensor_metadata(node)
        if tm is not None and len(tm.shape) == 2:
            shape = tuple(tm.shape)
    if shape is None:
        return None
    i0 = shape_dim_to_int(shape[0])
    i1 = shape_dim_to_int(shape[1])
    if i0 is None or i1 is None:
        return None
    return i0, i1


def shape_nd(node: torch.fx.Node) -> Optional[Tuple[int, ...]]:
    val = node.meta.get("val")
    shape: Optional[Tuple[Any, ...]] = None
    if isinstance(val, torch.Tensor):
        shape = tuple(val.shape)
    else:
        tm = _get_single_tensor_metadata(node)
        if tm is not None:
            shape = tuple(tm.shape)
    if shape is None:
        return None
    out: List[int] = []
    for s in shape:
        i = shape_dim_to_int(s)
        if i is None:
            return None
        out.append(i)
    return tuple(out)


def bias_1d_length(node: torch.fx.Node) -> Optional[int]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor) and val.dim() == 1:
        return int(val.numel())
    tm = _get_single_tensor_metadata(node)
    if tm is not None and len(tm.shape) == 1:
        return int(tm.shape[0])
    return None


# ---------------------------------------------------------------------------
# Generalized shard plan construction
# ---------------------------------------------------------------------------
def populate_legacy_mlp_fields(
    plan: Dict[str, Any],
    *,
    x_node: str,
    a_node: str,
    b_node: str,
    b1_node: Optional[str] = None,
    b2_node: Optional[str] = None,
    a_slice_dim: int = 1,
    b_slice_dim: int = 0,
    b1_slice_dim: int = 0,
) -> None:
    """Populate back-compat MLP legacy fields on a shard plan so wrapper code
    paths that still key off them keep working. Also populates the
    generalized lists so new code can drive entirely off those."""
    plan.setdefault("x_node", x_node)
    plan.setdefault("a_node", a_node)
    plan.setdefault("b_node", b_node)
    plan.setdefault("b1_node", b1_node)
    plan.setdefault("b2_node", b2_node)
    plan.setdefault("a_slice_dim", a_slice_dim)
    plan.setdefault("b_slice_dim", b_slice_dim)
    plan.setdefault("b1_slice_dim", b1_slice_dim)

    col: List[Tuple[str, int]] = list(plan.setdefault("column_parallel_nodes", []))
    col.append((a_node, a_slice_dim))
    if b1_node is not None:
        col.append((b1_node, b1_slice_dim))
    plan["column_parallel_nodes"] = col

    row: List[Tuple[str, int]] = list(plan.setdefault("row_parallel_nodes", []))
    row.append((b_node, b_slice_dim))
    plan["row_parallel_nodes"] = row

    rep: List[str] = list(plan.setdefault("replicated_nodes", []))
    if x_node not in rep:
        rep.append(x_node)
    if b2_node is not None and b2_node not in rep:
        rep.append(b2_node)
    plan["replicated_nodes"] = rep

    zn: List[str] = list(plan.setdefault("zero_on_nonzero_nodes", []))
    if b2_node is not None and b2_node not in zn:
        zn.append(b2_node)
    plan["zero_on_nonzero_nodes"] = zn


def ensure_generalized_lists(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``plan`` with ``column_parallel_nodes``,
    ``row_parallel_nodes``, ``replicated_nodes``, ``zero_on_nonzero_nodes``
    populated from legacy fields if missing. Used by the wrapper as a
    compatibility shim for plans emitted before the generalized schema."""
    if not isinstance(plan, dict):
        return plan
    # Early exit: plan already has generalized fields.
    if any(
        k in plan
        for k in (
            "column_parallel_nodes",
            "row_parallel_nodes",
            "replicated_nodes",
            "zero_on_nonzero_nodes",
        )
    ):
        # Fill missing lists with empty defaults.
        out = dict(plan)
        out.setdefault("column_parallel_nodes", [])
        out.setdefault("row_parallel_nodes", [])
        out.setdefault("replicated_nodes", [])
        out.setdefault("zero_on_nonzero_nodes", [])
        return out

    out = dict(plan)
    x_node = out.get("x_node")
    a_node = out.get("a_node")
    b_node = out.get("b_node")
    b1_node = out.get("b1_node")
    b2_node = out.get("b2_node")
    a_slice_dim = int(out.get("a_slice_dim", 1))
    b_slice_dim = int(out.get("b_slice_dim", 0))
    b1_slice_dim = int(out.get("b1_slice_dim", 0))

    col: List[Tuple[str, int]] = []
    if isinstance(a_node, str):
        col.append((a_node, a_slice_dim))
    if isinstance(b1_node, str):
        col.append((b1_node, b1_slice_dim))
    out["column_parallel_nodes"] = col

    row: List[Tuple[str, int]] = []
    if isinstance(b_node, str):
        row.append((b_node, b_slice_dim))
    out["row_parallel_nodes"] = row

    rep: List[str] = []
    if isinstance(x_node, str):
        rep.append(x_node)
    if isinstance(b2_node, str):
        rep.append(b2_node)
    out["replicated_nodes"] = rep

    zn: List[str] = []
    if isinstance(b2_node, str):
        zn.append(b2_node)
    out["zero_on_nonzero_nodes"] = zn

    return out
