from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .. import config

aten = torch.ops.aten

# ``torch.fx.symbolic_trace`` may record ``torch.mm`` / ``torch.addmm`` *or* the
# ``torch.Tensor`` method objects (``<built-in method addmm of type object>``);
# post-grad Inductor graphs typically use ``aten.*.default``. Match all of these.
_MM_TARGETS = frozenset(
    {aten.mm.default, torch.mm, torch.Tensor.mm}
)
_ADDMM_TARGETS = frozenset(
    {aten.addmm.default, torch.addmm, torch.Tensor.addmm}
)
_gelu: set[Any] = {aten.gelu.default, torch.nn.functional.gelu}
if hasattr(torch, "gelu"):
    _gelu.add(torch.gelu)
if hasattr(torch.Tensor, "gelu"):
    _gelu.add(torch.Tensor.gelu)
_cnn = getattr(torch._C, "_nn", None)
if _cnn is not None and hasattr(_cnn, "gelu"):
    _gelu.add(_cnn.gelu)
_GELU_TARGETS = frozenset(_gelu)

# Decomposed tanh-approx ``aten.gelu`` / ``F.gelu``: ``0.5 * x * (1 + erf(x / sqrt(2)))``
_GELU_TANH_HALF = 0.5
_GELU_TANH_INV_SQRT2 = 0.7071067811865476


def _target_is_mm(t: Any) -> bool:
    """Match ``mm`` including distinct ``torch.Tensor.mm`` bindings across builds."""
    return t in _MM_TARGETS or getattr(t, "__name__", None) == "mm"


def _target_is_addmm(t: Any) -> bool:
    return t in _ADDMM_TARGETS or getattr(t, "__name__", None) == "addmm"


def _target_is_gelu(t: Any) -> bool:
    return t in _GELU_TARGETS or getattr(t, "__name__", None) == "gelu"


def _debug_enabled() -> bool:
    return bool(getattr(config, "_numa_mlp_tp_debug", False))


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[numa_mlp_tp] {msg}", flush=True)


def _debug_graph(graph: torch.fx.Graph, title: str) -> None:
    if not _debug_enabled():
        return
    print(f"[numa_mlp_tp] --- {title} ---", flush=True)
    graph.print_tabular()


def _node_sig(n: torch.fx.Node) -> str:
    t = getattr(n, "target", n)
    name = getattr(t, "__name__", str(t))
    return f"{n.name}: {name}"


def _is_linear_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    t = node.target
    return _target_is_mm(t) or _target_is_addmm(t)


def _is_gelu_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and _target_is_gelu(node.target)


def _target_is_mul_tensor(t: Any) -> bool:
    return t is aten.mul.Tensor or getattr(t, "__name__", None) == "mul"


def _fx_float(x: Any) -> Optional[float]:
    if isinstance(x, torch.fx.Node):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mul_tensor_tensor_and_scalar(
    mul_node: torch.fx.Node, tensor_node: torch.fx.Node
) -> Optional[float]:
    """If ``mul_node`` is ``tensor * scalar`` or ``scalar * tensor``, return scalar."""
    if mul_node.op != "call_function" or not _target_is_mul_tensor(mul_node.target):
        return None
    if len(mul_node.args) < 2:
        return None
    a0, a1 = mul_node.args[0], mul_node.args[1]
    if a0 is tensor_node:
        return _fx_float(a1)
    if a1 is tensor_node:
        return _fx_float(a0)
    return None


def _add_tensor_plus_one(add_node: torch.fx.Node, erf_node: torch.fx.Node) -> bool:
    """``add.Tensor(erf, 1)`` or ``add.Tensor(1, erf)``."""
    if add_node.op != "call_function" or add_node.target is not aten.add.Tensor:
        return False
    if len(add_node.args) < 2:
        return False
    a0, a1 = add_node.args[0], add_node.args[1]
    if a0 is erf_node and _fx_float(a1) == 1.0:
        return True
    if a1 is erf_node and _fx_float(a0) == 1.0:
        return True
    return False


def _match_decomposed_tanh_gelu(
    linear1: torch.fx.Node,
) -> Optional[Tuple[torch.fx.Node, List[torch.fx.Node]]]:
    """
    Match post-decomposition GELU (e.g. under ``torch.compile``): two ``mul.Tensor``
    from ``linear1``, ``erf``, ``add.Tensor`` (+1), final ``mul.Tensor``.

    Returns ``(gelu_output_node, ordered_chain_for_shape_meta)``.
    """
    # AOTAutograd joint forward+backward saves ``linear1`` as a backward
    # activation, so ``linear1.users`` may include extra consumers beyond the
    # two decomposed GELU muls. Accept any user set that contains exactly one
    # ``* 0.5`` and one ``* 1/sqrt(2)`` scalar-mul of ``linear1``; non-mul or
    # other-constant users are tolerated (they don't invalidate the pattern).
    mul_half: Optional[torch.fx.Node] = None
    mul_inv: Optional[torch.fx.Node] = None
    for u in linear1.users:
        c = _mul_tensor_tensor_and_scalar(u, linear1)
        if c is None:
            continue
        if math.isclose(c, _GELU_TANH_HALF, rel_tol=1e-5, abs_tol=1e-6):
            if mul_half is not None:
                return None
            mul_half = u
        elif math.isclose(c, _GELU_TANH_INV_SQRT2, rel_tol=1e-4, abs_tol=1e-5):
            if mul_inv is not None:
                return None
            mul_inv = u
    if mul_half is None or mul_inv is None:
        return None

    # AOTAutograd may save intermediate nodes (mul_inv, erf, add, mul_half) as
    # backward activations, giving them extra users beyond the next forward
    # consumer. Match by finding the expected downstream op among users, not
    # by requiring a single user.
    erf_n: Optional[torch.fx.Node] = None
    for u in mul_inv.users:
        if (
            u.op == "call_function"
            and u.target is aten.erf.default
            and len(u.args) >= 1
            and u.args[0] is mul_inv
        ):
            erf_n = u
            break
    if erf_n is None:
        return None

    add_n: Optional[torch.fx.Node] = None
    for u in erf_n.users:
        if _add_tensor_plus_one(u, erf_n):
            add_n = u
            break
    if add_n is None:
        return None

    mul_out: Optional[torch.fx.Node] = None
    for u in add_n.users:
        if u.op != "call_function" or not _target_is_mul_tensor(u.target):
            continue
        if len(u.args) < 2:
            continue
        m0, m1 = u.args[0], u.args[1]
        if (m0 is mul_half and m1 is add_n) or (m0 is add_n and m1 is mul_half):
            mul_out = u
            break
    if mul_out is None:
        return None

    if mul_out not in mul_half.users:
        return None

    chain = [mul_half, mul_inv, erf_n, add_n, mul_out]
    return mul_out, chain


def _gelu_region_after_linear1(
    linear1: torch.fx.Node,
) -> Optional[Tuple[torch.fx.Node, List[torch.fx.Node]]]:
    """
    Single ``gelu``/``aten.gelu`` op, or decomposed tanh GELU subgraph.

    Returns ``(activation_output_node, nodes_to_refresh_shapes)``.
    """
    gelu_users = [u for u in linear1.users if _is_gelu_node(u)]
    if len(gelu_users) == 1:
        g = gelu_users[0]
        return g, [g]
    if len(gelu_users) > 1:
        return None
    return _match_decomposed_tanh_gelu(linear1)


def _extract_linear_parts(
    node: torch.fx.Node,
) -> Optional[Tuple[Optional[torch.fx.Node], torch.fx.Node, torch.fx.Node]]:
    if _target_is_mm(node.target):
        mat1, mat2 = node.args
        if not isinstance(mat1, torch.fx.Node) or not isinstance(mat2, torch.fx.Node):
            return None
        return None, mat1, mat2
    if _target_is_addmm(node.target):
        bias, mat1, mat2 = node.args
        if (
            not isinstance(bias, torch.fx.Node)
            or not isinstance(mat1, torch.fx.Node)
            or not isinstance(mat2, torch.fx.Node)
        ):
            return None
        return bias, mat1, mat2
    return None


def _shape_dim_to_int(x: Any) -> Optional[int]:
    if isinstance(x, int):
        return x
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _get_single_tensor_metadata(node: torch.fx.Node) -> Optional[Any]:
    """ShapeProp stores ``tensor_meta`` (TensorMetadata); Inductor often uses ``meta['val']``."""
    tm = node.meta.get("tensor_meta")
    if tm is None:
        return None
    from torch.fx.passes.shape_prop import TensorMetadata

    return tm if isinstance(tm, TensorMetadata) else None


def _shape_2d(node: torch.fx.Node) -> Optional[Tuple[int, int]]:
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
    i0 = _shape_dim_to_int(shape[0])
    i1 = _shape_dim_to_int(shape[1])
    if i0 is None or i1 is None:
        return None
    return i0, i1


def _bias_1d_length(node: torch.fx.Node) -> Optional[int]:
    """Length of a 1D bias from ``val`` or ``tensor_meta``."""
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor) and val.dim() == 1:
        return int(val.numel())
    tm = _get_single_tensor_metadata(node)
    if tm is not None and len(tm.shape) == 1:
        return int(tm.shape[0])
    return None


def _get_numa_world_size() -> int:
    override = int(getattr(config, "_numa_mlp_tp_world_size", 0))
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


def _get_compute_shard_id(world_size: int) -> int:
    shard_id = int(getattr(config, "_numa_mlp_tp_compute_shard", 0))
    if world_size <= 0:
        return 0
    if shard_id < 0:
        return 0
    if shard_id >= world_size:
        return world_size - 1
    return shard_id


def _set_numa_meta(
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


def _clone_meta(node: torch.fx.Node, src: torch.fx.Node) -> None:
    node.meta.update(src.meta)


def _resize_node_val_and_tensor_meta(
    node: torch.fx.Node, new_shape: Tuple[int, ...]
) -> None:
    """
    After rewiring matmul operands, FakeTensor shape in node.meta['val'] is stale.
    Refresh val + tensor_meta so activation and downstream passes see [s, d/N].
    """
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        new_val = val.new_empty(torch.Size(new_shape))
    else:
        tm = _get_single_tensor_metadata(node)
        if tm is None:
            return
        new_val = torch.empty(
            torch.Size(new_shape), dtype=tm.dtype, device="cpu"
        )
    node.meta["val"] = new_val
    try:
        from torch.fx.passes.shape_prop import _extract_tensor_metadata

        node.meta["tensor_meta"] = _extract_tensor_metadata(new_val)
    except Exception:
        pass


def _find_mlp_triples(
    graph: torch.fx.Graph,
) -> List[Tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, List[torch.fx.Node]]]:
    """
    Each tuple is ``(linear1, activation_out, linear2, gelu_chain)``.

    ``activation_out`` is the tensor feeding the second linear (``aten.gelu`` or the
    final ``mul`` of decomposed GELU). ``gelu_chain`` lists nodes whose 2D shape should
    match that activation after sharding (single-op ``[gelu]`` or decomposed subgraph).
    """
    triples = []
    for linear1 in graph.nodes:
        if not _is_linear_node(linear1):
            continue
        region = _gelu_region_after_linear1(linear1)
        if region is None:
            continue
        act, gelu_chain = region
        linear2_users = [
            u
            for u in act.users
            if _is_linear_node(u)
            and (
                (_target_is_mm(u.target) and len(u.args) >= 1 and u.args[0] is act)
                or (
                    _target_is_addmm(u.target)
                    and len(u.args) >= 2
                    and u.args[1] is act
                )
            )
        ]
        if len(linear2_users) != 1:
            continue
        linear2 = linear2_users[0]
        triples.append((linear1, act, linear2, gelu_chain))
    return triples


def numa_mlp_tp_pass(graph: torch.fx.Graph) -> None:
    """
    Rewrites linear->gelu->linear dimensions for NUMA tensor-parallel execution.

    Instead of inserting clone nodes into the FX graph, this pass only:
    1. Inserts ``aten.slice.Tensor`` nodes for column/row sharding of weight matrices
    2. Rewires existing linear nodes to use sliced inputs directly
    3. Annotates all nodes with ``numa_shard_plan`` metadata

    NUMA-aware buffer allocation (clone with ``numa_node``) and thread binding
    happen downstream in the Inductor wrapper codegen and Triton CPU launcher.

    The ``numa_shard_plan`` metadata on graph nodes describes per-tensor shard
    layout so the wrapper codegen can emit NUMA clone calls for all N shards.
    """
    world_size = _get_numa_world_size()
    _debug(
        f"pass start: world_size={world_size} "
        f"(override={getattr(config, '_numa_mlp_tp_world_size', 0)})"
    )
    if world_size < 2:
        _debug(f"skip: world_size={world_size} < 2")
        return

    plan_counter = itertools.count()
    changed = False
    for linear1, act, linear2, gelu_chain in _find_mlp_triples(graph):
        parts1 = _extract_linear_parts(linear1)
        parts2 = _extract_linear_parts(linear2)
        if parts1 is None or parts2 is None:
            continue
        b1, x, a = parts1
        b2, y_in, b = parts2
        if y_in is not act:
            continue

        x_shape = _shape_2d(x)
        a_shape = _shape_2d(a)
        b_shape = _shape_2d(b)
        if x_shape is None or a_shape is None or b_shape is None:
            continue
        s, h = x_shape
        ah, d = a_shape
        bd, hout = b_shape
        if h != ah or d != bd:
            continue
        if d % world_size != 0:
            continue

        if b1 is not None:
            b1_shape = _shape_2d(b1)
            if b1_shape is not None:
                continue
            if _bias_1d_length(b1) != d:
                continue
        if b2 is not None:
            b2_shape = _shape_2d(b2)
            if b2_shape is not None:
                continue
            if _bias_1d_length(b2) != hout:
                continue

        plan_id = next(plan_counter)
        shard_d = d // world_size
        shard_id = _get_compute_shard_id(world_size)
        _debug(
            f"match plan_id={plan_id}: "
            f"{_node_sig(linear1)} -> {_node_sig(act)} -> {_node_sig(linear2)} | "
            f"x=[{s},{h}] A=[{ah},{d}] B=[{bd},{hout}] | "
            f"shard_d={shard_d} compute_shard={shard_id}/{world_size}"
        )

        # Build per-shard slice nodes for A, B, b1 (no clones — just slices).
        # Only insert the slice for the compute shard; other shards are described
        # in metadata for the wrapper codegen to allocate at codegen time.
        a_slices: List[torch.fx.Node] = []
        b_slices: List[torch.fx.Node] = []
        b1_slices: List[torch.fx.Node] = []

        b1_slice_dim = 0
        with graph.inserting_before(linear1):
            for sid in range(world_size):
                start = sid * shard_d
                end = (sid + 1) * shard_d

                a_slice = graph.call_function(
                    aten.slice.Tensor,
                    args=(a, 1, start, end, 1),
                )
                # ``SliceView.create`` derives the sharded shape and the
                # parent-tensor strides directly from the slice-op args, so
                # Inductor's MM lowering already sees N = shard_d with
                # stride_bk = parent row stride. The slice node's meta is
                # only a fallback for shape-prop consumers; keep it as a
                # plain clone of the source meta to avoid surprising any
                # downstream FX pass that reads it.
                _clone_meta(a_slice, a)
                _set_numa_meta(
                    a_slice,
                    plan_id=plan_id,
                    shard_id=sid,
                    total_shards=world_size,
                    role="a_slice",
                    target_node=sid,
                )
                a_slices.append(a_slice)

                b_slice = graph.call_function(
                    aten.slice.Tensor,
                    args=(b, 0, start, end, 1),
                )
                _clone_meta(b_slice, b)
                _set_numa_meta(
                    b_slice,
                    plan_id=plan_id,
                    shard_id=sid,
                    total_shards=world_size,
                    role="b_slice",
                    target_node=sid,
                )
                b_slices.append(b_slice)

                if b1 is not None:
                    b1_val = b1.meta.get("val")
                    b1_slice_dim = (
                        1
                        if isinstance(b1_val, torch.Tensor) and b1_val.dim() > 1
                        else 0
                    )
                    b1_slice = graph.call_function(
                        aten.slice.Tensor,
                        args=(b1, b1_slice_dim, start, end, 1),
                    )
                    _clone_meta(b1_slice, b1)
                    _set_numa_meta(
                        b1_slice,
                        plan_id=plan_id,
                        shard_id=sid,
                        total_shards=world_size,
                        role="b1_slice",
                        target_node=sid,
                    )
                    b1_slices.append(b1_slice)

        _debug(
            f"plan_id={plan_id}: inserted per-shard slices "
            f"(a/b{f'/b1' if b1 is not None else ''} x {world_size})"
        )

        # Annotate input X with NUMA metadata (replicated to all nodes, no slice needed).
        _set_numa_meta(
            x,
            plan_id=plan_id,
            shard_id=0,
            total_shards=world_size,
            role="x_replica",
            target_node=0,
        )

        # Build the numa_shard_plan describing the full shard layout for wrapper codegen.
        shard_plan: Dict[str, Any] = {
            "plan_id": plan_id,
            "world_size": world_size,
            "shard_d": shard_d,
            "compute_shard": shard_id,
            "x_node": x.name,
            "a_node": a.name,
            "b_node": b.name,
            "b1_node": b1.name if b1 is not None else None,
            "b2_node": b2.name if b2 is not None else None,
            "a_slice_dim": 1,
            "b_slice_dim": 0,
            "b1_slice_dim": b1_slice_dim,
            "s": s,
            "h": h,
            "d": d,
            "hout": hout,
        }
        # Persist latest full plan so wrapper codegen can recover semantics even
        # if a backend path drops direct origin-node metadata.
        setattr(config, "_numa_mlp_tp_shard_plan", dict(shard_plan))

        # Propagate the shard plan to all per-shard slice nodes so downstream
        # passes and wrapper codegen can recover the layout from any of them.
        for slice_node in itertools.chain(a_slices, b_slices, b1_slices):
            slice_node.meta["numa_shard_plan"] = shard_plan

        # Rewire existing linear1 to use compute-shard sliced operands directly.
        a_shard = a_slices[shard_id]
        if _target_is_mm(linear1.target):
            linear1.args = (x, a_shard)
            _debug(
                f"plan_id={plan_id}: rewire linear1 {_node_sig(linear1)} "
                f"-> mm({_node_sig(x)}, {_node_sig(a_shard)})"
            )
        else:
            assert _target_is_addmm(linear1.target)
            assert b1 is not None
            b1_shard = b1_slices[shard_id]
            linear1.args = (b1_shard, x, a_shard)
            _debug(
                f"plan_id={plan_id}: rewire linear1 {_node_sig(linear1)} "
                f"-> addmm({_node_sig(b1_shard)}, "
                f"{_node_sig(x)}, {_node_sig(a_shard)})"
            )
        _set_numa_meta(
            linear1,
            plan_id=plan_id,
            shard_id=shard_id,
            total_shards=world_size,
            role="l1_shard_out",
            target_node=shard_id,
        )
        linear1.meta["numa_shard_plan"] = shard_plan
        _resize_node_val_and_tensor_meta(linear1, (s, shard_d))

        # Keep existing activation node; only annotate shard ownership.
        _set_numa_meta(
            act,
            plan_id=plan_id,
            shard_id=shard_id,
            total_shards=world_size,
            role="y_shard",
            target_node=shard_id,
        )
        act.meta["numa_shard_plan"] = shard_plan
        for n in gelu_chain:
            _resize_node_val_and_tensor_meta(n, (s, shard_d))
            n.meta["numa_shard_plan"] = shard_plan
        _debug(
            f"plan_id={plan_id}: refreshed meta linear1 and {len(gelu_chain)} activation "
            f"node(s) val shape ({s}, {shard_d}) for sharded activation I/O"
        )

        # Rewire existing linear2 to shard B operand directly.
        b_shard = b_slices[shard_id]
        if _target_is_mm(linear2.target):
            linear2.args = (act, b_shard)
            _debug(
                f"plan_id={plan_id}: rewire linear2 {_node_sig(linear2)} "
                f"-> mm({_node_sig(act)}, {_node_sig(b_shard)})"
            )
        else:
            assert _target_is_addmm(linear2.target)
            assert b2 is not None
            linear2.args = (b2, act, b_shard)
            _debug(
                f"plan_id={plan_id}: rewire linear2 {_node_sig(linear2)} "
                f"-> addmm(b2, {_node_sig(act)}, {_node_sig(b_shard)})"
            )
        _set_numa_meta(
            linear2,
            plan_id=plan_id,
            shard_id=shard_id,
            total_shards=world_size,
            role="z_shard",
            target_node=shard_id,
        )
        linear2.meta["numa_shard_plan"] = shard_plan
        changed = True

    if changed:
        graph.lint()
        _debug_graph(graph, "FX graph (after pass)")
        _debug("pass done: graph updated")
    else:
        _debug("pass done: no graph changes")
