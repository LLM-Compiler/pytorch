from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .. import config
from .numa_tp_common import merge_global_shard_plan

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
    try:
        graph.print_tabular()
    except ImportError:
        for n in graph.nodes:
            tgt = getattr(n.target, "__name__", str(n.target))
            val = n.meta.get("val")
            shape = tuple(val.shape) if hasattr(val, "shape") else None
            print(f"  {n.op:14s} {n.name:24s} {tgt:32s} args={n.args}  shape={shape}", flush=True)


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


# ---------------------------------------------------------------------------
# GeGLU pattern (Gemma): mm(x, gate_w) and mm(x, up_w) sharing x via separate
# reshapes, tanh-approx GELU on gate, elem-mul with up, then mm(_, down_w).
#
# Post-grad shape (verified against torch.compile of F.linear + F.gelu(...,
# approximate="tanh")):
#
#   view      = reshape(x, [s, h])                      # x = (B, S, h)
#   permute   = permute(gate_w, [1, 0])                 # gate_w: (d, h) -> (h, d)
#   gate_mm   = mm(view, permute)                        # (s, d)
#   view_1    = reshape(gate_mm, [B, S, d])
#   mul_4     = view_1 * 0.5
#   mul       = view_1 * view_1
#   mul_1     = mul * view_1
#   mul_2     = mul_1 * 0.044715
#   add       = view_1 + mul_2
#   mul_3     = add * 0.7978845608028654
#   tanh      = tanh(mul_3)
#   add_1     = tanh + 1
#   mul_5     = mul_4 * add_1                            # gelu_tanh(gate)
#   view_2    = reshape(x, [s, h])
#   permute_1 = permute(up_w, [1, 0])
#   mm_1      = mm(view_2, permute_1)                    # (s, d)
#   view_3    = reshape(mm_1, [B, S, d])
#   mul_6     = mul_5 * view_3
#   view_4    = reshape(mul_6, [s, d])
#   permute_2 = permute(down_w, [1, 0])                  # down_w: (h, d) -> (d, h)
#   mm_2      = mm(view_4, permute_2)                    # (s, h)
#
# Sharding: shard_d = d // world_size.
#   - gate_w (placeholder, shape (d, h)): column-parallel, slice dim 0
#   - up_w   (placeholder, shape (d, h)): column-parallel, slice dim 0
#   - down_w (placeholder, shape (h, d)): row-parallel,    slice dim 1
# Per-shard rewire: insert ``slice -> permute`` and replace the mm operand.
# ---------------------------------------------------------------------------
_GELU_TANH_HALF_C = 0.5
_GELU_TANH_CUBE_C = 0.044715
_GELU_TANH_SQRT_2_OVER_PI = 0.7978845608028654


def _is_reshape(node: Any) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is aten.reshape.default
    )


def _is_permute(node: Any) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is aten.permute.default
    )


def _reshape_source(node: torch.fx.Node) -> torch.fx.Node:
    """If ``node`` is a reshape, return its source; else return ``node``."""
    if _is_reshape(node) and isinstance(node.args[0], torch.fx.Node):
        return node.args[0]  # type: ignore[return-value]
    return node


def _shape_last_dim(node: torch.fx.Node) -> Optional[int]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor) and val.dim() >= 1:
        return _shape_dim_to_int(val.shape[-1])
    tm = _get_single_tensor_metadata(node)
    if tm is not None and len(tm.shape) >= 1:
        return _shape_dim_to_int(tm.shape[-1])
    return None


def _match_tanh_approx_gelu(
    view1: torch.fx.Node,
) -> Optional[List[torch.fx.Node]]:
    """Match the 9-op tanh-approx GELU chain rooted at ``view1`` (the reshape
    that consumes the first mm output). Returns the ordered chain
    ``[mul_4, mul, mul_1, mul_2, add, mul_3, tanh, add_1, mul_5]``, or None.
    The final element is the GELU output (``mul_5``).
    """
    mul_half: Optional[torch.fx.Node] = None  # view1 * 0.5
    mul_sq: Optional[torch.fx.Node] = None    # view1 * view1
    for u in view1.users:
        if u.op != "call_function" or not _target_is_mul_tensor(u.target):
            continue
        if len(u.args) < 2:
            continue
        a0, a1 = u.args[0], u.args[1]
        c = _mul_tensor_tensor_and_scalar(u, view1)
        if c is not None and math.isclose(c, _GELU_TANH_HALF_C, rel_tol=1e-5, abs_tol=1e-6):
            if mul_half is None:
                mul_half = u
        elif a0 is view1 and a1 is view1:
            if mul_sq is None:
                mul_sq = u
    if mul_half is None or mul_sq is None:
        return None

    # mul_1 = mul_sq * view1 (or view1 * mul_sq)
    mul_cube: Optional[torch.fx.Node] = None
    for u in mul_sq.users:
        if u.op != "call_function" or not _target_is_mul_tensor(u.target):
            continue
        if len(u.args) < 2:
            continue
        a0, a1 = u.args[0], u.args[1]
        if (a0 is mul_sq and a1 is view1) or (a0 is view1 and a1 is mul_sq):
            mul_cube = u
            break
    if mul_cube is None:
        return None

    # mul_2 = mul_cube * 0.044715
    mul_044: Optional[torch.fx.Node] = None
    for u in mul_cube.users:
        c = _mul_tensor_tensor_and_scalar(u, mul_cube)
        if c is not None and math.isclose(c, _GELU_TANH_CUBE_C, rel_tol=1e-4, abs_tol=1e-6):
            mul_044 = u
            break
    if mul_044 is None:
        return None

    # add = view1 + mul_044 (or mul_044 + view1)
    add_node: Optional[torch.fx.Node] = None
    for u in mul_044.users:
        if u.op != "call_function" or u.target is not aten.add.Tensor:
            continue
        if len(u.args) < 2:
            continue
        a0, a1 = u.args[0], u.args[1]
        if (a0 is view1 and a1 is mul_044) or (a0 is mul_044 and a1 is view1):
            add_node = u
            break
    if add_node is None:
        return None

    # mul_3 = add * sqrt(2/pi)
    mul_sqrt: Optional[torch.fx.Node] = None
    for u in add_node.users:
        c = _mul_tensor_tensor_and_scalar(u, add_node)
        if c is not None and math.isclose(
            c, _GELU_TANH_SQRT_2_OVER_PI, rel_tol=1e-5, abs_tol=1e-6
        ):
            mul_sqrt = u
            break
    if mul_sqrt is None:
        return None

    # tanh
    tanh_node: Optional[torch.fx.Node] = None
    for u in mul_sqrt.users:
        if u.op == "call_function" and u.target is aten.tanh.default:
            tanh_node = u
            break
    if tanh_node is None:
        return None

    # add_1 = tanh + 1
    add1_node: Optional[torch.fx.Node] = None
    for u in tanh_node.users:
        if u.op != "call_function" or u.target is not aten.add.Tensor:
            continue
        if len(u.args) < 2:
            continue
        a0, a1 = u.args[0], u.args[1]
        if (a0 is tanh_node and _fx_float(a1) == 1.0) or (
            a1 is tanh_node and _fx_float(a0) == 1.0
        ):
            add1_node = u
            break
    if add1_node is None:
        return None

    # mul_5 = mul_half * add_1
    mul_out: Optional[torch.fx.Node] = None
    for u in add1_node.users:
        if u.op != "call_function" or not _target_is_mul_tensor(u.target):
            continue
        if len(u.args) < 2:
            continue
        m0, m1 = u.args[0], u.args[1]
        if (m0 is mul_half and m1 is add1_node) or (m0 is add1_node and m1 is mul_half):
            mul_out = u
            break
    if mul_out is None:
        return None

    return [mul_half, mul_sq, mul_cube, mul_044, add_node, mul_sqrt, tanh_node, add1_node, mul_out]


def _find_geglu_quintuples(graph: torch.fx.Graph) -> List[Dict[str, Any]]:
    """Find Gemma-style GeGLU MLP blocks in the post-grad graph.

    Returns a list of dicts with keys:
      gate_mm, up_mm, down_mm, gate_view (reshape after gate_mm),
      up_view (reshape after up_mm), mul_node (gelu*up), down_view
      (reshape feeding down_mm), gelu_chain (list of 9 nodes).
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for gate_mm in list(graph.nodes):
        if gate_mm in seen:
            continue
        if gate_mm.op != "call_function" or not _target_is_mm(gate_mm.target):
            continue
        if len(gate_mm.args) < 2:
            continue
        view_x_g = gate_mm.args[0]
        gate_perm = gate_mm.args[1]
        if not isinstance(view_x_g, torch.fx.Node) or not isinstance(gate_perm, torch.fx.Node):
            continue
        if not _is_permute(gate_perm):
            continue

        # gate_mm -> view_1 (reshape)
        view_1 = None
        for u in gate_mm.users:
            if _is_reshape(u) and u.args[0] is gate_mm:
                view_1 = u
                break
        if view_1 is None:
            continue

        chain = _match_tanh_approx_gelu(view_1)
        if chain is None:
            continue
        gelu_out = chain[-1]

        # gelu_out * view_3 (a reshape of an mm output)
        mul_node = None
        view_3 = None
        for u in gelu_out.users:
            if u.op != "call_function" or not _target_is_mul_tensor(u.target):
                continue
            if len(u.args) < 2:
                continue
            a0, a1 = u.args[0], u.args[1]
            other = a1 if a0 is gelu_out else (a0 if a1 is gelu_out else None)
            if not isinstance(other, torch.fx.Node):
                continue
            if not _is_reshape(other):
                continue
            src = other.args[0]
            if (
                isinstance(src, torch.fx.Node)
                and src.op == "call_function"
                and _target_is_mm(src.target)
            ):
                mul_node = u
                view_3 = other
                break
        if mul_node is None or view_3 is None:
            continue
        up_mm = view_3.args[0]
        if not isinstance(up_mm, torch.fx.Node):
            continue
        if len(up_mm.args) < 2:
            continue
        view_x_u = up_mm.args[0]
        up_perm = up_mm.args[1]
        if not isinstance(view_x_u, torch.fx.Node) or not isinstance(up_perm, torch.fx.Node):
            continue
        if not _is_permute(up_perm):
            continue

        # gate_mm and up_mm must read reshapes of the same source x.
        x_src_g = _reshape_source(view_x_g)
        x_src_u = _reshape_source(view_x_u)
        if x_src_g is not x_src_u:
            continue

        # mul_node -> view_4 (reshape) -> down_mm
        view_4 = None
        for u in mul_node.users:
            if _is_reshape(u) and u.args[0] is mul_node:
                view_4 = u
                break
        if view_4 is None:
            continue
        down_mm = None
        for u in view_4.users:
            if (
                u.op == "call_function"
                and _target_is_mm(u.target)
                and len(u.args) >= 2
                and u.args[0] is view_4
            ):
                down_mm = u
                break
        if down_mm is None:
            continue
        down_perm = down_mm.args[1]
        if not isinstance(down_perm, torch.fx.Node) or not _is_permute(down_perm):
            continue

        seen.add(gate_mm)
        seen.add(up_mm)
        out.append(
            {
                "gate_mm": gate_mm,
                "up_mm": up_mm,
                "down_mm": down_mm,
                "gate_view": view_1,
                "up_view": view_3,
                "mul_node": mul_node,
                "down_view": view_4,
                "gelu_chain": chain,
                "x_src": x_src_g,
                "gate_perm": gate_perm,
                "up_perm": up_perm,
                "down_perm": down_perm,
            }
        )
    return out


def _bsl_to_2d_last(shape: Tuple[int, ...], new_last: int) -> Tuple[int, ...]:
    return tuple(list(shape[:-1]) + [new_last])


def _apply_geglu_plan(
    graph: torch.fx.Graph,
    match: Dict[str, Any],
    *,
    world_size: int,
    plan_id: int,
    shard_id: int,
) -> bool:
    gate_mm: torch.fx.Node = match["gate_mm"]
    up_mm: torch.fx.Node = match["up_mm"]
    down_mm: torch.fx.Node = match["down_mm"]
    gate_perm: torch.fx.Node = match["gate_perm"]
    up_perm: torch.fx.Node = match["up_perm"]
    down_perm: torch.fx.Node = match["down_perm"]
    x_src: torch.fx.Node = match["x_src"]
    gelu_chain: List[torch.fx.Node] = match["gelu_chain"]
    gate_view: torch.fx.Node = match["gate_view"]
    up_view: torch.fx.Node = match["up_view"]
    mul_node: torch.fx.Node = match["mul_node"]
    down_view: torch.fx.Node = match["down_view"]

    # Original placeholder weights.
    gate_w = gate_perm.args[0] if isinstance(gate_perm.args[0], torch.fx.Node) else None
    up_w = up_perm.args[0] if isinstance(up_perm.args[0], torch.fx.Node) else None
    down_w = down_perm.args[0] if isinstance(down_perm.args[0], torch.fx.Node) else None
    if gate_w is None or up_w is None or down_w is None:
        return False

    # Shape sanity. gate_w / up_w: (d, h). down_w: (h, d).
    g_shape = _shape_2d(gate_w)
    u_shape = _shape_2d(up_w)
    dw_shape = _shape_2d(down_w)
    if g_shape is None or u_shape is None or dw_shape is None:
        return False
    d_g, h_g = g_shape
    d_u, h_u = u_shape
    h_d, d_d = dw_shape
    if d_g != d_u or d_g != d_d or h_g != h_u or h_g != h_d:
        _debug(
            f"GeGLU shape mismatch: gate={g_shape} up={u_shape} down={dw_shape}; skipping"
        )
        return False
    d = d_g
    h = h_g
    if d % world_size != 0:
        _debug(f"GeGLU d={d} not divisible by world_size={world_size}; skipping")
        return False
    shard_d = d // world_size

    _debug(
        f"GeGLU match plan_id={plan_id}: gate={_node_sig(gate_mm)} "
        f"up={_node_sig(up_mm)} down={_node_sig(down_mm)} | "
        f"d={d} h={h} shard_d={shard_d} compute_shard={shard_id}/{world_size}"
    )

    # Build per-shard slice -> permute for each weight, anchored before gate_mm.
    gate_shards: List[torch.fx.Node] = []
    up_shards: List[torch.fx.Node] = []
    down_shards: List[torch.fx.Node] = []
    with graph.inserting_before(gate_mm):
        for sid in range(world_size):
            start = sid * shard_d
            end = (sid + 1) * shard_d

            g_slice = graph.call_function(
                aten.slice.Tensor, args=(gate_w, 0, start, end, 1)
            )
            _clone_meta(g_slice, gate_w)
            _set_numa_meta(
                g_slice, plan_id=plan_id, shard_id=sid, total_shards=world_size,
                role="gate_w_slice", target_node=sid,
            )
            g_slice_perm = graph.call_function(
                aten.permute.default, args=(g_slice, [1, 0])
            )
            _clone_meta(g_slice_perm, gate_perm)
            _resize_node_val_and_tensor_meta(g_slice_perm, (h, shard_d))
            _set_numa_meta(
                g_slice_perm, plan_id=plan_id, shard_id=sid, total_shards=world_size,
                role="gate_w_slice_perm", target_node=sid,
            )
            gate_shards.append(g_slice_perm)

            u_slice = graph.call_function(
                aten.slice.Tensor, args=(up_w, 0, start, end, 1)
            )
            _clone_meta(u_slice, up_w)
            _set_numa_meta(
                u_slice, plan_id=plan_id, shard_id=sid, total_shards=world_size,
                role="up_w_slice", target_node=sid,
            )
            u_slice_perm = graph.call_function(
                aten.permute.default, args=(u_slice, [1, 0])
            )
            _clone_meta(u_slice_perm, up_perm)
            _resize_node_val_and_tensor_meta(u_slice_perm, (h, shard_d))
            _set_numa_meta(
                u_slice_perm, plan_id=plan_id, shard_id=sid, total_shards=world_size,
                role="up_w_slice_perm", target_node=sid,
            )
            up_shards.append(u_slice_perm)

            # down_w shape (h, d): row-parallel, slice along input dim = 1.
            d_slice = graph.call_function(
                aten.slice.Tensor, args=(down_w, 1, start, end, 1)
            )
            _clone_meta(d_slice, down_w)
            _set_numa_meta(
                d_slice, plan_id=plan_id, shard_id=sid, total_shards=world_size,
                role="down_w_slice", target_node=sid,
            )
            d_slice_perm = graph.call_function(
                aten.permute.default, args=(d_slice, [1, 0])
            )
            _clone_meta(d_slice_perm, down_perm)
            _resize_node_val_and_tensor_meta(d_slice_perm, (shard_d, h))
            _set_numa_meta(
                d_slice_perm, plan_id=plan_id, shard_id=sid, total_shards=world_size,
                role="down_w_slice_perm", target_node=sid,
            )
            down_shards.append(d_slice_perm)

    shard_plan: Dict[str, Any] = {
        "kind": "mlp",
        "subkind": "geglu",
        "plan_id": plan_id,
        "world_size": world_size,
        "shard_d": shard_d,
        "compute_shard": shard_id,
        "x_node": x_src.name,
        "column_parallel_nodes": [(gate_w.name, 0), (up_w.name, 0)],
        "row_parallel_nodes": [(down_w.name, 1)],
        "replicated_nodes": [x_src.name],
        "zero_on_nonzero_nodes": [],
        "h": h,
        "d": d,
    }
    merge_global_shard_plan("_numa_mlp_tp_shard_plan", shard_plan)

    for sn in itertools.chain(gate_shards, up_shards, down_shards):
        sn.meta["numa_shard_plan"] = shard_plan

    # Annotate replicated activation source.
    _set_numa_meta(
        x_src, plan_id=plan_id, shard_id=0, total_shards=world_size,
        role="x_replica", target_node=0,
    )

    # Rewire compute-shard operands.
    gate_mm.args = (gate_mm.args[0], gate_shards[shard_id])
    up_mm.args = (up_mm.args[0], up_shards[shard_id])
    down_mm.args = (down_mm.args[0], down_shards[shard_id])

    for n, role in (
        (gate_mm, "gate_mm_shard"),
        (up_mm, "up_mm_shard"),
        (down_mm, "down_mm_shard"),
    ):
        _set_numa_meta(
            n, plan_id=plan_id, shard_id=shard_id, total_shards=world_size,
            role=role, target_node=shard_id,
        )
        n.meta["numa_shard_plan"] = shard_plan

    # Refresh activation-chain shapes from full-d to shard_d.
    # gate_mm and up_mm: 2D (s, d) -> (s, shard_d). down_mm: 2D (s, d) -> (s, h) unchanged.
    gate_mm_shape = shape_nd_or_none(gate_mm)
    if gate_mm_shape is not None:
        _resize_node_val_and_tensor_meta(
            gate_mm, _bsl_to_2d_last(gate_mm_shape, shard_d)
        )
    up_mm_shape = shape_nd_or_none(up_mm)
    if up_mm_shape is not None:
        _resize_node_val_and_tensor_meta(up_mm, _bsl_to_2d_last(up_mm_shape, shard_d))

    # Activation chain: gate_view, gelu_chain (9 nodes), up_view, mul_node, down_view.
    chain_to_resize: List[torch.fx.Node] = [gate_view] + list(gelu_chain) + [up_view, mul_node, down_view]
    for n in chain_to_resize:
        sh = shape_nd_or_none(n)
        if sh is None:
            continue
        new_shape = _bsl_to_2d_last(sh, shard_d)
        # Reshape ops carry the target shape as an explicit list literal in
        # args[1]; that literal must be updated alongside meta['val'] or the
        # downstream view fails ("shape [..., d] invalid for input of size ...").
        if _is_reshape(n) and len(n.args) >= 2 and isinstance(n.args[1], (list, tuple)):
            old_shape_arg = list(n.args[1])
            if len(old_shape_arg) == len(new_shape):
                new_shape_arg = list(old_shape_arg[:-1]) + [shard_d]
                n.args = (n.args[0], new_shape_arg) + tuple(n.args[2:])
        _resize_node_val_and_tensor_meta(n, new_shape)
        n.meta["numa_shard_plan"] = shard_plan
        _set_numa_meta(
            n, plan_id=plan_id, shard_id=shard_id, total_shards=world_size,
            role="y_shard", target_node=shard_id,
        )
    return True


def shape_nd_or_none(node: torch.fx.Node) -> Optional[Tuple[int, ...]]:
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
        i = _shape_dim_to_int(s)
        if i is None:
            return None
        out.append(i)
    return tuple(out)


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
            "kind": "mlp",
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
        merge_global_shard_plan("_numa_mlp_tp_shard_plan", shard_plan)

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

    # ---- GeGLU pattern (Gemma) ----
    geglu_matches = _find_geglu_quintuples(graph)
    _debug(f"GeGLU candidates: {len(geglu_matches)}")
    for m in geglu_matches:
        plan_id = next(plan_counter)
        shard_id = _get_compute_shard_id(world_size)
        if _apply_geglu_plan(
            graph, m, world_size=world_size, plan_id=plan_id, shard_id=shard_id
        ):
            changed = True

    if changed:
        graph.lint()
        _debug_graph(graph, "FX graph (after pass)")
        _debug("pass done: graph updated")
    else:
        _debug("pass done: no graph changes")
