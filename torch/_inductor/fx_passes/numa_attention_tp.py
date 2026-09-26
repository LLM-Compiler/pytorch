"""
Megatron-style NUMA tensor-parallel FX pass for attention blocks.

Matches the Q/K/V column-parallel + output row-parallel attention pattern and
rewires the compute-shard's operands onto per-socket weight slices, while
tagging the whole chain with ``numa_shard_plan`` metadata so the Inductor
wrapper codegen emits per-NUMA-node clone/empty_strided allocations and the
Triton-CPU launcher pins one thread group per socket (same mechanism used by
``numa_mlp_tp_pass``).

The pass targets the post-grad / post-aot-autograd graph of a module of the
following (functional) shape:

    def forward(self, x, Wq, Wk, Wv, bq, bk, bv, Wo, bo):
        q = addmm(bq, x, Wq)     # [B*S, H_qkv]
        k = addmm(bk, x, Wk)
        v = addmm(bv, x, Wv)
        q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, S, num_heads, head_dim).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B*S, num_heads * head_dim)
        return addmm(bo, y, Wo)

Under ``SDPBackend.MATH`` the SDPA call decomposes in-graph to primitive ops
(``matmul/bmm + _safe_softmax + matmul/bmm``), so every heavy op in the block
becomes a Triton kernel under ``max_autotune_gemm_backends="TRITON"`` and the
existing ``numa_args`` / ``driver.py`` pinning path works end-to-end.

Column-parallel slicing: ``Wq/Wk/Wv[*, :H_qkv]`` along dim 1 in chunks of
``shard_d = H_qkv // world_size`` (head-aligned iff ``num_heads`` divides
``world_size``); ``bq/bk/bv`` along dim 0. Row-parallel slicing: ``Wo[:H_qkv, *]``
along dim 0 in chunks of ``shard_d``; ``bo`` replicated on shard 0, zeroed on
others (reuse of wrapper's row-parallel bias policy).
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from .. import config
from .numa_tp_common import (
    bias_1d_length,
    clone_meta,
    get_compute_shard_id,
    get_numa_world_size,
    merge_global_shard_plan,
    resize_node_val_and_tensor_meta,
    set_numa_meta,
    shape_2d,
    shape_nd,
)

aten = torch.ops.aten


_MM_TARGETS = frozenset({aten.mm.default, torch.mm, torch.Tensor.mm})
_ADDMM_TARGETS = frozenset({aten.addmm.default, torch.addmm, torch.Tensor.addmm})

_VIEW_TARGETS: Set[Any] = {
    aten.view.default,
    aten._unsafe_view.default,
    aten.reshape.default,
}
# Ops that take an explicit target-shape list as their second positional arg
# AND whose output numel must equal that shape's product (or whose output
# shape must match that list for broadcasting, e.g. ``expand``).
_SHAPE_PARAM_TARGETS: Set[Any] = _VIEW_TARGETS | {aten.expand.default}

# ``view`` / ``reshape`` / ``expand`` may carry literal ``num_heads`` while Q/K/V
# activations are column-sharded. Their output/input numel ratio is often **not**
# ``world_size`` (e.g. ``expand`` uses ``S / E``; post-``bmm`` reshape may see
# ratio ``world_size ** 2``), so ``_try_update_shape_param`` applies literal head
# shrinking before the ratio gate below for these ops only.
_LITERAL_HEAD_SHAPE_OPS_WITHOUT_RATIO_GATE = frozenset(
    {
        aten.expand.default,
        aten.view.default,
        aten._unsafe_view.default,
        aten.reshape.default,
    }
)

# Nodes that PRODUCE a tensor from a constant shape list (no tensor input).
# These often arise from ``zeros_like``/``ones_like`` decompositions and must
# be included in the attention chain so their shape list is rewritten when
# they feed a chain node.
_CONSTANT_SHAPE_PRODUCERS: Set[Any] = {
    aten.full.default,
    aten.zeros.default,
    aten.ones.default,
    aten.empty.memory_format,
}


def _target_is_mm(t: Any) -> bool:
    return t in _MM_TARGETS or getattr(t, "__name__", None) == "mm"


def _target_is_addmm(t: Any) -> bool:
    return t in _ADDMM_TARGETS or getattr(t, "__name__", None) == "addmm"


def _target_is_view(t: Any) -> bool:
    if t in _VIEW_TARGETS:
        return True
    name = getattr(t, "__name__", None)
    return name in ("view", "reshape", "_unsafe_view")


def _target_has_shape_param(t: Any) -> bool:
    if t in _SHAPE_PARAM_TARGETS:
        return True
    name = getattr(t, "__name__", None)
    return name in ("view", "reshape", "_unsafe_view", "expand")


def _target_is_constant_shape_producer(t: Any) -> bool:
    if t in _CONSTANT_SHAPE_PRODUCERS:
        return True
    name = getattr(t, "__name__", None)
    return name in ("full", "zeros", "ones", "empty")


def _is_linear_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    return _target_is_mm(node.target) or _target_is_addmm(node.target)


def _debug_enabled() -> bool:
    return bool(getattr(config, "_numa_attention_tp_debug", False))


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[numa_attention_tp] {msg}", flush=True)


def _debug_graph(graph: torch.fx.Graph, title: str) -> None:
    if not _debug_enabled():
        return
    print(f"[numa_attention_tp] --- {title} ---", flush=True)
    graph.print_tabular()


def _node_sig(n: torch.fx.Node) -> str:
    t = getattr(n, "target", n)
    name = getattr(t, "__name__", str(t))
    return f"{n.name}: {name}"


def _extract_linear_parts(
    node: torch.fx.Node,
) -> Optional[Tuple[Optional[torch.fx.Node], torch.fx.Node, torch.fx.Node]]:
    """Return (bias_or_none, mat1, mat2) for mm/addmm nodes; ``None`` otherwise."""
    if _target_is_mm(node.target):
        if len(node.args) < 2:
            return None
        mat1, mat2 = node.args[0], node.args[1]
        if not isinstance(mat1, torch.fx.Node) or not isinstance(mat2, torch.fx.Node):
            return None
        return None, mat1, mat2
    if _target_is_addmm(node.target):
        if len(node.args) < 3:
            return None
        bias, mat1, mat2 = node.args[0], node.args[1], node.args[2]
        if (
            not isinstance(bias, torch.fx.Node)
            or not isinstance(mat1, torch.fx.Node)
            or not isinstance(mat2, torch.fx.Node)
        ):
            return None
        return bias, mat1, mat2
    return None


# ---------------------------------------------------------------------------
# Pattern search
# ---------------------------------------------------------------------------
def _find_qkv_groups(
    graph: torch.fx.Graph,
) -> List[Tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node]]:
    """Find groups of (x, q_linear, k_linear, v_linear) sharing the same X.

    A group is the set of mm/addmm nodes whose ``mat1`` is the same FX node
    and whose ``mat2`` is a 2D tensor of the same output width. We pick groups
    of exactly 3 (Q/K/V); fused-QKV (single linear) is handled by a different
    matcher (out of scope here per plan decision).
    """
    by_x: Dict[str, List[torch.fx.Node]] = {}
    for node in graph.nodes:
        if not _is_linear_node(node):
            continue
        parts = _extract_linear_parts(node)
        if parts is None:
            continue
        _, mat1, _ = parts
        by_x.setdefault(mat1.name, []).append(node)

    groups: List[Tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node, torch.fx.Node]] = []
    for x_name, linears in by_x.items():
        if len(linears) < 3:
            continue
        # Pick the first triple whose outputs all have same 2D shape.
        # If the user has more than 3 linears sharing X (rare), try 3-subsets.
        for triple in itertools.combinations(linears, 3):
            shapes = [shape_2d(n) for n in triple]
            if any(s is None for s in shapes):
                continue
            if not (shapes[0] == shapes[1] == shapes[2]):
                continue
            parts0 = _extract_linear_parts(triple[0])
            if parts0 is None:
                continue
            x_node = parts0[1]
            groups.append((x_node, triple[0], triple[1], triple[2]))
            break
    return groups


def _collect_chain_and_out_linear(
    qkv_linears: Tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node],
) -> Optional[Tuple[torch.fx.Node, Set[torch.fx.Node]]]:
    """Forward-reachable closure from {q,k,v}_linear until the first downstream
    ``mm``/``addmm`` (candidate ``out_linear``) whose ``mat1`` is inside the
    reachable set.

    Also augments the chain with any constant-shape-producing node (e.g.
    ``aten.full.default``) that is an arg of a chain node — such nodes arise
    from decompositions like ``zeros_like(softmax_out) -> full([shape], 0)``
    and need their explicit shape argument rewritten to the sharded form so
    downstream broadcast ops (``where``, ``add``) match the sharded flow.

    Returns ``(out_linear, chain_nodes)`` — ``chain_nodes`` excludes the three
    input linears and the out_linear themselves; it is the intermediate chain
    whose ``meta['val']`` and shape-arguments (if applicable) need refreshing.
    """
    reachable: Set[torch.fx.Node] = set()
    frontier = list(qkv_linears)
    while frontier:
        n = frontier.pop()
        if n in reachable:
            continue
        reachable.add(n)
        for u in n.users:
            if u not in reachable:
                frontier.append(u)

    # Look for the FIRST candidate out_linear in topological order: a
    # call_function mm/addmm whose mat1 is in reachable and whose mat2 is a
    # fresh input (not in reachable). Iterating the reachable set directly is
    # incorrect — Python sets have no guaranteed iteration order, so a
    # downstream linear whose mat2 happens to hash earlier (e.g. an MLP's
    # weight slice inserted by an earlier post-grad pass) can be picked
    # before the real Wo, causing a size mismatch and silent skip. Walk
    # graph.nodes so the attention-output linear (closest to Q/K/V) always
    # wins.
    graph_obj = qkv_linears[0].graph
    out_linear: Optional[torch.fx.Node] = None
    for node in graph_obj.nodes:
        if node not in reachable:
            continue
        if node in qkv_linears:
            continue
        if not _is_linear_node(node):
            continue
        parts = _extract_linear_parts(node)
        if parts is None:
            continue
        _, mat1, mat2 = parts
        if mat1 in reachable and mat2 not in reachable:
            out_linear = node
            break

    if out_linear is None:
        return None

    # Restrict chain to nodes that are BOTH forward-reachable from QKV AND
    # backward-reachable from out_linear (i.e. strictly between them). The
    # raw forward closure also contains everything downstream of out_linear
    # — residual add, MLP, and subsequent transformer layers — and rewriting
    # shapes/meta on those would corrupt later attention groups whose own
    # QKV linears have not yet been sharded.
    ancestors: Set[torch.fx.Node] = set()
    back_frontier: List[torch.fx.Node] = [out_linear]
    while back_frontier:
        n = back_frontier.pop()
        if n in ancestors:
            continue
        ancestors.add(n)
        for a in n.all_input_nodes:
            if a in reachable and a not in ancestors:
                back_frontier.append(a)

    chain: Set[torch.fx.Node] = set()
    for n in reachable:
        if n in qkv_linears or n is out_linear:
            continue
        if n not in ancestors:
            continue
        chain.add(n)

    # Pull in constant-shape producers that feed any chain node (decomp of
    # ``zeros_like(tensor)`` → ``aten.full.default([shape], 0)``, etc.).
    extra: Set[torch.fx.Node] = set()
    for n in chain:
        for a in n.all_input_nodes:
            if (
                a not in chain
                and a not in qkv_linears
                and a is not out_linear
                and a.op == "call_function"
                and _target_is_constant_shape_producer(a.target)
            ):
                extra.add(a)
    chain |= extra
    return out_linear, chain


# ---------------------------------------------------------------------------
# Shape refresh
# ---------------------------------------------------------------------------
def _effective_head_tuple(
    shape: List[Any],
    *,
    num_heads: int,
    head_dim: int,
    h_qkv: int,
) -> Tuple[int, int]:
    """Resolve (num_heads, head_dim) for BSHE layout checks.

    When the FX graph hides the pre-SDPA 4D reshape, ``num_heads``/``head_dim``
    may be (0, 0). If ``shape`` looks like ``(B, S, H, E)`` with ``H*E == H_qkv``,
    infer ``H, E`` from the last two dims so we never shrink **S** when
    ``S == H_qkv`` (e.g. seq 1024 with 4×256 heads).
    """
    if num_heads > 0 and head_dim > 0:
        return num_heads, head_dim
    if (
        len(shape) == 4
        and isinstance(shape[2], int)
        and isinstance(shape[3], int)
        and shape[2] > 0
        and shape[3] > 0
        and shape[2] * shape[3] == h_qkv
    ):
        return shape[2], shape[3]
    return 0, 0


def _is_bshe(
    shape: List[Any],
    *,
    num_heads: int,
    head_dim: int,
    h_qkv: int,
) -> bool:
    nh, hd = _effective_head_tuple(
        shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv
    )
    if nh <= 0 or hd <= 0:
        return False
    return (
        len(shape) == 4
        and isinstance(shape[2], int)
        and isinstance(shape[3], int)
        and shape[2] == nh
        and shape[3] == hd
    )


def _is_bhse(
    shape: List[Any],
    *,
    num_heads: int,
    head_dim: int,
    h_qkv: int,
) -> bool:
    """``(B, H, S, E)`` layout after ``transpose`` from ``(B, S, H, E)``.

    When ``S == H_qkv`` (possible iff ``S == num_heads * head_dim``), axis ``S``
    matches ``h_qkv`` numerically but must not be column-sharded like QKV width.
    """
    nh, hd = _effective_head_tuple(
        shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv
    )
    if nh <= 0 or hd <= 0:
        return False
    return (
        len(shape) == 4
        and isinstance(shape[1], int)
        and isinstance(shape[2], int)
        and isinstance(shape[3], int)
        and shape[1] == nh
        and shape[3] == hd
    )


def _is_two_d_token_feature_row(
    shape: List[Any],
    *,
    h_qkv: int,
) -> bool:
    """``(B*S, H_qkv)`` activations use dim 0 for tokens and dim 1 for features.

    When ``B*S == H_qkv`` (e.g. one token per channel width), both dims numerically
    equal ``h_qkv``; only dim 1 must be column-sharded for QKV/Wo views.
    """
    return (
        len(shape) == 2
        and isinstance(shape[1], int)
        and shape[1] == h_qkv
    )


def _skip_h_qkv_for_attn_square_axes(shape: List[Any], axis_idx: int) -> bool:
    """True if ``axis_idx`` is a trailing sequence × sequence axis pair.

    SDPA math decomposition uses ``[..., S, S]`` layouts for scores / softmax /
    probs. When ``S`` equals ``H_qkv`` (e.g. GPT-2 Medium width 1024 at seq 1024),
    those axes numerically match the QKV feature width but must not be rewritten
    to ``shard_d`` — doing so halves the wrong dimensions and breaks reshapes.
    """
    n = len(shape)
    if n < 3:
        return False
    s0, s1 = shape[-2], shape[-1]
    if (
        not isinstance(s0, int)
        or not isinstance(s1, int)
        or s0 <= 0
        or s0 != s1
    ):
        return False
    return axis_idx >= n - 2


def _skip_h_qkv_replace_rank3(shape: Sequence[Any]) -> bool:
    """Avoid ``h_qkv → shard_d`` on rank-3 activation tensors.

    SDPA math views become ``[BH, ?, ?]``; when ``S == H_qkv``, a sequence axis can
    equal ``h_qkv`` without being the QKV feature dimension (column-sharding axis
    is always dim 1 of rank-2 ``(B*S, H_qkv)`` tensors).
    """
    return len(shape) == 3


def _try_update_shape_param(
    node: torch.fx.Node,
    *,
    num_heads: int,
    shard_heads: int,
    h_qkv: int,
    shard_d: int,
    world_size: int,
    head_dim: int = 0,
) -> None:
    """For view/reshape/_unsafe_view/expand nodes with an explicit integer
    shape argument, rewrite the shape so that the (single) dim that's sharded
    is narrowed appropriately:

    - ``expand`` / ``view`` / ``reshape`` / ``_unsafe_view``: shrink literal
      ``num_heads`` before ratio checks — broadcast / reshape shapes often have
      output/input ratios like ``S / head_dim`` or ``world_size ** 2``.
    - If a dim equals ``num_heads``, replace with ``shard_heads``.
    - Else if a dim equals ``h_qkv``, replace with ``shard_d``.
    - Else if exactly one dim is divisible by ``world_size`` and rewriting it
      makes numel match the input's (sharded) numel, rewrite that dim.
    - For tensors containing a ``-1`` (infer) dim, compute that dim from the
      input's numel.

    Any update is validated against ``input.meta['val'].numel()`` and
    rolled back if it doesn't match.
    """
    # ``view/reshape/_unsafe_view/expand`` layout: ``(input_tensor, shape_list, ...)``
    # ``full/zeros/ones`` layout: ``(shape_list, fill_value, *)`` — no tensor input.
    is_shape_producer = _target_is_constant_shape_producer(node.target)
    if node.op != "call_function" or not (
        _target_has_shape_param(node.target) or is_shape_producer
    ):
        return
    if is_shape_producer:
        if len(node.args) < 1:
            return
        shape_arg_idx = 0
        input_arg = None
    else:
        if len(node.args) < 2:
            return
        shape_arg_idx = 1
        input_arg = node.args[0]
    shape_arg = node.args[shape_arg_idx]
    if not isinstance(shape_arg, (list, tuple)):
        return

    if is_shape_producer:
        # Target output numel is the product of the shape ARG after we shrink
        # it; since there's no tensor input, we compute ``new_input_numel``
        # from the caller-provided replacements by trying the same literal
        # replacements and accepting the first that yields a well-formed
        # shape. We still cross-check via the ``old_numel -> old_numel /
        # world_size`` rule below.
        new_input_numel = 0
    else:
        if not isinstance(input_arg, torch.fx.Node):
            return
        input_val = input_arg.meta.get("val")
        if not isinstance(input_val, torch.Tensor):
            return
        new_input_numel = int(input_val.numel())

    old_shape: List[Any] = list(shape_arg)

    def _rewrite(new_shape_list: List[Any]) -> None:
        new_args = list(node.args)
        new_args[shape_arg_idx] = type(shape_arg)(new_shape_list)
        node.args = tuple(new_args)

    # Inferred (``-1``) dim present → recompute it from input numel.
    if any(isinstance(d, int) and d == -1 for d in old_shape):
        if is_shape_producer:
            return  # producers can't have -1
        neg_idx = next(i for i, d in enumerate(old_shape) if d == -1)
        fixed = 1
        for i, d in enumerate(old_shape):
            if i == neg_idx:
                continue
            if isinstance(d, int) and d > 0:
                fixed *= d
            else:
                return  # can't resolve with mixed types
        if fixed <= 0 or new_input_numel % fixed != 0:
            return
        old_shape[neg_idx] = new_input_numel // fixed
        _rewrite(old_shape)
        return

    # No -1: for shape-param ops, numel must shrink to match input. For
    # constant-shape producers, there is no input — just apply the
    # head/h_qkv literal replacements.
    old_numel = 1
    for d in old_shape:
        if not isinstance(d, int) or d <= 0:
            return
        old_numel *= d

    # ``view`` / ``reshape`` / ``expand`` targets often retain literal ``num_heads``
    # while activations are column-sharded; output/input ratio vs tensor input may
    # not be ``world_size`` (see ``_LITERAL_HEAD_SHAPE_OPS_WITHOUT_RATIO_GATE``).
    if (
        not is_shape_producer
        and node.target in _LITERAL_HEAD_SHAPE_OPS_WITHOUT_RATIO_GATE
        and shard_heads
    ):
        nh_lit, _hd_lit = _effective_head_tuple(
            old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv
        )
        if nh_lit:
            bshe_lit = _is_bshe(
                old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv
            )
            bhse_lit = _is_bhse(
                old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv
            )
            for i, d in enumerate(old_shape):
                if d != nh_lit:
                    continue
                if bshe_lit and i != 2:
                    continue
                if bhse_lit and i != 1:
                    continue
                trial = list(old_shape)
                trial[i] = shard_heads
                _rewrite(trial)
                return

    if is_shape_producer:
        # Try literal replacements; do NOT require numel-shrink ratio match
        # (the producer outputs whatever shape we give it).
        nh_eff, hd_eff = _effective_head_tuple(
            old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv
        )
        if nh_eff and shard_heads:
            for i, d in enumerate(old_shape):
                if d != nh_eff:
                    continue
                if _is_bshe(old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv) and i != 2:
                    continue
                if _is_bhse(old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv) and i != 1:
                    continue
                trial = list(old_shape)
                trial[i] = shard_heads
                _rewrite(trial)
                return
        if h_qkv and shard_d and not _skip_h_qkv_replace_rank3(old_shape):
            for i, d in enumerate(old_shape):
                if d != h_qkv:
                    continue
                if _is_two_d_token_feature_row(old_shape, h_qkv=h_qkv) and i != 1:
                    continue
                if _is_bshe(old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv) and i == 1:
                    continue
                if _is_bhse(old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv) and i == 2:
                    continue
                if _skip_h_qkv_for_attn_square_axes(old_shape, i):
                    continue
                trial = list(old_shape)
                trial[i] = shard_d
                _rewrite(trial)
                return
        return

    if old_numel == new_input_numel:
        return
    if old_numel <= new_input_numel or old_numel % new_input_numel != 0:
        return
    ratio = old_numel // new_input_numel
    if ratio != world_size:
        return

    def _try_replace(idx: int, new_val: int) -> Optional[List[int]]:
        trial = list(old_shape)
        trial[idx] = new_val
        numel = 1
        for d in trial:
            numel *= d
        return trial if numel == new_input_numel else None

    nh_eff, _hd_eff = _effective_head_tuple(
        old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv
    )
    bshe = _is_bshe(old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv)
    bhse = _is_bhse(old_shape, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv)

    # Prefer replacing num_heads (in any dim containing that literal).
    if nh_eff and shard_heads:
        for i, d in enumerate(old_shape):
            if d != nh_eff:
                continue
            if bshe and i != 2:
                continue
            if bhse and i != 1:
                continue
            trial = _try_replace(i, shard_heads)
            if trial is not None:
                _rewrite(trial)
                return

    # Then try H_qkv -> shard_d (never axis ``S`` in ``(B,S,H,E)`` when S == H_qkv;
    # never dim 0 of ``(B*S, H_qkv)`` when B*S == H_qkv).
    if h_qkv and shard_d and not _skip_h_qkv_replace_rank3(old_shape):
        for i, d in enumerate(old_shape):
            if d != h_qkv:
                continue
            if _is_two_d_token_feature_row(old_shape, h_qkv=h_qkv) and i != 1:
                continue
            if bshe and i == 1:
                continue
            if bhse and i == 2:
                continue
            if _skip_h_qkv_for_attn_square_axes(old_shape, i):
                continue
            trial = _try_replace(i, shard_d)
            if trial is not None:
                _rewrite(trial)
                return

    # Then try B*H-style dims: any dim exactly divisible by world_size whose
    # quotient makes the numel match.
    for i, d in enumerate(old_shape):
        if d % world_size != 0:
            continue
        if bshe and i == 1:
            continue
        if bhse and i == 2:
            continue
        if (
            _is_two_d_token_feature_row(old_shape, h_qkv=h_qkv)
            and i == 0
        ):
            continue
        trial = _try_replace(i, d // world_size)
        if trial is not None:
            _rewrite(trial)
            return


def _shrink_shape(
    shape: Tuple[Any, ...],
    *,
    num_heads: int,
    shard_heads: int,
    h_qkv: int,
    shard_d: int,
    world_size: int,
    head_dim: int = 0,
) -> Tuple[Any, ...]:
    """Shrink any dim that matches ``num_heads``/``h_qkv`` to its sharded
    equivalent. A single occurrence of ``num_heads`` (resp. ``h_qkv``) is
    replaced; if multiple dims collide with the same value (pathological),
    all are replaced — this is acceptable since the attention chain is
    expected to mention the head axis no more than once per tensor.
    """
    shape_list = list(shape)
    nh_eff, _hd_eff = _effective_head_tuple(
        shape_list, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv
    )
    bshe = _is_bshe(shape_list, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv)
    bhse = _is_bhse(shape_list, num_heads=num_heads, head_dim=head_dim, h_qkv=h_qkv)
    new_shape: List[Any] = []
    for i, d in enumerate(shape):
        if isinstance(d, int):
            if nh_eff and d == nh_eff:
                if bshe and i != 2:
                    new_shape.append(d)
                    continue
                if bhse and i != 1:
                    new_shape.append(d)
                    continue
                new_shape.append(shard_heads)
                continue
            if h_qkv and d == h_qkv:
                if _skip_h_qkv_replace_rank3(shape_list):
                    new_shape.append(d)
                    continue
                if len(shape) == 2 and i != 1:
                    new_shape.append(d)
                    continue
                if bshe and i == 1:
                    new_shape.append(d)
                    continue
                if bhse and i == 2:
                    new_shape.append(d)
                    continue
                if _skip_h_qkv_for_attn_square_axes(shape_list, i):
                    new_shape.append(d)
                    continue
                new_shape.append(shard_d)
                continue
            if (
                world_size
                and d > 0
                and d % world_size == 0
                and nh_eff
                and not (bshe and i == 1)
                and not (bhse and i == 2)
                and not (
                    len(shape) == 2
                    and i == 0
                    and isinstance(shape[1], int)
                    and shape[1] == h_qkv
                )
                and (d // world_size) * world_size == d
                and d == nh_eff * max(1, d // max(1, nh_eff))
            ):
                # B*num_heads-style dim
                new_shape.append(d // world_size)
                continue
        new_shape.append(d)
    return tuple(new_shape)


def _refresh_chain_vals_via_fake_tensor(
    nodes: List[torch.fx.Node],
    *,
    shape_rewrite_kwargs: Optional[Dict[str, int]] = None,
) -> None:
    """Best-effort forward propagation of ``meta['val']`` for chain nodes.

    Walks ``nodes`` in topological order and for each node:

    1. If its target takes an explicit shape argument (view/reshape/expand),
       rewrite the shape argument to the sharded equivalent via
       ``_try_update_shape_param`` (using the caller-supplied
       ``shape_rewrite_kwargs``). This MUST happen before evaluating the op
       so that the FakeTensor call sees a consistent shape arg.
    2. Recompute ``meta['val']`` by calling the op on the now-sharded
       upstream FakeTensors.
    3. On failure (e.g. broadcast mismatch because a sibling tensor slipped
       through without being shape-updated), synthesize a new FakeTensor by
       shrinking the stale ``meta['val']`` along the head axis using
       ``_shrink_shape``. This prevents failure cascades and ensures
       downstream Inductor lowering sees a consistent sharded chain.
    """
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx.passes.shape_prop import _extract_tensor_metadata

    fake_mode: Optional[FakeTensorMode] = None
    for n in nodes:
        v = n.meta.get("val")
        if isinstance(v, FakeTensor):
            fake_mode = v.fake_mode
            break
    if fake_mode is None:
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    def _materialize(arg: Any) -> Any:
        if isinstance(arg, torch.fx.Node):
            return arg.meta.get("val", None)
        if isinstance(arg, (list, tuple)):
            return type(arg)(_materialize(a) for a in arg)
        if isinstance(arg, dict):
            return {k: _materialize(v) for k, v in arg.items()}
        return arg

    def _record_val(node: torch.fx.Node, result: Any) -> None:
        node.meta["val"] = result
        if isinstance(result, torch.Tensor):
            try:
                node.meta["tensor_meta"] = _extract_tensor_metadata(result)
            except Exception:
                pass

    def _shrink_fallback(node: torch.fx.Node) -> bool:
        """Synthesize a sharded FakeTensor for ``node`` by shrinking its
        current ``meta['val']`` shape along known head-axis values."""
        if shape_rewrite_kwargs is None:
            return False
        old_val = node.meta.get("val")
        if not isinstance(old_val, torch.Tensor):
            return False
        new_shape = _shrink_shape(tuple(old_val.shape), **shape_rewrite_kwargs)
        if new_shape == tuple(old_val.shape):
            return False
        with fake_mode:
            new_val = old_val.new_empty(torch.Size(new_shape))
        _record_val(node, new_val)
        _debug(
            f"shape-refresh: fallback shrink {_node_sig(node)}: "
            f"{tuple(old_val.shape)} -> {tuple(new_val.shape)}"
        )
        return True

    for node in nodes:
        if node.op != "call_function":
            continue
        if shape_rewrite_kwargs is not None and (
            _target_has_shape_param(node.target)
            or _target_is_constant_shape_producer(node.target)
        ):
            _try_update_shape_param(node, **shape_rewrite_kwargs)
        try:
            m_args = _materialize(node.args)
            m_kwargs = _materialize(dict(node.kwargs))
            with fake_mode:
                result = node.target(*m_args, **m_kwargs)  # type: ignore[misc]
            _record_val(node, result)
        except Exception as exc:
            _debug(
                f"shape-refresh: op-run failed for {_node_sig(node)} "
                f"({type(exc).__name__}: {exc}); trying shape-shrink fallback"
            )
            if not _shrink_fallback(node):
                _debug(
                    f"shape-refresh: giving up on {_node_sig(node)} (no shrink available)"
                )


def _topo_sorted_subset(
    graph: torch.fx.Graph, subset: Set[torch.fx.Node]
) -> List[torch.fx.Node]:
    """Return ``subset`` nodes in topological order (inputs before users).

    ``torch.fx.Graph.nodes`` iteration order follows tracing but is **not**
    guaranteed to be a valid execution order when filtering to ``subset``.
    Shape refresh must walk producers before consumers so ``meta['val']`` is
    current before ``expand`` / ``view`` rewrite runs.
    """
    order: List[torch.fx.Node] = []
    visited: Set[torch.fx.Node] = set()

    def _dfs(n: torch.fx.Node) -> None:
        if n not in subset or n in visited:
            return
        for inp in n.all_input_nodes:
            _dfs(inp)
        visited.add(n)
        order.append(n)

    for node in graph.nodes:
        _dfs(node)
    return order


# ---------------------------------------------------------------------------
# Core pass
# ---------------------------------------------------------------------------
def numa_attention_tp_pass(graph: torch.fx.Graph) -> None:
    """Rewrite attention blocks (Q/K/V linears + SDPA chain + out linear) for
    NUMA tensor-parallel execution. See module docstring for semantics."""
    world_size = get_numa_world_size("_numa_attention_tp_world_size")
    _debug(
        f"pass start: world_size={world_size} "
        f"(override={getattr(config, '_numa_attention_tp_world_size', 0)})"
    )
    if world_size < 2:
        _debug(f"skip: world_size={world_size} < 2")
        return

    plan_counter = itertools.count()
    changed = False
    for x_node, q_lin, k_lin, v_lin in _find_qkv_groups(graph):
        parts_q = _extract_linear_parts(q_lin)
        parts_k = _extract_linear_parts(k_lin)
        parts_v = _extract_linear_parts(v_lin)
        if parts_q is None or parts_k is None or parts_v is None:
            continue
        bq, _, Wq = parts_q
        bk, _, Wk = parts_k
        bv, _, Wv = parts_v

        q_shape = shape_2d(q_lin)
        if q_shape is None:
            continue
        bs, h_qkv = q_shape
        if h_qkv % world_size != 0:
            _debug(
                f"skip group x={_node_sig(x_node)}: H_qkv={h_qkv} "
                f"not divisible by world_size={world_size}"
            )
            continue

        # Inspect optional biases — must each be 1D length H_qkv for the
        # simple column-parallel rewrite to be safe.
        for b_candidate in (bq, bk, bv):
            if b_candidate is None:
                continue
            if bias_1d_length(b_candidate) != h_qkv:
                _debug(
                    f"skip group x={_node_sig(x_node)}: bias "
                    f"{_node_sig(b_candidate)} not 1D[H_qkv]={h_qkv}"
                )
                break
        else:
            # Find the reachable chain + out_linear.
            found = _collect_chain_and_out_linear((q_lin, k_lin, v_lin))
            if found is None:
                _debug(
                    f"skip group x={_node_sig(x_node)}: no out_linear found "
                    "downstream of Q/K/V"
                )
                continue
            out_linear, chain_nodes = found

            parts_o = _extract_linear_parts(out_linear)
            if parts_o is None:
                continue
            bo, _, Wo = parts_o

            out_shape = shape_2d(out_linear)
            if out_shape is None:
                continue
            out_bs, h_in = out_shape

            wo_shape = shape_2d(Wo)
            if wo_shape is None or wo_shape[0] != h_qkv:
                _debug(
                    f"skip: Wo shape mismatch (Wo={wo_shape}, expected rows={h_qkv})"
                )
                continue

            # Infer num_heads / head_dim from the first 4D view shape applied
            # to q_lin's output (pre-SDPA reshape).
            num_heads: Optional[int] = None
            head_dim: Optional[int] = None
            for u in q_lin.users:
                if u.op == "call_function" and _target_is_view(u.target):
                    if len(u.args) >= 2 and isinstance(u.args[1], (list, tuple)):
                        shape = u.args[1]
                        if len(shape) == 4:
                            nh = shape[2] if isinstance(shape[2], int) else None
                            hd = shape[3] if isinstance(shape[3], int) else None
                            if nh is not None and hd is not None and nh * hd == h_qkv:
                                num_heads, head_dim = nh, hd
                                break
            if num_heads is None or head_dim is None:
                # Fall back: assume 4D reshape isn't visible (e.g. the pre-SDPA
                # reshape was folded). The pass can still proceed — we just
                # won't be able to auto-rewrite view shape args (Inductor may
                # have already propagated shapes via other means).
                _debug(
                    f"group x={_node_sig(x_node)}: could not infer num_heads/head_dim "
                    f"from q_lin users; continuing with shard only"
                )
                num_heads, head_dim = 0, 0

            if num_heads and num_heads % world_size != 0:
                _debug(
                    f"skip: num_heads={num_heads} not divisible by "
                    f"world_size={world_size} (head-aligned sharding only)"
                )
                continue

            plan_id = next(plan_counter)
            shard_d = h_qkv // world_size
            shard_heads = (num_heads // world_size) if num_heads else 0
            shard_id = get_compute_shard_id(
                world_size, "_numa_attention_tp_compute_shard"
            )
            _debug(
                f"match plan_id={plan_id}: "
                f"{_node_sig(q_lin)}|{_node_sig(k_lin)}|{_node_sig(v_lin)} -> "
                f"{_node_sig(out_linear)} | "
                f"x=[{bs},{h_in}] (h_in inferred {h_in}) H_qkv={h_qkv} "
                f"H={num_heads} E={head_dim} | "
                f"shard_d={shard_d} shard_heads={shard_heads} "
                f"compute_shard={shard_id}/{world_size}"
            )

            # ---------------------------------------------------------------
            # 1. Insert per-shard slice nodes (no clones at FX level).
            # ---------------------------------------------------------------
            Wq_slices: List[torch.fx.Node] = []
            Wk_slices: List[torch.fx.Node] = []
            Wv_slices: List[torch.fx.Node] = []
            bq_slices: List[torch.fx.Node] = []
            bk_slices: List[torch.fx.Node] = []
            bv_slices: List[torch.fx.Node] = []
            Wo_slices: List[torch.fx.Node] = []

            with graph.inserting_before(q_lin):
                for sid in range(world_size):
                    start = sid * shard_d
                    end = (sid + 1) * shard_d

                    def _make_slice(src: torch.fx.Node, dim: int, role: str) -> torch.fx.Node:
                        s = graph.call_function(
                            aten.slice.Tensor, args=(src, dim, start, end, 1)
                        )
                        clone_meta(s, src)
                        set_numa_meta(
                            s,
                            plan_id=plan_id,
                            shard_id=sid,
                            total_shards=world_size,
                            role=role,
                            target_node=sid,
                        )
                        return s

                    Wq_slices.append(_make_slice(Wq, 1, "wq_slice"))
                    Wk_slices.append(_make_slice(Wk, 1, "wk_slice"))
                    Wv_slices.append(_make_slice(Wv, 1, "wv_slice"))
                    if bq is not None:
                        bq_slices.append(_make_slice(bq, 0, "bq_slice"))
                    if bk is not None:
                        bk_slices.append(_make_slice(bk, 0, "bk_slice"))
                    if bv is not None:
                        bv_slices.append(_make_slice(bv, 0, "bv_slice"))
                    Wo_slices.append(_make_slice(Wo, 0, "wo_slice"))

            _debug(
                f"plan_id={plan_id}: inserted per-shard slices "
                f"(Wq/Wk/Wv/Wo x {world_size}"
                f"{', bq/bk/bv x ' + str(world_size) if (bq is not None or bk is not None or bv is not None) else ''})"
            )

            # ---------------------------------------------------------------
            # 2. Annotate X as replicated.
            # ---------------------------------------------------------------
            set_numa_meta(
                x_node,
                plan_id=plan_id,
                shard_id=0,
                total_shards=world_size,
                role="x_replica",
                target_node=0,
            )

            # ---------------------------------------------------------------
            # 3. Build generalized shard plan for wrapper codegen.
            # ---------------------------------------------------------------
            shard_plan: Dict[str, Any] = {
                "kind": "attention",
                "plan_id": plan_id,
                "world_size": world_size,
                "shard_d": shard_d,
                "compute_shard": shard_id,
                "x_node": x_node.name,
                "column_parallel_nodes": [
                    (Wq.name, 1), (Wk.name, 1), (Wv.name, 1),
                ]
                + ([(bq.name, 0)] if bq is not None else [])
                + ([(bk.name, 0)] if bk is not None else [])
                + ([(bv.name, 0)] if bv is not None else []),
                "row_parallel_nodes": [(Wo.name, 0)],
                "replicated_nodes": [x_node.name]
                + ([bo.name] if bo is not None else []),
                "zero_on_nonzero_nodes": [bo.name] if bo is not None else [],
                # Shape metadata for debugging / downstream passes.
                "num_heads": num_heads,
                "head_dim": head_dim,
                "h_in": h_in,
                "h_qkv": h_qkv,
                "bs": bs,
            }
            # Persist the latest full plan so wrapper codegen can recover
            # semantics even if a backend path drops direct origin metadata.
            merge_global_shard_plan("_numa_attention_tp_shard_plan", shard_plan)

            # Propagate the shard plan to all per-shard slice nodes so
            # downstream passes and wrapper codegen can recover the layout
            # from any of them.
            for s in (
                Wq_slices + Wk_slices + Wv_slices + Wo_slices
                + bq_slices + bk_slices + bv_slices
            ):
                s.meta["numa_shard_plan"] = shard_plan

            # ---------------------------------------------------------------
            # 4. Rewire compute-shard operands into the three QKV linears
            #    and the out_linear.
            # ---------------------------------------------------------------
            def _rewire_linear(
                lin: torch.fx.Node,
                bias_node: Optional[torch.fx.Node],
                bias_slices: List[torch.fx.Node],
                mat2_slices: List[torch.fx.Node],
                role: str,
            ) -> None:
                mat2_shard = mat2_slices[shard_id]
                if _target_is_mm(lin.target):
                    lin.args = (lin.args[0], mat2_shard)
                else:
                    assert _target_is_addmm(lin.target)
                    if bias_node is not None and bias_slices:
                        lin.args = (bias_slices[shard_id], lin.args[1], mat2_shard)
                    else:
                        lin.args = (lin.args[0], lin.args[1], mat2_shard)
                set_numa_meta(
                    lin,
                    plan_id=plan_id,
                    shard_id=shard_id,
                    total_shards=world_size,
                    role=role,
                    target_node=shard_id,
                )
                lin.meta["numa_shard_plan"] = shard_plan

            _rewire_linear(q_lin, bq, bq_slices, Wq_slices, "q_shard")
            _rewire_linear(k_lin, bk, bk_slices, Wk_slices, "k_shard")
            _rewire_linear(v_lin, bv, bv_slices, Wv_slices, "v_shard")

            # Out linear is row-parallel (``z_shard``) — this triggers the
            # existing post-kernel all-reduce emission in wrapper.py.
            Wo_shard = Wo_slices[shard_id]
            if _target_is_mm(out_linear.target):
                out_linear.args = (out_linear.args[0], Wo_shard)
            else:
                assert _target_is_addmm(out_linear.target)
                # Keep bo replicated in the graph (wrapper emits zero-bias on
                # non-zero sockets based on zero_on_nonzero_nodes).
                out_linear.args = (out_linear.args[0], out_linear.args[1], Wo_shard)
            set_numa_meta(
                out_linear,
                plan_id=plan_id,
                shard_id=shard_id,
                total_shards=world_size,
                role="z_shard",
                target_node=shard_id,
            )
            out_linear.meta["numa_shard_plan"] = shard_plan

            # ---------------------------------------------------------------
            # 5. Refresh shape metadata on Q/K/V linears and on the chain.
            # ---------------------------------------------------------------
            resize_node_val_and_tensor_meta(q_lin, (bs, shard_d))
            resize_node_val_and_tensor_meta(k_lin, (bs, shard_d))
            resize_node_val_and_tensor_meta(v_lin, (bs, shard_d))

            # Topologically walk the chain once. For each node (in order):
            #   1. If it takes an explicit shape argument (view/reshape/_unsafe_view/expand),
            #      rewrite the shape so a dim of ``num_heads`` becomes
            #      ``shard_heads`` (or ``h_qkv`` becomes ``shard_d``, or any
            #      dim divisible by ``world_size`` is narrowed).
            #   2. Re-run the op on the upstream sharded FakeTensors so
            #      ``meta['val']`` picks up the new shape.
            topo_chain = _topo_sorted_subset(graph, chain_nodes)
            shape_rewrite_kwargs = {
                "num_heads": num_heads,
                "shard_heads": shard_heads,
                "h_qkv": h_qkv,
                "shard_d": shard_d,
                "world_size": world_size,
                "head_dim": head_dim or 0,
            }
            _refresh_chain_vals_via_fake_tensor(
                topo_chain, shape_rewrite_kwargs=shape_rewrite_kwargs
            )
            # Tag every chain node with the plan so downstream codegen can
            # recover semantics from the kernel's origin node.
            for n in topo_chain:
                n.meta["numa_shard_plan"] = shard_plan
                if n.meta.get("numa_role") is None:
                    set_numa_meta(
                        n,
                        plan_id=plan_id,
                        shard_id=shard_id,
                        total_shards=world_size,
                        role="attn_shard",
                        target_node=shard_id,
                    )

            _debug(
                f"plan_id={plan_id}: refreshed meta on Q/K/V linears + "
                f"{len(topo_chain)} chain node(s); z_shard on out_linear "
                f"{_node_sig(out_linear)}"
            )
            if _debug_enabled():
                for cn in topo_chain:
                    v = cn.meta.get("val")
                    shape = tuple(v.shape) if isinstance(v, torch.Tensor) else None
                    _debug(f"  chain {_node_sig(cn)} val.shape={shape}")
            changed = True
            continue
        # The ``for b_candidate ... else`` block exits here if the for-loop
        # did NOT complete (i.e. a bad bias was found). Skip this group.

    if changed:
        graph.lint()
        _debug_graph(graph, "FX graph (after pass)")
        _debug("pass done: graph updated")
    else:
        _debug("pass done: no graph changes")
