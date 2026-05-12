"""NUMA tensor-parallel pass for Gemma-style prefill attention.

Targets the post-grad FX graph that ``torch.compile`` produces for a Gemma
decoder block whose ``GemmaAttention`` was rewritten to use *separate*
``Wq``/``Wk``/``Wv`` linears (no fused ``qkv_proj``), no QK-norm, no
attn-logit softcapping. Manual softmax (``add → amax → sub → exp → sum →
div``) and grouped-query attention with ``num_kv_heads < num_heads`` are
both handled; ``num_kv_heads=1`` (Gemma-3 270M default) means K and V are
**replicated** across NUMA nodes and only Q (and the output projection) are
sharded.

Sharding strategy (world_size = N):
    - Wq: column-parallel, slice along output dim (dim 0 of the placeholder
      whose shape is ``(q_size, hidden)``). ``q_size = num_heads * head_dim``;
      ``shard_q = q_size // N``.
    - Wk, Wv: replicated. The graph still references the full placeholders.
    - Wo: row-parallel, slice along input dim (dim 1 of the placeholder
      whose shape is ``(hidden, q_size)``).
    - Activation chain: every reshape/expand shape literal that contains
      ``q_size`` or ``num_heads`` is rewritten to ``shard_q`` /
      ``num_heads // N``. ``meta['val']`` is refreshed so downstream FX/
      Inductor passes see consistent shapes.

The pass leaves the K/V index_put_ writes into kv_cache untouched (the same
full-sized cache is shared across NUMA nodes; replicated-weight semantics
mean every node writes the same value, which is correct).
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch

from .. import config

aten = torch.ops.aten


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
def _debug_enabled() -> bool:
    return bool(getattr(config, "_numa_attention_tp_debug", False))


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[numa_gemma_attn_tp] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Config helpers (mirror numa_mlp_tp.py)
# ---------------------------------------------------------------------------
def _get_world_size() -> int:
    override = int(getattr(config, "_numa_attention_tp_world_size", 0))
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


def _get_compute_shard(world_size: int) -> int:
    sid = int(getattr(config, "_numa_attention_tp_compute_shard", 0))
    if world_size <= 0:
        return 0
    if sid < 0:
        return 0
    if sid >= world_size:
        return world_size - 1
    return sid


# ---------------------------------------------------------------------------
# Node target / shape utilities
# ---------------------------------------------------------------------------
def _is_call(n: Any, target: Any) -> bool:
    return isinstance(n, torch.fx.Node) and n.op == "call_function" and n.target is target


def _is_mm(n: Any) -> bool:
    return _is_call(n, aten.mm.default)


def _is_reshape(n: Any) -> bool:
    return _is_call(n, aten.reshape.default) or _is_call(n, aten.view.default)


def _is_permute(n: Any) -> bool:
    return _is_call(n, aten.permute.default)


def _is_expand(n: Any) -> bool:
    return _is_call(n, aten.expand.default)


def _is_clone(n: Any) -> bool:
    return _is_call(n, aten.clone.default)


def _is_bmm(n: Any) -> bool:
    return _is_call(n, aten.bmm.default)


def _is_index_put(n: Any) -> bool:
    return _is_call(n, aten.index_put.default)


def _is_copy_(n: Any) -> bool:
    return _is_call(n, aten.copy_.default)


def _find_cache_index_put(mm_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Walk forward from ``mm_node`` and return the first ``aten.index_put.default``
    node whose target (args[0]) is a placeholder. That target is the K or V
    cache that gets written by the in-place ``index_copy_`` in the source.
    Returns ``None`` if no such node is found in the forward cone (e.g.
    cacheless test harnesses or eager-only models).
    """
    visited: set = set()
    frontier: List[torch.fx.Node] = [mm_node]
    while frontier:
        n = frontier.pop()
        if n in visited:
            continue
        visited.add(n)
        for u in n.users:
            if (
                _is_index_put(u)
                and len(u.args) >= 1
                and isinstance(u.args[0], torch.fx.Node)
                and u.args[0].op == "placeholder"
            ):
                return u
            if u not in visited:
                frontier.append(u)
    return None


def _find_companion_copy_(cache_ph: torch.fx.Node, ip_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """After functionalization the in-place ``index_copy_`` becomes::

        ip = aten.index_put.default(cache_ph, [...], values)
        aten.copy_.default(cache_ph, ip)              # writes back via alias

    Return that ``copy_`` node, or ``None`` if it has been DCE'd / never existed.
    """
    for u in list(cache_ph.users):
        if (
            _is_copy_(u)
            and len(u.args) >= 2
            and u.args[0] is cache_ph
            and u.args[1] is ip_node
        ):
            return u
    return None


def _shape(n: torch.fx.Node) -> Optional[Tuple[int, ...]]:
    val = n.meta.get("val")
    if isinstance(val, torch.Tensor):
        out: List[int] = []
        for s in val.shape:
            try:
                out.append(int(s))
            except (TypeError, ValueError):
                return None
        return tuple(out)
    return None


def _refresh_val(n: torch.fx.Node, new_shape: Tuple[int, ...]) -> None:
    val = n.meta.get("val")
    if isinstance(val, torch.Tensor):
        n.meta["val"] = val.new_empty(torch.Size(new_shape))
        try:
            from torch.fx.passes.shape_prop import _extract_tensor_metadata
            n.meta["tensor_meta"] = _extract_tensor_metadata(n.meta["val"])
        except Exception:
            pass


def _replace_shape_arg(n: torch.fx.Node, *, old_dim: int, new_dim: int, arg_index: int = 1) -> None:
    """Rewrite a list/tuple shape arg on a reshape/expand: any element equal
    to ``old_dim`` becomes ``new_dim``. Leaves -1 and other dims alone."""
    if len(n.args) <= arg_index:
        return
    raw = n.args[arg_index]
    if not isinstance(raw, (list, tuple)):
        return
    new_list: List[Any] = []
    for v in raw:
        if isinstance(v, int) and v == old_dim:
            new_list.append(new_dim)
        else:
            new_list.append(v)
    head = list(n.args[:arg_index])
    tail = list(n.args[arg_index + 1:])
    n.args = tuple(head + [new_list] + tail)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------
def _trace_back_to_mm(n: torch.fx.Node, max_hops: int = 6) -> Optional[torch.fx.Node]:
    """Walk back through reshape/permute/clone/expand to find the source mm."""
    cur: Optional[torch.fx.Node] = n
    for _ in range(max_hops):
        if cur is None:
            return None
        if _is_mm(cur):
            return cur
        if (_is_reshape(cur) or _is_permute(cur) or _is_clone(cur) or _is_expand(cur)) and len(cur.args) >= 1:
            nxt = cur.args[0]
            cur = nxt if isinstance(nxt, torch.fx.Node) else None
            continue
        return None
    return None


def _find_softmax_chain_after_add(add_n: torch.fx.Node) -> Optional[Dict[str, torch.fx.Node]]:
    """Manual softmax: add -> amax -> sub -> exp -> sum -> div."""
    amax = next((u for u in add_n.users if _is_call(u, aten.amax.default)), None)
    if amax is None:
        return None
    sub = next((u for u in add_n.users if _is_call(u, aten.sub.Tensor) and amax in u.args), None)
    if sub is None:
        return None
    exp = next((u for u in sub.users if _is_call(u, aten.exp.default)), None)
    if exp is None:
        return None
    sum_n = next(
        (u for u in exp.users if _is_call(u, aten.sum.dim_IntList) or _is_call(u, aten.sum.default)),
        None,
    )
    if sum_n is None:
        return None
    div = next((u for u in exp.users if _is_call(u, aten.div.Tensor) and sum_n in u.args), None)
    if div is None:
        return None
    return {"amax": amax, "sub": sub, "exp": exp, "sum": sum_n, "div": div}


def _find_attention_blocks(graph: torch.fx.Graph) -> List[Dict[str, Any]]:
    """Scan the graph and return one match dict per Gemma attention block.

    A match is anchored on the first ``bmm`` (q@k.T), whose downstream chain
    we can deterministically walk forward to softmax → second bmm → output.
    """
    matches: List[Dict[str, Any]] = []
    for bmm0 in list(graph.nodes):
        if not _is_bmm(bmm0):
            continue
        # bmm output -> reshape -> [optional softcap: div -> tanh -> mul] -> add(mask) -> softmax
        view_qk = next((u for u in bmm0.users if _is_reshape(u)), None)
        if view_qk is None:
            continue
        softcap_nodes: List[torch.fx.Node] = []
        anchor: torch.fx.Node = view_qk
        # Gemma 9B/27B attn-logit softcapping: scores = mul(tanh(div(scores, cap)), cap).
        # Detect by looking for a div.Tensor user of view_qk whose other arg is a scalar
        # (the softcap value, e.g. 50.0 for Gemma-27B).
        div_n = next(
            (
                u for u in view_qk.users
                if _is_call(u, aten.div.Tensor) and len(u.args) >= 2 and not isinstance(u.args[1], torch.fx.Node)
            ),
            None,
        )
        if div_n is not None:
            tanh_n = next((u for u in div_n.users if _is_call(u, aten.tanh.default)), None)
            mul_n = (
                next(
                    (
                        u for u in tanh_n.users
                        if _is_call(u, aten.mul.Tensor) and len(u.args) >= 2 and not isinstance(u.args[1], torch.fx.Node)
                    ),
                    None,
                )
                if tanh_n is not None
                else None
            )
            if tanh_n is not None and mul_n is not None:
                softcap_nodes = [div_n, tanh_n, mul_n]
                anchor = mul_n
        add_mask = next((u for u in anchor.users if _is_call(u, aten.add.Tensor)), None)
        if add_mask is None:
            continue
        sm = _find_softmax_chain_after_add(add_mask)
        if sm is None:
            continue
        # softmax div -> expand -> reshape -> bmm_1 (with v_view)
        expand_attn = next((u for u in sm["div"].users if _is_expand(u)), None)
        if expand_attn is None:
            continue
        view_attn = next((u for u in expand_attn.users if _is_reshape(u)), None)
        if view_attn is None:
            continue
        bmm1 = next((u for u in view_attn.users if _is_bmm(u)), None)
        if bmm1 is None:
            continue
        # bmm_1 -> reshape -> permute -> clone -> reshape (collapse heads) -> reshape -> mm_3
        view_o0 = next((u for u in bmm1.users if _is_reshape(u)), None)
        if view_o0 is None:
            continue
        permute_o = next((u for u in view_o0.users if _is_permute(u)), None)
        if permute_o is None:
            continue
        clone_o = next((u for u in permute_o.users if _is_clone(u)), None)
        # clone may be missing; permit either
        view_o1_src = clone_o if clone_o is not None else permute_o
        view_o1 = next((u for u in view_o1_src.users if _is_reshape(u)), None)
        if view_o1 is None:
            continue
        view_o2 = next((u for u in view_o1.users if _is_reshape(u)), None)
        if view_o2 is None:
            continue
        mm_o = next((u for u in view_o2.users if _is_mm(u)), None)
        if mm_o is None:
            continue

        # Walk back to find Q-side mm and K-side mm.
        if len(bmm0.args) < 2:
            continue
        q_view, k_view = bmm0.args[0], bmm0.args[1]
        if not isinstance(q_view, torch.fx.Node) or not isinstance(k_view, torch.fx.Node):
            continue
        # On the Q side the chain back goes through reshape -> expand -> mul(scale) -> permute -> permute -> reshape -> ...
        # On the K side the chain goes through reshape -> expand -> permute -> permute -> reshape -> permute -> clone -> expand -> unsqueeze -> index_put -> permute -> reshape -> cat -> ...
        # In all cases there's a Q/K mm somewhere upstream — find it by BFS through reshape/permute/clone/expand/mul/cat/unsqueeze/split/getitem/view_as_complex/view_as_real/index_put.
        mm_q = _bfs_back_to_mm(q_view, "q")
        mm_k = _bfs_back_to_mm(k_view, "k")
        if mm_q is None or mm_k is None:
            continue

        # V-side mm: trace back from bmm1.args[1].
        v_view = bmm1.args[1]
        mm_v = _bfs_back_to_mm(v_view, "v") if isinstance(v_view, torch.fx.Node) else None
        if mm_v is None:
            continue

        matches.append(
            {
                "mm_q": mm_q,
                "mm_k": mm_k,
                "mm_v": mm_v,
                "bmm0": bmm0,
                "view_qk": view_qk,
                "softcap_nodes": softcap_nodes,
                "add_mask": add_mask,
                "softmax": sm,
                "expand_attn": expand_attn,
                "view_attn": view_attn,
                "bmm1": bmm1,
                "view_o0": view_o0,
                "permute_o": permute_o,
                "clone_o": clone_o,
                "view_o1": view_o1,
                "view_o2": view_o2,
                "mm_o": mm_o,
            }
        )
    return matches


def _bfs_back_to_mm(start: torch.fx.Node, side: str, max_visits: int = 80) -> Optional[torch.fx.Node]:
    """BFS upstream through view-like / elementwise / index_put ops to find an
    ``aten.mm`` node. Stops at placeholders. ``side`` is informational."""
    visited: set = set()
    frontier: List[torch.fx.Node] = [start]
    while frontier and len(visited) < max_visits:
        n = frontier.pop()
        if n in visited:
            continue
        visited.add(n)
        if _is_mm(n):
            return n
        if not isinstance(n, torch.fx.Node):
            continue
        if n.op == "placeholder":
            continue
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                frontier.append(a)
            elif isinstance(a, (list, tuple)):
                for x in a:
                    if isinstance(x, torch.fx.Node):
                        frontier.append(x)
    return None


# ---------------------------------------------------------------------------
# Apply: slice Wq/Wo, replicate Wk/Wv, rewrite chain shapes
# ---------------------------------------------------------------------------
def _collect_q_chain(match: Dict[str, Any]) -> List[torch.fx.Node]:
    """Return every node from mm_q output forward up to and including bmm0,
    so we can resize meta and update reshape/expand shape literals."""
    mm_q = match["mm_q"]
    bmm0 = match["bmm0"]
    chain: List[torch.fx.Node] = []
    visited: set = set()
    frontier: List[torch.fx.Node] = [mm_q]
    while frontier:
        n = frontier.pop(0)
        if n in visited:
            continue
        visited.add(n)
        chain.append(n)
        if n is bmm0:
            continue
        for u in n.users:
            if u in visited:
                continue
            # Stop at K/V side: the K-side bmm0 input is collected via the K chain.
            if u is match.get("bmm1"):
                continue
            frontier.append(u)
    return chain


def _placeholder_behind_permute(mm: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Return the placeholder weight that feeds an ``mm(_, permute(W))``."""
    if len(mm.args) < 2:
        return None
    perm = mm.args[1]
    if not isinstance(perm, torch.fx.Node) or not _is_permute(perm):
        return None
    w = perm.args[0] if len(perm.args) >= 1 else None
    if isinstance(w, torch.fx.Node) and w.op == "placeholder":
        return w
    return None


def _compute_mm_ancestry(graph: torch.fx.Graph, mm_q, mm_k, mm_v) -> Dict[torch.fx.Node, frozenset]:
    """Per node, set of {q,k,v} ancestors. Mixed sets identify post-bmm nodes
    where Q's num_heads axis dominates; pure {k} / {v} sets identify pre-GQA-
    expand nodes where the num_kv_heads axis is what to rewrite."""
    tags: Dict[torch.fx.Node, set] = {}
    for n in graph.nodes:
        s: set = set()
        for a in n.all_input_nodes:
            s |= tags.get(a, set())
        if n is mm_q:
            s.add("q")
        if n is mm_k:
            s.add("k")
        if n is mm_v:
            s.add("v")
        tags[n] = s
    return {k: frozenset(v) for k, v in tags.items()}


def _apply_plan(
    graph: torch.fx.Graph,
    match: Dict[str, Any],
    *,
    world_size: int,
    plan_id: int,
    shard_id: int,
) -> bool:
    mm_q = match["mm_q"]
    mm_k = match["mm_k"]
    mm_v = match["mm_v"]
    mm_o = match["mm_o"]

    wq = _placeholder_behind_permute(mm_q)
    wk = _placeholder_behind_permute(mm_k)
    wv = _placeholder_behind_permute(mm_v)
    wo = _placeholder_behind_permute(mm_o)
    if wq is None or wk is None or wv is None or wo is None:
        _debug("could not resolve Wq/Wk/Wv/Wo placeholders behind permute")
        return False

    # Shape sanity. Wq: (q_size, hidden). Wk/Wv: (kv_size, hidden). Wo: (hidden, q_size).
    wq_shape = _shape(wq)
    wk_shape = _shape(wk)
    wv_shape = _shape(wv)
    wo_shape = _shape(wo)
    if any(s is None or len(s) != 2 for s in (wq_shape, wk_shape, wv_shape, wo_shape)):
        return False
    q_size, hidden_q = wq_shape
    kv_size, hidden_k = wk_shape
    kv_size_v, hidden_v = wv_shape
    hidden_o, q_size_o = wo_shape
    if q_size != q_size_o or hidden_q != hidden_o or hidden_q != hidden_k or hidden_q != hidden_v:
        _debug(
            f"hidden / q_size mismatch: Wq={wq_shape} Wk={wk_shape} Wv={wv_shape} Wo={wo_shape}"
        )
        return False
    if kv_size != kv_size_v:
        _debug(f"Wk/Wv kv_size mismatch: {kv_size} vs {kv_size_v}")
        return False
    if q_size % world_size != 0:
        return False
    shard_q = q_size // world_size

    # Infer num_heads/head_dim from mm_q output reshape chain.
    H_total: Optional[int] = None
    head_dim: Optional[int] = None
    for u in mm_q.users:
        if _is_reshape(u):
            for u2 in u.users:
                if _is_reshape(u2):
                    shp2 = _shape(u2)
                    if shp2 is not None and len(shp2) == 4 and shp2[-1] * shp2[-2] == q_size:
                        H_total = shp2[-2]
                        head_dim = shp2[-1]
                        break
            if H_total is not None:
                break
    if H_total is None or head_dim is None:
        _debug("could not infer num_heads / head_dim from mm_q chain; skipping")
        return False
    if H_total % world_size != 0:
        return False
    H_shard = H_total // world_size

    if kv_size % head_dim != 0:
        _debug(f"kv_size={kv_size} not divisible by head_dim={head_dim}")
        return False
    num_kv_heads = kv_size // head_dim

    # K/V column-sharding (Wk/Wv slice) requires the kv_cache placeholder to
    # also be sliced per NUMA node — without that, the in-graph index_put_ op
    # broadcasts a sharded src into a full-size dest and downstream GQA expand
    # fails. That cache-placeholder handling lives in wrapper codegen and is
    # not yet wired up; until then, gate K/V slicing behind an opt-in config
    # flag and default to replicated K/V (which is correct for num_kv_heads=1
    # and at worst leaves K/V unsharded for larger variants).
    slice_kv_requested = bool(getattr(config, "_numa_attention_tp_slice_kv", False))
    slice_kv = (
        slice_kv_requested
        and num_kv_heads >= world_size
        and num_kv_heads % world_size == 0
    )
    shard_kv_size = kv_size // world_size if slice_kv else kv_size
    shard_kv_heads = num_kv_heads // world_size if slice_kv else num_kv_heads

    softcap_present = bool(match.get("softcap_nodes"))
    _debug(
        f"match plan_id={plan_id}: q_size={q_size} kv_size={kv_size} hidden={hidden_q} "
        f"H_total={H_total} num_kv_heads={num_kv_heads} head_dim={head_dim} | "
        f"shard_q={shard_q} H_shard={H_shard} slice_kv={slice_kv} "
        f"shard_kv_heads={shard_kv_heads} compute_shard={shard_id}/{world_size} "
        f"softcap={softcap_present}"
    )

    # Sequence length S is also part of the chain shapes; if it happens to
    # equal q_size, shard_q, kv_size, or shard_kv_size, value-based rewriting
    # becomes ambiguous. Check this BEFORE mutating the graph so we leave
    # everything untouched on bail-out (otherwise mm_q gets rewired to sliced
    # weights but downstream view literals stay un-rewritten, producing a
    # half-broken graph that fails at runtime with shape mismatches).
    S_pre: Optional[int] = None
    for u in mm_q.users:
        if _is_reshape(u):
            shp = _shape(u)
            if shp is not None and len(shp) == 3 and shp[-1] == q_size:
                S_pre = shp[1]
                break
    if S_pre is not None and S_pre in (q_size, shard_q, kv_size, shard_kv_size):
        _debug(
            f"S_dim={S_pre} collides with q_size/shard_q/kv_size/shard_kv_size "
            f"({q_size}/{shard_q}/{kv_size}/{shard_kv_size}) — value-based rewrite "
            f"would corrupt shapes; skipping this match before any graph mutation"
        )
        return False

    # Per-shard slice -> permute nodes, anchored before mm_q.
    wq_shards: List[torch.fx.Node] = []
    wo_shards: List[torch.fx.Node] = []
    wk_shards: List[torch.fx.Node] = []
    wv_shards: List[torch.fx.Node] = []
    with graph.inserting_before(mm_q):
        for sid in range(world_size):
            start_q = sid * shard_q
            end_q = (sid + 1) * shard_q
            wq_slice = graph.call_function(aten.slice.Tensor, args=(wq, 0, start_q, end_q, 1))
            wq_slice.meta.update(wq.meta)
            wq_slice_perm = graph.call_function(aten.permute.default, args=(wq_slice, [1, 0]))
            _refresh_val(wq_slice_perm, (hidden_q, shard_q))
            wq_shards.append(wq_slice_perm)

            wo_slice = graph.call_function(aten.slice.Tensor, args=(wo, 1, start_q, end_q, 1))
            wo_slice.meta.update(wo.meta)
            wo_slice_perm = graph.call_function(aten.permute.default, args=(wo_slice, [1, 0]))
            _refresh_val(wo_slice_perm, (shard_q, hidden_o))
            wo_shards.append(wo_slice_perm)

            if slice_kv:
                start_kv = sid * shard_kv_size
                end_kv = (sid + 1) * shard_kv_size
                wk_slice = graph.call_function(aten.slice.Tensor, args=(wk, 0, start_kv, end_kv, 1))
                wk_slice.meta.update(wk.meta)
                wk_slice_perm = graph.call_function(aten.permute.default, args=(wk_slice, [1, 0]))
                _refresh_val(wk_slice_perm, (hidden_q, shard_kv_size))
                wk_shards.append(wk_slice_perm)

                wv_slice = graph.call_function(aten.slice.Tensor, args=(wv, 0, start_kv, end_kv, 1))
                wv_slice.meta.update(wv.meta)
                wv_slice_perm = graph.call_function(aten.permute.default, args=(wv_slice, [1, 0]))
                _refresh_val(wv_slice_perm, (hidden_q, shard_kv_size))
                wv_shards.append(wv_slice_perm)

    # Rewire compute-shard mms.
    mm_q.args = (mm_q.args[0], wq_shards[shard_id])
    mm_o.args = (mm_o.args[0], wo_shards[shard_id])
    if slice_kv:
        mm_k.args = (mm_k.args[0], wk_shards[shard_id])
        mm_v.args = (mm_v.args[0], wv_shards[shard_id])

        # K/V CACHE SLICING — PHASE 1 (FX-side groundwork; not end-to-end).
        #
        # When Wk/Wv are column-sharded, the K/V cache placeholders that the
        # in-graph index_copy_ writes into are still full-size. Slice them in
        # the graph (compute-shard slice) so shapes line up, and tag the slice
        # so a future wrapper-codegen change ("Phase 2") can emit per-socket
        # index_put calls writing to the disjoint kv-head slice each socket
        # owns. Without Phase 2 every socket would race-write to the compute
        # shard's slice instead of its own — wrong for decode and breaks the
        # downstream prefill reads of ``index_put`` result.
        #
        # This block only runs when ``slice_kv=True``, which itself requires
        # ``config._numa_attention_tp_slice_kv = True`` (default ``False``).
        # The flag stays gated off until Phase 2 lands in wrapper.py.
        KV_SHARD_DIM = 2   # cache shape: (B, max_S, num_kv_heads, head_dim)
        kv_start = shard_id * shard_kv_heads
        kv_end = (shard_id + 1) * shard_kv_heads
        for mm_node, role in ((mm_k, "k_cache"), (mm_v, "v_cache")):
            ip_node = _find_cache_index_put(mm_node)
            if ip_node is None:
                # No cache write in this graph (e.g. cacheless test harness).
                # That's fine — the sharded K/V chain just stays in-graph.
                _debug(f"slice_kv: no cache index_put downstream of {role[0]}m mm; skipping cache slice")
                continue
            cache_ph = ip_node.args[0]
            cache_shape = _shape(cache_ph)
            if cache_shape is None or len(cache_shape) != 4:
                _debug(f"slice_kv: cache {cache_ph.name} has unexpected shape {cache_shape}; skipping")
                continue
            with graph.inserting_before(ip_node):
                cache_slice = graph.call_function(
                    aten.slice.Tensor, args=(cache_ph, KV_SHARD_DIM, kv_start, kv_end, 1)
                )
            new_cache_shape = list(cache_shape)
            new_cache_shape[KV_SHARD_DIM] = shard_kv_heads
            _refresh_val(cache_slice, tuple(new_cache_shape))
            kv_cache_plan = {
                "full_cache_name": cache_ph.name,
                "shard_dim": KV_SHARD_DIM,
                "shard_kv_heads": shard_kv_heads,
                "num_kv_heads": num_kv_heads,
                "world_size": world_size,
                "compute_shard": shard_id,
                "role": role,
                "plan_id": plan_id,
            }
            cache_slice.meta["numa_kv_cache_plan"] = kv_cache_plan
            # Rewire index_put.args[0] = slice (also refresh its shape).
            ip_node.args = (cache_slice,) + tuple(ip_node.args[1:])
            ip_node.meta["numa_kv_cache_plan"] = kv_cache_plan
            ip_val = ip_node.meta.get("val")
            if isinstance(ip_val, torch.Tensor):
                _refresh_val(ip_node, tuple(new_cache_shape))
            # Rewire the companion copy_(cache_ph, ip) → copy_(cache_slice, ip).
            cp_node = _find_companion_copy_(cache_ph, ip_node)
            if cp_node is not None:
                cp_node.args = (cache_slice,) + tuple(cp_node.args[1:])
                cp_node.meta["numa_kv_cache_plan"] = kv_cache_plan
            _debug(
                f"slice_kv: sliced {role} {cache_ph.name} -> {cache_slice.name} "
                f"[dim={KV_SHARD_DIM}, {kv_start}:{kv_end}]; ip={ip_node.name}"
            )

    # Positional rewrite for the 4 bmm-input 3D reshapes
    # (bmm0.args[0..1] and bmm1.args[0..1]). Each has shape literal
    # [B*H_total, S, E] (for bmm input) or [B*H_total, S, S]. Value-based
    # H_total->H_shard catches the leading dim only when B=1 (where it
    # equals H_total exactly); for B>1 the leading literal is B*H_total
    # which the value-based rule does not match. Compute the new leading
    # dim positionally as (old // H_total) * H_shard. Safe to also run for
    # B=1 — produces the same result as the value-based rule.
    bmm_input_nodes: List[torch.fx.Node] = []
    for bmm_node in (match.get("bmm0"), match.get("bmm1")):
        if bmm_node is None:
            continue
        for inp in bmm_node.args[:2]:
            if isinstance(inp, torch.fx.Node) and _is_reshape(inp):
                bmm_input_nodes.append(inp)

    # Nodes whose shape was set positionally above must be skipped by the
    # value-based loop below; otherwise the H_total -> H_shard rule would
    # re-rewrite an already-correct dim. This collides specifically when
    # B*H_shard == H_total (i.e. B == world_size, e.g. B=2 with world_size=2).
    positionally_rewritten: set = set()

    for n in bmm_input_nodes:
        if len(n.args) < 2 or not isinstance(n.args[1], (list, tuple)):
            continue
        raw = list(n.args[1])
        if len(raw) != 3:
            continue
        d0 = raw[0]
        if not isinstance(d0, int) or d0 <= 0 or H_total <= 0:
            continue
        if d0 % H_total != 0:
            continue
        new_d0 = (d0 // H_total) * H_shard
        if new_d0 == d0:
            continue
        raw[0] = new_d0
        head = list(n.args[:1])
        tail = list(n.args[2:])
        n.args = tuple(head + [raw] + tail)
        sh = _shape(n)
        if sh is not None and len(sh) == 3:
            _refresh_val(n, (new_d0, sh[1], sh[2]))
        positionally_rewritten.add(n)

    # Tag nodes by mm-ancestry. Used to disambiguate which axis to rewrite
    # when shape literals contain values that would otherwise be confusable
    # (e.g. num_heads == num_queries_per_kv == 4 in num_kv_heads=1 GQA).
    tags = _compute_mm_ancestry(graph, mm_q, mm_k, mm_v)

    # Collect all chain nodes reachable forward from the three Q/K/V mms,
    # stopping at mm_o (inclusive).
    nodes_to_rewrite: List[torch.fx.Node] = []
    visited: set = set()
    frontier: List[torch.fx.Node] = [mm_q, mm_k, mm_v]
    while frontier:
        n = frontier.pop()
        if n in visited:
            continue
        visited.add(n)
        nodes_to_rewrite.append(n)
        if n is mm_o:
            continue
        for u in n.users:
            if u in visited:
                continue
            frontier.append(u)
    if mm_o not in visited:
        nodes_to_rewrite.append(mm_o)

    has_q_in = lambda t: "q" in t  # noqa: E731

    # The GQA expand_1 / expand_3 broadcast (1, S, num_kv_heads, 1, head_dim)
    # to (1, S, num_kv_heads, num_queries_per_kv, head_dim). Value-based
    # rewriting can't disambiguate position 2 (num_kv_heads) from position 3
    # (num_queries_per_kv) when those happen to be equal. Handle this 5D
    # expand positionally so both axes are always set correctly:
    #   new_kv_heads        = shard_kv_heads (= num_kv_heads if not sliced)
    #   new_queries_per_kv  = H_shard / new_kv_heads
    new_queries_per_kv = (
        H_shard // shard_kv_heads if shard_kv_heads > 0 else None
    )

    for n in nodes_to_rewrite:
        tag = tags.get(n, frozenset())
        is_kv_only = tag == frozenset({"k"}) or tag == frozenset({"v"}) or tag == frozenset({"k", "v"})

        # 5D GQA expand: positional rewrite of (kv_heads, queries_per_kv).
        if (
            _is_expand(n)
            and is_kv_only
            and new_queries_per_kv is not None
            and len(n.args) >= 2
            and isinstance(n.args[1], (list, tuple))
            and len(n.args[1]) == 5
        ):
            target = list(n.args[1])
            target[2] = shard_kv_heads
            target[3] = new_queries_per_kv
            head = list(n.args[:1])
            tail = list(n.args[2:])
            n.args = tuple(head + [target] + tail)
            _refresh_val(n, tuple(target))
            # Don't run the generic value-based rules below for this node —
            # they could re-rewrite an already-correct dim.
            continue

        # Same reasoning for the bmm-input 3D reshapes handled positionally
        # before this loop: skip the value-based H_total -> H_shard rule so
        # we don't re-rewrite (B*H_shard) -> H_shard when B*H_shard == H_total.
        if n in positionally_rewritten:
            continue

        if _is_reshape(n) or _is_expand(n):
            # q_size appears in q-only chain (post-mm_q reshape, pre-head-split)
            # AND in the post-bmm1 output collapse [1, S, q_size] and [S, q_size]
            # which have tag {q,k,v}. Rewrite whenever "q" is in the tag set.
            if has_q_in(tag):
                _replace_shape_arg(n, old_dim=q_size, new_dim=shard_q)
            # Pre-GQA-expand K/V chain: rewrite kv_size and num_kv_heads, but
            # only when we sliced Wk/Wv (otherwise K/V replicate, full literals
            # remain valid).
            if slice_kv and is_kv_only:
                _replace_shape_arg(n, old_dim=kv_size, new_dim=shard_kv_size)
                if num_kv_heads != shard_kv_heads:
                    _replace_shape_arg(n, old_dim=num_kv_heads, new_dim=shard_kv_heads)
            # num_heads rewrite applies everywhere (Q 4D shapes, post-GQA-expand
            # K/V, post-bmm0 softmax chain, post-bmm1 output collapse).
            _replace_shape_arg(n, old_dim=H_total, new_dim=H_shard)

        sh = _shape(n)
        if sh is None:
            continue
        new_sh: List[int] = []
        changed = False
        for d in sh:
            if has_q_in(tag) and d == q_size:
                new_sh.append(shard_q); changed = True; continue
            if slice_kv and is_kv_only and d == kv_size:
                new_sh.append(shard_kv_size); changed = True; continue
            if (
                slice_kv
                and is_kv_only
                and d == num_kv_heads
                and num_kv_heads != shard_kv_heads
            ):
                new_sh.append(shard_kv_heads); changed = True; continue
            if d == H_total:
                new_sh.append(H_shard); changed = True; continue
            new_sh.append(d)
        if changed:
            _refresh_val(n, tuple(new_sh))

    column_parallel_nodes: List[Tuple[str, int]] = [(wq.name, 0)]
    if slice_kv:
        column_parallel_nodes.append((wk.name, 0))
        column_parallel_nodes.append((wv.name, 0))
    replicated_nodes: List[str] = [] if slice_kv else [wk.name, wv.name]

    shard_plan: Dict[str, Any] = {
        "kind": "attention",
        "subkind": "gemma_gqa",
        "plan_id": plan_id,
        "world_size": world_size,
        "shard_d": shard_q,
        "compute_shard": shard_id,
        "x_node": _trace_x_placeholder(mm_q).name if _trace_x_placeholder(mm_q) else "",
        "column_parallel_nodes": column_parallel_nodes,
        "row_parallel_nodes": [(wo.name, 1)],
        "replicated_nodes": replicated_nodes,
        "zero_on_nonzero_nodes": [],
        "q_size": q_size,
        "kv_size": kv_size,
        "h": hidden_q,
        "num_heads": H_total,
        "num_kv_heads": num_kv_heads,
        "slice_kv": slice_kv,
        "head_dim": head_dim,
    }
    setattr(config, "_numa_attention_tp_shard_plan", dict(shard_plan))
    for n in itertools.chain(wq_shards, wo_shards, wk_shards, wv_shards, [mm_q, mm_k, mm_v, mm_o]):
        n.meta["numa_shard_plan"] = shard_plan
    return True


def _trace_x_placeholder(mm: torch.fx.Node) -> Optional[torch.fx.Node]:
    a0 = mm.args[0] if len(mm.args) >= 1 else None
    cur = a0
    for _ in range(4):
        if isinstance(cur, torch.fx.Node) and cur.op == "placeholder":
            return cur
        if isinstance(cur, torch.fx.Node) and _is_reshape(cur) and len(cur.args) >= 1:
            cur = cur.args[0]
            continue
        return None
    return None


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------
def numa_gemma_attention_tp_pass(graph: torch.fx.Graph) -> None:
    world_size = _get_world_size()
    _debug(f"pass start: world_size={world_size}")
    if world_size < 2:
        return
    matches = _find_attention_blocks(graph)
    _debug(f"Gemma attention candidates: {len(matches)}")
    if not matches:
        return
    plan_counter = itertools.count(start=1000)  # offset to avoid clashing with mlp/qkv plans
    changed = False
    for m in matches:
        plan_id = next(plan_counter)
        shard_id = _get_compute_shard(world_size)
        if _apply_plan(graph, m, world_size=world_size, plan_id=plan_id, shard_id=shard_id):
            changed = True
    if changed:
        graph.lint()
        _debug("pass done: graph updated")
