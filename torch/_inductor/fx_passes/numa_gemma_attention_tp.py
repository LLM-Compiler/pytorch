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
        # bmm output -> reshape -> add(mask) -> softmax chain
        view_qk = next((u for u in bmm0.users if _is_reshape(u)), None)
        if view_qk is None:
            continue
        add_mask = next((u for u in view_qk.users if _is_call(u, aten.add.Tensor)), None)
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


def _apply_plan(
    graph: torch.fx.Graph,
    match: Dict[str, Any],
    *,
    world_size: int,
    plan_id: int,
    shard_id: int,
) -> bool:
    mm_q = match["mm_q"]
    mm_o = match["mm_o"]

    # Wq placeholder: arg of permute that feeds mm_q at index 1.
    wq_perm = mm_q.args[1] if len(mm_q.args) >= 2 else None
    if not isinstance(wq_perm, torch.fx.Node) or not _is_permute(wq_perm):
        return False
    wq = wq_perm.args[0] if len(wq_perm.args) >= 1 else None
    if not isinstance(wq, torch.fx.Node) or wq.op != "placeholder":
        return False

    wo_perm = mm_o.args[1] if len(mm_o.args) >= 2 else None
    if not isinstance(wo_perm, torch.fx.Node) or not _is_permute(wo_perm):
        return False
    wo = wo_perm.args[0] if len(wo_perm.args) >= 1 else None
    if not isinstance(wo, torch.fx.Node) or wo.op != "placeholder":
        return False

    # Shape sanity. Wq: (q_size, hidden). Wo: (hidden, q_size).
    wq_shape = _shape(wq)
    wo_shape = _shape(wo)
    if wq_shape is None or wo_shape is None or len(wq_shape) != 2 or len(wo_shape) != 2:
        return False
    q_size, hidden_q = wq_shape
    hidden_o, q_size_o = wo_shape
    if q_size != q_size_o or hidden_q != hidden_o:
        _debug(f"Wq/Wo shape mismatch: Wq={wq_shape} Wo={wo_shape}; skipping")
        return False
    if q_size % world_size != 0:
        _debug(f"q_size={q_size} not divisible by world_size={world_size}; skipping")
        return False
    shard_q = q_size // world_size

    # Inferred num_heads from a downstream view that reshapes to [B, S, H, E].
    # mm_q output (s, q_size) -> reshape [1, S, q_size] -> reshape [1, S, H, E] -> permute.
    H_total: Optional[int] = None
    head_dim: Optional[int] = None
    for u in mm_q.users:
        if _is_reshape(u):
            shp = _shape(u)
            if shp is not None and len(shp) == 3:
                # Then look forward for a 4D reshape.
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
        _debug(f"num_heads={H_total} not divisible by world_size={world_size}; skipping")
        return False
    H_shard = H_total // world_size

    _debug(
        f"match plan_id={plan_id}: q_size={q_size} hidden={hidden_q} "
        f"H_total={H_total} head_dim={head_dim} "
        f"shard_q={shard_q} H_shard={H_shard} compute_shard={shard_id}/{world_size}"
    )

    # Build per-shard slice -> permute for Wq and Wo.
    wq_shards: List[torch.fx.Node] = []
    wo_shards: List[torch.fx.Node] = []
    with graph.inserting_before(mm_q):
        for sid in range(world_size):
            start = sid * shard_q
            end = (sid + 1) * shard_q
            wq_slice = graph.call_function(aten.slice.Tensor, args=(wq, 0, start, end, 1))
            wq_slice.meta.update(wq.meta)
            wq_slice.meta["numa_plan_id"] = plan_id
            wq_slice.meta["numa_shard_id"] = sid
            wq_slice_perm = graph.call_function(aten.permute.default, args=(wq_slice, [1, 0]))
            _refresh_val(wq_slice_perm, (hidden_q, shard_q))
            wq_shards.append(wq_slice_perm)

            wo_slice = graph.call_function(aten.slice.Tensor, args=(wo, 1, start, end, 1))
            wo_slice.meta.update(wo.meta)
            wo_slice.meta["numa_plan_id"] = plan_id
            wo_slice.meta["numa_shard_id"] = sid
            wo_slice_perm = graph.call_function(aten.permute.default, args=(wo_slice, [1, 0]))
            _refresh_val(wo_slice_perm, (shard_q, hidden_o))
            wo_shards.append(wo_slice_perm)

    # Rewire compute-shard mm_q and mm_o to per-shard permutes.
    mm_q.args = (mm_q.args[0], wq_shards[shard_id])
    mm_o.args = (mm_o.args[0], wo_shards[shard_id])

    # Walk every reshape/expand in the activation chain (Q side, then through
    # K-broadcast, the bmm output chain, the softmax chain, V-broadcast, and
    # the output collapse) and rewrite shape literals from full → sharded.
    nodes_to_rewrite: List[torch.fx.Node] = []
    visited: set = set()
    frontier: List[torch.fx.Node] = [mm_q, match["mm_k"], match["mm_v"]]
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
    nodes_to_rewrite.append(mm_o)

    for n in nodes_to_rewrite:
        # Shape-arg rewrite for reshape and expand.
        if _is_reshape(n) or _is_expand(n):
            _replace_shape_arg(n, old_dim=q_size, new_dim=shard_q)
            _replace_shape_arg(n, old_dim=H_total, new_dim=H_shard)
        # Refresh meta['val']: any dim equal to q_size or H_total becomes sharded.
        sh = _shape(n)
        if sh is None:
            continue
        new_sh: List[int] = []
        changed = False
        for d in sh:
            if d == q_size:
                new_sh.append(shard_q)
                changed = True
            elif d == H_total:
                new_sh.append(H_shard)
                changed = True
            else:
                new_sh.append(d)
        if changed:
            _refresh_val(n, tuple(new_sh))

    # mm_o output stays at (s, hidden) — no resize needed.
    # Build a shard plan (consumed by wrapper codegen for per-NUMA clones).
    shard_plan: Dict[str, Any] = {
        "kind": "attention",
        "subkind": "gemma_gqa",
        "plan_id": plan_id,
        "world_size": world_size,
        "shard_d": shard_q,
        "compute_shard": shard_id,
        "x_node": _trace_x_placeholder(mm_q).name if _trace_x_placeholder(mm_q) else "",
        "column_parallel_nodes": [(wq.name, 0)],
        "row_parallel_nodes": [(wo.name, 1)],
        "replicated_nodes": [],
        "zero_on_nonzero_nodes": [],
        "q_size": q_size,
        "h": hidden_q,
        "num_heads": H_total,
        "head_dim": head_dim,
    }
    setattr(config, "_numa_attention_tp_shard_plan", dict(shard_plan))
    for n in itertools.chain(wq_shards, wo_shards, [mm_q, mm_o]):
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
