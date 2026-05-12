# NUMA Tensor-Parallel Gemma Prefill — End-to-End Documentation

Complete reference for the CPU-only NUMA tensor-parallel (TP) inference
acceleration implemented for the Gemma model family via Inductor FX passes +
Triton CPU launcher thread pinning.

---

## 1. Goal

Accelerate Gemma prefill on a **multi-socket CPU** by sharding attention and
MLP weights across NUMA sockets, with each socket running its compute path
pinned to socket-local CPUs and reading socket-local weight clones.

**Verified end-to-end speedups (world_size=2, S=4096):**

| Model | Layers | Config | Speedup |
|---|---|---|---|
| Gemma-270M | 18 | default (K/V replicated) | **1.366×** |
| Gemma-270M | 1 | default | 1.535–1.582× |
| Gemma-4B | 1 | K/V replicated, cache-free | 1.380× |
| Gemma-4B | 1 | **K/V sharded**, cache-free | 1.312× |
| Gemma 9B/27B style | 1 | softcap=50.0, cache | 1.644× |
| GQA num_kv_heads=2 | 1 | default | 1.538× |
| QK-norm + kv=1 | 1 | default | 1.509× |

---

## 2. End-to-end pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  USER CODE                                                                   │
│  ─────────                                                                   │
│  torch._C._cpu._set_numa_enabled(True)                                       │
│  config._numa_attention_tp = True                                            │
│  config._numa_mlp_tp       = True                                            │
│  config._numa_attention_tp_slice_kv = True   ← opt-in K/V sharding           │
│  model = SimpleGemmaStack(num_layers=18, use_kv_cache=False, ...)            │
│  compiled = torch.compile(model)                                             │
│  out = compiled(input_ids, freqs_cis, None, None, mask)                      │
└──────────────────────────────────────────┬───────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  DYNAMO + AOT AUTOGRAD                                                       │
│  ────────────────────                                                        │
│  • Traces all L layers into a single flat FX graph                           │
│  • Lifts each nn.Parameter to a distinct placeholder                         │
│  • Decomposes high-level ops:                                                │
│      F.linear     → reshape · permute · mm                                   │
│      F.gelu(tanh) → 9-op chain (mul · mul · mul · 0.044715 · + · 0.7978...   │
│                      · tanh · +1 · mul)                                      │
│      SDPA(MATH)   → q·scale @ kᵀ · +mask · amax · sub · exp · sum · div      │
│      RoPE         → view_as_complex · mul(freqs) · view_as_real cat chain    │
│      GQA expand   → unsqueeze · expand · clone · reshape                     │
│      Softcap      → div(cap) · tanh · mul(cap)  (Gemma 9B/27B)               │
│      QK-norm      → 4-op RMSNorm chain on Q and K                            │
└──────────────────────────────────────────┬───────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  INDUCTOR POST-GRAD FX PASSES                                                │
│  ────────────────────────────                                                │
│  Registered in torch/_inductor/fx_passes/post_grad.py order:                 │
│    1. numa_attention_tp_pass         (GPT-2 SDPA-style; skips Gemma)         │
│    2. numa_gemma_attention_tp_pass   ← Gemma split-QKV / GQA / manual softmax│
│    3. numa_mlp_tp_pass               ← GeGLU and erf-GELU                    │
│    4. numa_layer_norm_tp_pass        (row-parallel reassembly)               │
│    5. numa_residual_tp_pass          (residual add absorbed into TP region)  │
│                                                                              │
│  For each match (per attention block, per MLP block):                        │
│    • insert per-socket aten.slice + aten.permute nodes                       │
│    • rewire mm operands to compute-shard's slice                             │
│    • refresh meta['val'] + reshape/expand shape literals (positional + value)│
│    • slice K/V cache placeholders (when slice_kv=True and cache present)     │
│    • tag rewired nodes with node.meta['numa_shard_plan']                     │
└──────────────────────────────────────────┬───────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  INDUCTOR LOWERING + WRAPPER CODEGEN                                         │
│  ───────────────────────────────────                                         │
│  PythonWrapperCodegen._emit_numa_kernel_call() reads numa_shard_plan, emits: │
│                                                                              │
│    # Column-parallel weight (per-node shard size from 3-tuple)               │
│    buf_n0 = empty_strided(...).clone(W[0:shard_size, :],   numa_node=0)      │
│    buf_n1 = empty_strided(...).clone(W[shard_size:, :],    numa_node=1)      │
│    kernel.run(..., numa_args=[[buf_n0, ...], [buf_n1, ...]])                 │
│                                                                              │
│  Per-(plan_id, target_node) cache via self._numa_tensor_cache: clone runs    │
│  once per (layer, socket) on the first prefill call, reused thereafter.      │
└──────────────────────────────────────────┬───────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  TRITON CPU LAUNCHER  (external repo: triton-cpu-private)                    │
│  ───────────────────                                                         │
│  third_party/cpu/backend/driver.py::make_launcher() generated C extension:   │
│    if kernel_metadata.numa_args present:                                     │
│      for node in numa_nodes:                                                 │
│        cpus = read /sys/devices/system/node/nodeN/cpulist                    │
│        for t in thread_group[node]:                                          │
│          std::thread([&]{                                                    │
│            pthread_setaffinity_np(self, cpus)   ← pin to socket              │
│            run_kernel_with(per_node_ptrs[node]) ← socket-local weights       │
│          })                                                                  │
│    else: fall back to run_omp_kernels()                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Per-Gemma-variant parallelization (N=2 sockets)

| Model | num_heads | num_kv_heads | Wq | Wk, Wv (slice_kv=False) | Wk, Wv (slice_kv=True) | Wo | MLP | Softcap | QK-norm |
|---|---|---|---|---|---|---|---|---|---|
| 270M | 4 | 1 | col-split 2+2 | replicated (forced — 1 head) | replicated (cannot split) | row-split | full TP | no | optional |
| 1B | 4 | 1 | col-split 2+2 | replicated | replicated | row-split | full TP | no | optional |
| 4B | 8 | 4 | col-split 4+4 | replicated by default | **col-split 2+2 KV heads** | row-split | full TP | no | optional |
| 9B | 16 | 8 | col-split 8+8 | replicated by default | **col-split 4+4 KV heads** | row-split | full TP | **yes** | optional |
| 12B | 16 | 8 | col-split 8+8 | replicated by default | **col-split 4+4 KV heads** | row-split | full TP | yes | optional |
| 27B | 32 | 16 | col-split 16+16 | replicated by default | **col-split 8+8 KV heads** | row-split | full TP | **yes** | optional |

`slice_kv=True` requires:
- `config._numa_attention_tp_slice_kv = True`
- `num_kv_heads >= world_size` and `num_kv_heads % world_size == 0`
- **No in-graph KV cache writes** (`use_kv_cache=False` in the model, or no
  cache placeholders in the traced graph). The cache + sharded-K/V combo
  triggers an Inductor fused index_copy stride bug.

---

## 4. Per-layer parallelization diagram (concrete dims for 4B with slice_kv=True)

`hidden=2560, q_size=2048, kv_size=1024, num_heads=8, num_kv_heads=4, head_dim=256, intermediate=10240`, N=2.

```
                            INPUT x : (B, S, 2560)   REPLICATED on both sockets
                                     │
                                     ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    ATTENTION BLOCK                              │
   │                                                                 │
   │   Wq (2048, 2560) ──► COL-SPLIT (output rows)                   │
   │                    [0:1024, :]  on NODE 0                       │
   │                    [1024:2048, :] on NODE 1                     │
   │                                                                 │
   │   Wk (1024, 2560) ──► COL-SPLIT (slice_kv=True)                 │
   │                    [0:512, :]  on NODE 0   (KV heads 0–1)       │
   │                    [512:1024, :] on NODE 1 (KV heads 2–3)       │
   │                                                                 │
   │   Wv (1024, 2560) ──► same as Wk                                │
   │                                                                 │
   │                          NODE 0                NODE 1           │
   │                          ──────                ──────           │
   │   mm_q   :  x @ Wq.T     (B,S,1024)            (B,S,1024)       │
   │   mm_k   :  x @ Wk.T     (B,S,512)             (B,S,512)        │
   │   mm_v   :  x @ Wv.T     (B,S,512)             (B,S,512)        │
   │                                                                 │
   │   reshape: q (B,S,4,256)  k (B,S,2,256)  v (B,S,2,256)          │
   │            (H_shard=4 Q heads, shard_kv_heads=2 KV heads each)  │
   │                                                                 │
   │   QK-norm (optional) ── per-socket, head-local                  │
   │   RoPE on q,k        ── per-socket, head-local                  │
   │                                                                 │
   │   GQA expand 5D pos-rewrite:                                    │
   │     (B,S,2,1,256) → (B,S,2,2,256)                               │
   │     positions [2]=shard_kv_heads=2, [3]=queries_per_kv=2        │
   │                                                                 │
   │   bmm0 :  q @ kᵀ + softcap (Gemma 9B/27B) + mask + softmax       │
   │           [B*4, S, S]      per-socket                           │
   │                                                                 │
   │   bmm1 :  attn @ v         [B*4, S, 256]   per-socket           │
   │                                                                 │
   │   reshape collapse: (B,S,4,256) → (B,S,1024)  per-socket        │
   │                                                                 │
   │   Wo (2560, 2048) ──► ROW-SPLIT (input cols)                    │
   │                    [:, 0:1024]   on NODE 0                      │
   │                    [:, 1024:2048] on NODE 1                     │
   │                                                                 │
   │   mm_o : partial(B,S,2560)   partial(B,S,2560)                  │
   │                                                                 │
   │                   ┌────────┴─────────┐                          │
   │                   │   REASSEMBLY     │  sum partials            │
   │                   │   (sum / concat) │  ─ full (B,S,2560)       │
   │                   └────────┬─────────┘                          │
   └────────────────────────────│────────────────────────────────────┘
                                ▼
                         residual add → full hidden
                                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    MLP (GeGLU) BLOCK                            │
   │                                                                 │
   │  gate (10240, 2560) ──► COL-SPLIT  (5120, 2560) per socket      │
   │  up   (10240, 2560) ──► COL-SPLIT  (5120, 2560) per socket      │
   │                                                                 │
   │   mm_gate, mm_up    (B,S,5120) per socket                       │
   │   gelu_tanh(gate) · up    per-socket                            │
   │                                                                 │
   │  down (2560, 10240) ──► ROW-SPLIT (2560, 5120) per socket       │
   │                                                                 │
   │   mm_down : partial(B,S,2560) per socket                        │
   │                                                                 │
   │                   ┌────────┴─────────┐                          │
   │                   │   REASSEMBLY     │  sum partials            │
   │                   └────────┬─────────┘                          │
   └────────────────────────────│────────────────────────────────────┘
                                ▼
                       residual add → full hidden → next layer
```

---

## 5. Column vs row parallelism (Megatron-style)

### Column-parallel (W sliced along output dim 0)
```
Activation x:    full (B,S,H_in)  on both sockets
Weight W:        (H_out, H_in) → slice rows by per-node shard size
mm output:       (B,S, H_out/N or shard_kv_size)   per socket
Communication:   NONE — sockets stay independent
```

### Row-parallel (W sliced along input dim 1)
```
Activation x:    each socket has (B,S, H_in/N)  ← from upstream column-parallel
Weight W:        (H_out, H_in) → slice cols by per-node shard size
mm output:       (B,S, H_out)   per socket  ← partial sum
Communication:   sum-reduce across sockets to get final result
```

The Megatron pattern alternates **column → row** so the column-parallel sharded
output flows directly into the row-parallel input without communication. Only
**one reassembly per stack** (attention `mm_o`, MLP `mm_down`).

### Per-block role assignment

| Block | Column-parallel | Row-parallel | Replicated (default) |
|---|---|---|---|
| Attention | Wq (shard_q = q_size/N) | Wo (shard_q) | Wk, Wv |
| Attention (slice_kv=True) | Wq (shard_q), **Wk, Wv (shard_kv_size)** | Wo (shard_q) | — |
| MLP (GeGLU) | gate, up (intermediate/N) | down (intermediate/N) | — |

---

## 6. Shard plan schema

Every rewired node carries `node.meta["numa_shard_plan"]`:

```python
{
  "kind": "mlp" | "attention",
  "subkind": "geglu" | "gemma_gqa",
  "plan_id": int,                                   # unique per block
  "world_size": int,
  "shard_d": int,                                   # plan-wide default chunk
  "compute_shard": int,                             # which socket runs compute
  "x_node": str,                                    # replicated input

  # PER-NODE SHARD SIZE (3-tuple form, added in this branch)
  # Wq uses shard_q; Wk/Wv use shard_kv_size under slice_kv=True (GQA with
  # num_kv_heads < num_heads).
  "column_parallel_nodes": List[Tuple[str, int, int]],  # (name, dim, shard_size)
  "row_parallel_nodes":    List[Tuple[str, int, int]],

  "replicated_nodes":      List[str],
  "zero_on_nonzero_nodes": List[str],

  # Attention-specific:
  "q_size": int, "kv_size": int,
  "num_heads": int, "num_kv_heads": int, "head_dim": int,
  "slice_kv": bool,

  # MLP-specific:
  "h": int, "d": int,
}
```

**Backward-compat:** the wrapper still accepts 2-tuples `(name, dim)` from
legacy passes (`numa_attention_tp.py`, `numa_mlp_tp.py`) and falls back to the
plan-wide `shard_d` for those.

The K/V cache slice nodes carry a separate `numa_kv_cache_plan` meta (set by
Phase 1 cache-slicing code, still in the tree as documented groundwork even
though we now prefer cache-free mode for slice_kv).

---

## 7. Pattern matching details

### Gemma MLP (GeGLU) — `numa_mlp_tp.py::_find_geglu_quintuples`

Post-grad pattern:
```
view      = reshape(x, [s, h])
gate_mm   = mm(view, permute(gate_w))             # (s, d)
view_1    = reshape(gate_mm, [B, S, d])
mul_4     = view_1 * 0.5                          # GELU(tanh) 9-op chain begins
mul       = view_1 * view_1
mul_1     = mul * view_1
mul_2     = mul_1 * 0.044715
add       = view_1 + mul_2
mul_3     = add * 0.7978845608028654
tanh      = tanh(mul_3)
add_1     = tanh + 1
mul_5     = mul_4 * add_1                         # GELU(tanh)(gate)
view_2    = reshape(x, [s, h])
up_mm     = mm(view_2, permute(up_w))
view_3    = reshape(up_mm, [B, S, d])
mul_6     = mul_5 * view_3
view_4    = reshape(mul_6, [s, d])
down_mm   = mm(view_4, permute(down_w))           # (s, h)
view_5    = reshape(down_mm, [B, S, h])
```

### Gemma attention — `numa_gemma_attention_tp.py::_find_attention_blocks`

Anchored on `bmm0` (q@kᵀ), walks deterministically forward:
1. `bmm0 → reshape (view_qk)`
2. **Optional softcap chain** (Gemma 9B/27B): `view_qk → div(cap) → tanh → mul(cap)`
3. `→ add(mask)`
4. Manual softmax: `add → amax → sub → exp → sum → div`
5. Softmax `div → expand → reshape → bmm_1` (with v_view)
6. Output collapse: `bmm_1 → reshape → permute → clone? → reshape → reshape → mm_o`

Q/K/V mms found via BFS upstream from `bmm0.args` / `bmm1.args` through
view-like ops, RoPE chain, GQA expand, optional QK-norm.

The matcher requires the model to use **separate `Wq`/`Wk`/`Wv`** parameters
(no fused `qkv_proj`). The demo splits Gemma's reference fused QKV into three
linears for this reason.

---

## 8. Shape-rewrite mechanics

After mm operands are rewired, every downstream `reshape`/`expand` node has
**stale shape literals** baked into `args[1]`. The pass walks the chain forward
from rewired mms via BFS, and for each node:

1. **Tag with mm-ancestry** (`{q}`, `{k}`, `{v}`, `{q,k,v}`, etc.) so the
   right axis is rewritten when literal values collide (e.g. `num_heads ==
   num_queries_per_kv == 4` when `num_kv_heads=1` GQA).
2. **Positional rewrite first** for the 4 bmm-input 3D reshapes
   `[B*H_total, S, E]`. The value `B*H_total` never matches any standalone
   dim, so it must be rewritten by axis position. Tracked in a
   `positionally_rewritten` set so the value-based loop skips them
   (otherwise double-rewrite when `B*H_shard == H_total`).
3. **Value-based rewrite second**: `q_size → shard_q`, `H_total → H_shard`,
   `kv_size → shard_kv_size` (when sliced), `num_kv_heads → shard_kv_heads`.
4. **5D GQA expand positional rewrite**: positions `[2]` and `[3]` set
   explicitly to `(shard_kv_heads, H_shard/shard_kv_heads)`.
5. **Refresh `meta['val']`** by computing the new tensor metadata.

---

## 9. K/V cache: replicated (default) vs sharded (opt-in)

### Default: K/V replicated (`slice_kv=False`)
```
Wk, Wv full → cloned to every socket → mm_k/mm_v produce full xk/xv on each
socket → fed into GQA expand → bmm
```
- Each socket does **identical K/V mm work** (duplicated, no compute savings)
- But: both sockets run in parallel → no extra wall time vs single-socket K/V
- Required for `num_kv_heads < world_size` (can't split a single head)
- Default for ALL Gemma models today

### Opt-in: K/V sharded (`slice_kv=True`)
```
Wk, Wv sliced along dim 0 by shard_kv_size = kv_size // N
mm_k/mm_v produce (B,S, shard_kv_size) per socket
shape rewrites: kv_size → shard_kv_size, num_kv_heads → shard_kv_heads
GQA expand: positionally rewritten to (shard_kv_heads, queries_per_kv)
```
- Each socket does **half the K/V mm work** → real compute speedup
- Each socket reads only its own kv_heads from local memory
- Required: `num_kv_heads >= N`, `num_kv_heads % N == 0`
- **Required: no in-graph KV cache** (use `use_kv_cache=False` in the demo model)

### Why K/V sharding requires cache-free mode

The in-graph `index_copy_(k_cache, idx, xk)` becomes (after AOT functionalize):
```python
index_put = aten.index_put.default(k_cache_placeholder, [None, idx], xk)
copy_ = aten.copy_.default(k_cache_placeholder, index_put)
```

Phase 1 of K/V sharding slices the cache placeholder so shapes line up:
```python
slice = aten.slice.Tensor(arg8_1_kcache, dim=2, start=0, end=shard_kv_heads, step=1)
index_put = aten.index_put.default(slice, [None, idx], xk_sharded)
copy_(slice, index_put)
```

**Inductor's fused `index_copy` codegen has a stride bug:** it computes the
kernel grid for the SLICED shape but passes the FULL placeholder pointer →
out-of-bounds writes → SIGSEGV.

The workaround: skip the cache entirely. With `use_kv_cache=False`, the FX
graph has no cache placeholders and no `index_put_`; the K/V sharded path
goes mm → shape rewrite → bmm directly. Verified working at 4B.

---

## 10. Cache-free mode (new in this branch)

`SimpleGemmaStack(use_kv_cache=False)` skips:
- KV cache placeholders (`k_cache`, `v_cache`)
- `kv_write_indices` placeholder
- `index_copy_` writes

The forward becomes:
```python
xq = F.linear(hidden, Wq).view(B, S, num_heads, head_dim)
xk = F.linear(hidden, Wk).view(B, S, num_kv_heads, head_dim)
xv = F.linear(hidden, Wv).view(B, S, num_kv_heads, head_dim)
if use_qk_norm: xq, xk = q_norm(xq), k_norm(xk)
xq = apply_rotary_emb(xq, freqs_cis)
xk = apply_rotary_emb(xk, freqs_cis)
key, value = xk, xv             # ← no cache; use xk/xv directly
# ... GQA expand → bmm → softmax → bmm → Wo ...
```

The speedup CLI uses this mode by default
(`use_kv_cache=False` passed at construction). The demo CLI keeps caches
enabled for fidelity with the reference Gemma forward.

---

## 11. Multi-layer scaling

There is **no special multi-layer logic**. The FX matchers operate at the
op-pattern level, so they fire independently on each block they find. For
L=18:

- Each `_find_*` call returns 18 matches (one per block).
- Each match gets its own `plan_id` (1000, 1001, …, 1017 for attention;
  0, 1, …, 17 for MLP).
- Each block's NUMA clones are keyed independently in `_numa_tensor_cache`.

Depth is effectively unbounded; the implementation scales to any number of
layers Dynamo can trace.

---

## 12. Multi-batch (B > 1) support

The 4 bmm-input 3D reshapes (`bmm0.args[0..1]`, `bmm1.args[0..1]`) have shape
`[B*H_total, S, E]`. The value `B*H_total` never matches `H_total` as a
standalone dim, so the value-based rewriter would miss it for B > 1. Fixed
via a positional pre-pass that rewrites the leading dim as
`(old_d0 // H_total) * H_shard`. Skipped in the value-based loop to avoid
double-rewrite when `B == world_size` (e.g. B=2, world_size=2 would otherwise
re-rewrite `4 → 2`).

Verified at B ∈ {1, 2, 4, 8}.

---

## 13. S-collision guard

The value-based rewriter relies on dim values being distinct. When the
sequence length `S` equals any of `{q_size, shard_q, kv_size, shard_kv_size}`,
the rewrite becomes ambiguous. The pass detects this **before mutation** and
bails:

```python
if S_dim in (q_size, shard_q, kv_size, shard_kv_size):
    _debug(f"S_dim={S_dim} collides — skipping match")
    return False  # graph untouched, std path takes over
```

For Gemma-270M (`q_size=1024, shard_q=512, kv_size=256`) this trips at
S ∈ {256, 512, 1024}. Real prefill at S ≥ 2048 is unaffected.

---

## 14. Reassembly (the all-reduce)

After row-parallel `mm_o` (attention) and `mm_down` (MLP), partials on
both sockets must combine. `numa_layer_norm_tp_pass` /
`numa_residual_tp_pass` detect the row-parallel mm output (tagged with
`row_parallel_nodes` in its shard plan) and emit:

- **SUM geometry** (`numa_output_geometry="sum"`): elementwise sum of partials
- **CONCAT geometry**: M-split concatenation (different consumer pattern)

Reassembly is a **single serial thread group** at the end of each block,
which caps per-block speedup around 1.5–1.6×.

---

## 15. Configuration flags

```python
import torch._inductor.config as config

# Enable passes
config._numa_mlp_tp = True
config._numa_attention_tp = True

# Override NUMA world size (number of sockets) for testing
config._numa_mlp_tp_world_size = 2
config._numa_attention_tp_world_size = 2

# Which shard runs the compute path (0-indexed)
config._numa_mlp_tp_compute_shard = 0
config._numa_attention_tp_compute_shard = 0

# Opt-in K/V column-sharding (requires cache-free model)
config._numa_attention_tp_slice_kv = False  # set True for 4B+ cache-free

# Debug prints
config._numa_mlp_tp_debug = True
config._numa_attention_tp_debug = True
```

---

## 16. Pattern features matched

| Feature | Status | Notes |
|---|---|---|
| Split Wq/Wk/Wv (no fused qkv) | ✅ | Required model shape — demo splits Gemma's fused qkv |
| Fused qkv_proj | ❌ | Not matched; would need three-region placeholder slicing |
| Manual softmax (`add(mask) → amax → sub → exp → sum → div`) | ✅ | Anchor for the pass |
| SDPA via SDPBackend.MATH | ✅ | Different pass (`numa_attention_tp.py`) for GPT-2 |
| GQA (num_kv_heads < num_heads) | ✅ | Handled via 5D GQA-expand positional rewrite |
| QK-norm | ✅ | Walks through per-head RMSNorm on Q and K |
| Attn-logit softcapping (9B/27B) | ✅ | Optional `div → tanh → mul` chain between bmm0 and add(mask) |
| GeGLU MLP (Gemma) | ✅ | Tanh-approx GELU 9-op chain matcher |
| erf-GELU MLP (GPT-2) | ✅ | Legacy matcher in same pass |
| KV cache writes (index_copy_) | ✅ replicated mode; ⚠️ sharded mode requires cache-free | Inductor fused index_copy bug |
| Multi-batch B > 1 | ✅ | Positional bmm-input rewrite |
| Multi-layer L > 1 | ✅ | Matcher is local, scales to any depth |
| RoPE | ✅ | view_as_complex chain — shapes refresh through it |

---

## 17. Active files

### Inductor FX passes — core feature code

| File | Role |
|---|---|
| `torch/_inductor/fx_passes/numa_attention_tp.py` | GPT-2 SDPA-style attention pass (Q/K/V/Wo) |
| `torch/_inductor/fx_passes/numa_gemma_attention_tp.py` | **Gemma split-QKV + GQA + manual softmax pass** (this work) |
| `torch/_inductor/fx_passes/numa_mlp_tp.py` | MLP pass (erf-GELU + **GeGLU patterns** from this work) |
| `torch/_inductor/fx_passes/numa_tp_common.py` | Shared helpers for shard-plan schema |
| `torch/_inductor/fx_passes/numa_layer_norm_tp.py` | Row-parallel reassembly (SUM / CONCAT) |
| `torch/_inductor/fx_passes/numa_residual_tp.py` | Residual-add reassembly geometry |
| `torch/_inductor/fx_passes/post_grad.py` | Registers all passes in pipeline |
| `torch/_inductor/codegen/wrapper.py` | Emits per-NUMA `clone()`, `numa_args` list, **per-node shard sizes** |
| `torch/_inductor/config.py` | NUMA TP flags including `_numa_attention_tp_slice_kv` |

### ATen / core extension

| File | Role |
|---|---|
| `aten/src/ATen/native/TensorFactories.cpp` | `clone(..., numa_node=N)` via `c10::NUMAMove` |
| `aten/src/ATen/native/native_functions.yaml` | Schema: `clone(..., int? numa_node=None)` |
| `torch/csrc/cpu/Module.cpp` | Python bindings: `_is_numa_enabled`, `_num_numa_nodes` |

### External: Triton CPU launcher

| File | Role |
|---|---|
| `triton-cpu-private/third_party/cpu/backend/driver.py` | `make_launcher()` generates C extension that pins `std::thread`s per socket |

### Demo and benchmark scripts

| File | Purpose |
|---|---|
| `scripts/inductor/simple_gemma_decoder_block_torch_compile_demo.py` | Standard `torch.compile` baseline; **now with `use_kv_cache=False` mode** |
| `scripts/inductor/simple_gemma_decoder_block_numa_tp_torch_compile_demo.py` | NUMA-TP enabled run |
| `scripts/inductor/simple_gemma_decoder_block_speedup_cli.py` | Side-by-side timing comparison; **uses cache-free mode** |
| `scripts/inductor/gemma_multi_size_compile_debug_sweep.py` | Sweep across model sizes (270m/1b/4b/12b/27b) |
| `scripts/inductor/gemma_batch_sweep_S4096.sh` | Batch-size sweep helper |

---

## 18. Commit history (this branch)

| Commit | What |
|---|---|
| `f24e4eefa6` | `numa_mlp_tp`: match Gemma GeGLU MLP for NUMA tensor-parallel sharding |
| `52436df852` | `numa_gemma_attention_tp`: add Gemma-specific attention NUMA-TP pass |
| `8811393f0b` | `numa_gemma_attention_tp`: support QK-norm and num_kv_heads>1 |
| `4091510935` | `numa_gemma_attention_tp`: bail on S-collision before any graph mutation |
| `6df500e7e9` | `numa_gemma_attention_tp`: positionally rewrite bmm-input 3D reshapes for B>1 |
| `4b8f0c6fa0` | gemma test scripts |
| `60ccfcb28f` | gemma batch sweep |
| `fbd39bc3d5` | `numa_gemma_attention_tp`: match Gemma attn-logit softcapping (9B/27B) |
| `5d674787e2` | `numa_gemma_attention_tp`: Phase 1 — in-graph K/V cache slicing (gated) |
| *(uncommitted)* | Cache-free model mode + per-node shard size (3-tuple) + slice_kv enabled |

---

## 19. Test commands

```bash
conda activate triton_cpu_integrated
cd /home/kumara/pytorch-private

# Unit tests
python test/inductor/test_numa_mlp_tp_pass.py
python test/inductor/test_numa_attention_tp_pass.py

# Single-layer 270M microbenchmark
python scripts/inductor/simple_gemma_decoder_block_speedup_cli.py \
    --batch-size 1 --seq-length 4096 --num-layers 1 --warmup 10 --iters 50

# 4B microbenchmark (K/V replicated, default)
python scripts/inductor/simple_gemma_decoder_block_speedup_cli.py \
    --batch-size 1 --seq-length 4096 --num-layers 1 \
    --h-in 2560 --num-heads 8 --num-kv-heads 4 --head-dim 256 \
    --mlp-hidden 10240 --warmup 10 --iters 50

# 4B microbenchmark with K/V SHARDED (requires cache-free, set by CLI default)
python -c "
import torch._inductor.config as ic
ic._numa_attention_tp_slice_kv = True
import runpy, sys
sys.argv = ['cli',
    '--batch-size', '1', '--seq-length', '4096', '--num-layers', '1',
    '--h-in', '2560', '--num-heads', '8', '--num-kv-heads', '4', '--head-dim', '256',
    '--mlp-hidden', '10240', '--warmup', '10', '--iters', '50',
]
runpy.run_path('scripts/inductor/simple_gemma_decoder_block_speedup_cli.py', run_name='__main__')
"

# Full Gemma-270M L=18 (verified 1.366x at S=4096)
python scripts/inductor/simple_gemma_decoder_block_speedup_cli.py \
    --batch-size 1 --seq-length 4096 --num-layers 18 --warmup 5 --iters 20

# Softcap path (Gemma 9B/27B style)
python scripts/inductor/simple_gemma_decoder_block_speedup_cli.py \
    --batch-size 1 --seq-length 4096 --num-layers 1 \
    --attn-logit-softcapping 50.0 --warmup 10 --iters 50

# Multi-model sweep
python scripts/inductor/gemma_multi_size_compile_debug_sweep.py \
    --result-dir ./gemma_sweep --models 270m 1b 4b \
    --batch-sizes 1 --seq-lens 2048 4096 \
    --warmup 5 --iters 20 --world-size 2

# Batch sweep at fixed S
bash scripts/inductor/gemma_batch_sweep_S4096.sh
```

---

## 20. Build

```bash
conda activate triton_cpu_integrated
cd /home/kumara/pytorch-private

export USE_CUDA=0 USE_ROCM=0 USE_XPU=0 USE_MKL=0 USE_MKLDNN=0
export BLAS=OpenBLAS USE_NUMA=1 MAX_JOBS=16

python -m pip install -e . -v --no-build-isolation \
  --config-settings=cmake.define.USE_MKL=OFF \
  --config-settings=cmake.define.USE_MKLDNN=OFF \
  --config-settings=cmake.define.BLAS=OpenBLAS \
  --config-settings=cmake.define.USE_NUMA=ON
```

Verify after build:
```bash
cd /tmp && python -c "
import torch
cpu = torch._C._cpu
print('numa_enabled:', cpu._is_numa_enabled())
print('numa_nodes:', cpu._num_numa_nodes())
"
```

---

## 21. Known limitations

1. **S-collision** at `S ∈ {q_size, shard_q, kv_size, shard_kv_size}` — pass
   bails to standard path. 270M is affected at S ∈ {256, 512, 1024}. Real
   prefill (S ≥ 2048) is unaffected.
2. **K/V sharding + KV cache combo is broken** — Inductor fused index_copy
   stride bug. Workaround: use `use_kv_cache=False` for slice_kv runs. Real
   fix requires Inductor-side patch (force `IndexPutFallback` over fused
   index_copy when destination is a sliced placeholder).
3. **Fused qkv_proj not matched** — model must use split `Wq`/`Wk`/`Wv`.
   Demo splits Gemma's reference fused QKV automatically.
4. **Serial reassembly** — sum-reduce after row-parallel mms is a single
   thread group, capping per-block speedup around 1.5–1.6×.
5. **K/V sharding gain is modest at 4B** — 1.380× (replicated) → 1.312×
   (sharded) at L=1 S=4096; the small K/V mm doesn't dominate. Larger
   models (27B kv_size=4096) should benefit more.

---

## 22. Quick reference: what gets sliced for each model

For N=2 sockets, `_numa_attention_tp_slice_kv` = False (default):

```
ALL MODELS:
  Wq            → COL-SPLIT (rows 0 to q_size/2 on n0, q_size/2 to q_size on n1)
  Wo            → ROW-SPLIT (cols 0 to q_size/2 on n0, q_size/2 to q_size on n1)
  MLP gate      → COL-SPLIT (rows 0 to intermediate/2 on n0, etc.)
  MLP up        → COL-SPLIT
  MLP down      → ROW-SPLIT
  Wk, Wv        → REPLICATED (full clone on each socket)
```

For N=2, `_numa_attention_tp_slice_kv = True` (4B+ with cache-free model):

```
ADDITIONALLY:
  Wk            → COL-SPLIT (rows 0 to kv_size/2 on n0, kv_size/2 to kv_size on n1)
  Wv            → COL-SPLIT (same)
```
