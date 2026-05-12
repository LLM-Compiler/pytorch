# Gemma NUMA Tensor-Parallel: End-to-End Implementation Guide

Architecture-first walkthrough of the CPU NUMA tensor-parallel feature as it
applies to the **Gemma** family (2 and 3). Each section starts from the
mathematical block, maps it to the eager PyTorch implementation, then traces
how the Inductor FX pass and wrapper codegen turn that block into per-socket
shards executed by the Triton CPU launcher.

> Companion document: `NUMA_TP_GEMMA_ARCHITECTURE.md` (this directory) — same
> feature, but indexed by *implementation file*. Use that one when you need
> to find a flag, a matcher predicate, or a commit. Use **this** doc when
> you want to understand how Gemma's algebra becomes shards.

---

## 1. Gemma decoder block at a glance

A Gemma decoder layer (Gemma-2 and Gemma-3 share this skeleton):

```
                  x_ℓ
                   │
            ┌──────┴──────┐
            │   RMSNorm   │   ← input_norm
            └──────┬──────┘
                   │
        ┌──────────┴───────────┐
        │   GQA Attention      │   §2
        │   (Q,K,V proj → SDPA │
        │    → O proj)         │
        └──────────┬───────────┘
                   │
            ┌──────┴──────┐
            │   RMSNorm   │   ← post_attention_norm   (Gemma-3 only)
            └──────┬──────┘
                   │
                   + ────────── residual x_ℓ
                   │
            ┌──────┴──────┐
            │   RMSNorm   │   ← pre_ffw_norm
            └──────┬──────┘
                   │
        ┌──────────┴───────────┐
        │   GeGLU MLP          │   §3
        │   (gate, up → GELU⊙  │
        │    → down)           │
        └──────────┬───────────┘
                   │
            ┌──────┴──────┐
            │   RMSNorm   │   ← post_ffw_norm           (Gemma-3 only)
            └──────┬──────┘
                   │
                   + ────────── residual (post-attention)
                   │
                  x_ℓ₊₁
```

Only the **GQA Attention** and **GeGLU MLP** sub-blocks are rewritten by the
FX passes. RMSNorms, residuals, and rotary embeddings are pointwise/cheap
and stay replicated on every socket.

### 1.1 Reference module (eager)

`scripts/inductor/simple_gemma_decoder_block_torch_compile_demo.py` defines
`SimpleGemmaStack` — a faithful mini-Gemma that the speedup CLI compiles.
It mirrors `gemma_pytorch/gemma/model.py`:

| Module | Where in graph | Sharded? |
|---|---|---|
| `RMSNorm`           | 4× per layer (Gemma-3) | replicated |
| `qkv_proj` (Linear) | attention input        | **column-parallel** (§2) |
| `apply_rotary_emb`  | post Q/K split         | replicated |
| `index_copy_` KV    | optional cache path    | n/a (cache-free preferred) |
| `q@kᵀ → softmax → @v` | attention core      | replicated activation (§2.3) |
| `o_proj` (Linear)   | attention output       | **row-parallel** (§2) |
| `gate_proj`, `up_proj` (Linear) | MLP input  | **column-parallel** (§3) |
| `gelu(gate)*up`     | MLP gate               | local pointwise |
| `down_proj` (Linear)| MLP output             | **row-parallel** (§3) |

### 1.2 Gemma-3 size classes

| Model | H | n_h | n_kv | g | d | H_f | L |
|---|---:|---:|---:|---:|---:|---:|---:|
| 270M | 1152 | 4  | 1  | 4 | 256 | 6912  | 18 |
| 1B   | 1152 | 4  | 1  | 4 | 256 | 6912  | 26 |
| 4B   | 2560 | 8  | 4  | 2 | 256 | 10240 | 34 |
| 27B  | 4608 | 32 | 16 | 2 | 128 | 36864 | 62 |

`g = n_h / n_kv` is the GQA group factor. 270M / 1B are MQA (`n_kv=1`),
4B / 27B are true GQA.

---

## 2. Attention block — Gemma to NUMA TP

### 2.1 Math

```
Q = x · W_Q     W_Q ∈ ℝ^{H × n_h·d}            # column-parallel
K = x · W_K     W_K ∈ ℝ^{H × n_kv·d}           # column-parallel (or replicated)
V = x · W_V     W_V ∈ ℝ^{H × n_kv·d}           # column-parallel (or replicated)
Q̃ = RoPE(Q),  K̃ = RoPE(K)
A  = softmax( (Q̃ · rep_g(K̃)ᵀ) / √d  ⊕  softcap?  ⊕  mask )
O  = A · rep_g(V)
y  = O · W_O    W_O ∈ ℝ^{n_h·d × H}            # row-parallel  → ∑ across sockets
```

`softcap?` is the Gemma-2 / Gemma-3 9B/27B logit clamp:
`logits ← c·tanh(logits/c)` with `c = 50`.

### 2.2 What gets sliced (P = num NUMA sockets)

```
Per socket p ∈ {0…P-1}:

  W_Q,p  = W_Q [:,  p·(n_h·d/P)  : (p+1)·(n_h·d/P) ]      # cols
  W_K,p  = W_K [:,  p·(n_kv·d/P) : (p+1)·(n_kv·d/P)]      # cols (if slice_kv)
  W_V,p  = W_V [:,  p·(n_kv·d/P) : (p+1)·(n_kv·d/P)]      # cols (if slice_kv)
  W_O,p  = W_O [p·(n_h·d/P)  : (p+1)·(n_h·d/P), :]        # rows

  y_p = (A_p · rep_g(V_p)) · W_O,p
  y   = Σ_p y_p     # cross-socket reduction
```

**MQA degeneracy (270M / 1B).** When `n_kv = 1`, the K/V column slice has
size `d/P` which is well below a useful tile. The pass detects this and
falls back to **K/V replication**: `W_K, W_V` are cloned on each socket
unchanged; only `W_Q` is column-sliced. Bandwidth saved on K/V is small
because `n_kv·d ≪ n_h·d` in MQA anyway.

**GQA (4B / 27B).** `n_kv ≥ P` so column-slicing K/V is well-defined and
keeps the `rep_g` step socket-local (the same `g` query heads still map to
the same KV head on the same socket).

### 2.3 Replicated activations, sharded weights

The activation `x` is replicated on both sockets (cheap; one tensor of
shape `[B,S,H]`). The expensive linears are sharded. The attention core
(`q·kᵀ`, softmax, `·v`) runs on the **column-sliced** Q/K/V so it
naturally narrows to `n_h/P` heads per socket. Only `W_O` requires a
cross-socket sum.

### 2.4 Code path — attention

1. **Pattern match.** `torch/_inductor/fx_passes/numa_attention_tp.py::_find_attention_blocks`
   walks the post-grad FX graph looking for the
   `mm(x, W_Q) / mm(x, W_K) / mm(x, W_V) → … → mm(O, W_O)` skeleton with
   tolerance for:
   - bias or no-bias (`addmm` vs `mm`) — Gemma is no-bias
   - SDPA-MATH decomposition: `bmm → softcap_chain? → softmax → bmm`
   - the `div→tanh→mul` softcap chain (Gemma 9B/27B)
   - reshape/view/expand collisions around `rep_g`
2. **Slice insertion.** For each matched block the pass inserts
   `aten.slice.Tensor` nodes immediately before the weight uses and rewires
   downstream view/reshape/expand chains. `numa_tp_common.py::
   _refresh_chain_vals_via_fake_tensor` re-runs the chain on FakeTensors so
   `meta['val']` stays consistent.
3. **Shard plan attached.** Each rewritten node carries
   `node.meta['numa_shard_plan']` (schema below), which the wrapper reads.
4. **Wrapper codegen.** `torch/_inductor/codegen/wrapper.py` reads each
   plan, emits per-socket `clone(.., numa_node=p)` for column/row-parallel
   weights, and builds `numa_args = [[ptrs_socket_0], [ptrs_socket_1]]`
   plus `numa_nodes = [0, 1, …]`.
5. **Launcher.** The Triton CPU driver
   (`triton-cpu/third_party/cpu/backend/driver.py`) sees `numa_args` on the
   kernel metadata, spawns one `std::thread` group per socket pinned via
   `pthread_setaffinity_np` to `/sys/devices/system/node/nodeN/cpulist`,
   and dispatches the same kernel with `per_node_ptrs[p]`.

### 2.5 Per-block role assignment (column vs row)

| Weight | Slice axis | Reason |
|---|---|---|
| `W_Q`, `W_K`, `W_V` | dim 0 of the kernel weight (= output / head axis) | **column-parallel**: outputs are independent per head |
| `W_O` | dim 0 of the kernel weight (= input axis of `O @ W_O`) | **row-parallel**: contracts over the partial dimension → needs sum |
| `b_O` (output bias) | – | **replicated and zeroed on sockets ≠ 0** to avoid double-counting after the sum |

> Wrapper-side slice dim conventions follow PyTorch linear weight layout:
> the weight matrix is `[out_features, in_features]`, so "column-parallel"
> on the math side is slice dim 0 in the wrapper. Don't confuse the two.

---

## 3. MLP block — Gemma GeGLU to NUMA TP

### 3.1 Math

```
g = x · W_gate                    # [B,S,H_f]   column-parallel
u = x · W_up                      # [B,S,H_f]   column-parallel
h = GELU(g, approximate='tanh') ⊙ u    # pointwise, local
y = h · W_down                    # [B,S,H]     row-parallel → ∑ across sockets
```

This is GeGLU. The same shape applies to SwiGLU (LLaMA family) — the matcher
does not hard-code GELU.

### 3.2 Sharding

```
Per socket p:
  W_gate,p = W_gate[:,  p·(H_f/P) : (p+1)·(H_f/P) ]   # cols
  W_up,p   = W_up  [:,  p·(H_f/P) : (p+1)·(H_f/P) ]   # cols
  W_down,p = W_down[ p·(H_f/P) : (p+1)·(H_f/P), : ]   # rows

  g_p = x · W_gate,p
  u_p = x · W_up,p
  h_p = GELU(g_p) ⊙ u_p                         # local
  y_p = h_p · W_down,p
  y   = Σ_p y_p
```

All ops up to `W_down` are **communication-free**. Only the final
row-parallel matmul produces a partial sum.

### 3.3 Pattern matching

`numa_mlp_tp.py::_find_geglu_quintuples` matches the 5-tuple
`(mm_gate, mm_up, act, mul, mm_down)` with `act ∈ {gelu (approx tanh), silu, sigmoid·x}`.
The classic two-linear-with-GELU MLP (BERT/GPT-2 style) is matched by the
older pattern in the same file; both produce the same shard-plan shape.

---

## 4. Shard plan schema (the contract between pass and wrapper)

```python
node.meta["numa_shard_plan"] = {
    "kind": "attention" | "mlp",
    "plan_id": int,
    "world_size": P,
    "shard_d": int,            # dominant chunk size (n_h·d/P or H_f/P)
    "compute_shard": 0,
    "x_node": str,             # name of replicated activation input

    # (arg_name, slice_dim) or (arg_name, slice_dim, shard_size)
    "column_parallel_nodes": [...],
    "row_parallel_nodes":    [...],

    "replicated_nodes":      [...],   # K/V when slice_kv=False, etc.
    "zero_on_nonzero_nodes": [...],   # row-parallel biases
}
```

The 3-tuple form (commit `1237df10e3`) lets a single plan mix shards of
different sizes — needed for GQA where `W_Q` shards at `n_h·d/P` but `W_K`,
`W_V` shard at `n_kv·d/P`.

---

## 5. End-to-end compile path

```
SimpleGemmaStack (eager nn.Module)
        │
        │  torch.compile(model)
        ▼
Dynamo trace ──▶ AOT Autograd FX graph (one flat graph for all L layers)
        │
        │  inductor post-grad passes
        ▼
┌───────────────────────────────────────┐
│  numa_attention_tp_pass               │   §2.4
│    • _find_attention_blocks           │
│    • _refresh_chain_vals_via_fake_tensor
│    • inserts aten.slice.Tensor nodes  │
│  numa_mlp_tp_pass                     │   §3.3
│    • _find_geglu_quintuples           │
│    • same slice insertion             │
└───────────────────┬───────────────────┘
                    │
                    ▼
Inductor lowering + scheduling
                    │
                    ▼
PythonWrapperCodegen
   • _emit_slice_clone(plan)            ← reads numa_shard_plan
   • per-socket clone(numa_node=p)
   • builds numa_args / numa_nodes
                    │
                    ▼
Triton CPU launcher (driver.py)
   • spawns std::thread groups per socket
   • bind_current_thread_to_cpu()
   • each thread runs kernel with per_node_ptrs[p]
                    │
                    ▼
            sum reduction → y_ℓ
```

---

## 6. Knobs that matter for Gemma

| Flag | Default | Use |
|---|---|---|
| `config._numa_attention_tp` | `False` | enable attention pass |
| `config._numa_mlp_tp`       | `False` | enable MLP pass |
| `config._numa_attention_tp_world_size` | NUMA nodes | force P (sweep, single-socket dev) |
| `config._numa_mlp_tp_world_size`       | NUMA nodes | same |
| `config._numa_attention_tp_slice_kv`   | `False` | column-slice `W_K`,`W_V` (GQA only; demands cache-free) |
| `config._numa_attention_tp_compute_shard` | `0` | which shard owns the SDPA compute path |
| `config._numa_*_debug`      | `False` | matcher + rewrite traces |
| `torch._C._cpu._set_numa_enabled(True)` | – | runtime gate read by `aten::clone(numa_node=…)` |

### 6.1 Cache-free vs cached prefill

The default Gemma reference writes K/V into preallocated caches via
`index_copy_`. Inductor currently fuses these writes in a way that's
incompatible with `slice_kv=True` (segfault on stride mismatch). The
`SimpleGemmaStack(use_kv_cache=False)` path skips the in-graph
`index_copy_` and uses `xk, xv` directly; this is the only path on which
`slice_kv=True` is safe today. See §9 of
`NUMA_TP_GEMMA_ARCHITECTURE.md` for the gory details.

---

## 7. Concrete walk-through: Gemma-3 4B, P=2, slice_kv=False

**Per layer, the pass inserts the following slices** (B=1, S=4096):

```
W_Q  : [2560, 2048]            slice dim 0 → socket 0: [0:1024,:]  socket 1: [1024:2048,:]
W_K  : [2560, 1024]            replicated (slice_kv=False)
W_V  : [2560, 1024]            replicated
W_O  : [2048, 2560]            slice dim 0 → socket 0: [0:1024,:]  socket 1: [1024:2048,:]
                                                          ↑ row-parallel → ∑

W_gate : [2560, 10240]         slice dim 0 → 5120 per socket
W_up   : [2560, 10240]         slice dim 0 → 5120 per socket
W_down : [10240, 2560]         slice dim 0 → 5120 per socket  → ∑
```

**Per layer, the runtime emits**:

- 2 socket-local clones for `W_Q`, `W_O`, `W_gate`, `W_up`, `W_down`
  (5 weights × 2 sockets = 10 clones per layer, first run only — cached
  thereafter in `wrapper._numa_tensor_cache`).
- 2 thread groups, each pinned to one socket's cpulist, executing the
  shard's kernels with `per_node_ptrs[p]`.
- 1 sum across the two `y_p` partial outputs of attention and MLP.

**Repeated × 34 layers** for the full 4B base model. The attention pass
fires per-layer (each layer's attention is its own subgraph after AOT
flattening); the MLP pass likewise.

---

## 8. Reproducing the end-to-end numbers

```bash
conda activate triton_cpu_integrated

# Single config, full base model (34 layers)
python scripts/inductor/simple_gemma_decoder_block_speedup_cli.py \
    --batch-size 1 --seq-length 4096 --num-layers 34 \
    --h-in 2560 --num-heads 8 --num-kv-heads 4 --head-dim 256 \
    --mlp-hidden 10240 --warmup 5 --iters 20

# 4-config sweep (1,4096) (2,2048) (4,512) (8,256)
bash scripts/inductor/gemma4b_bs_seq_sweep.sh
```

The CLI runs **standard then NUMA-TP** in one process and prints
`std_ms / numa_ms`. Numbers >1 mean NUMA-TP is faster. The 4B numbers in
the table in §1 of `NUMA_TP_GEMMA_ARCHITECTURE.md` are taken from this
exact code path with L=1; full-L numbers are layer-count-linear in
practice because every layer reuses the cached socket clones.

---

## 9. Gemma-specific quirks the implementation handles

| Quirk | Models affected | Handling |
|---|---|---|
| `attn_logit_softcapping` (`logits ← c·tanh(logits/c)`) | 2-9B, 2-27B, 3-9B-style | matcher allows `div→tanh→mul` between QK and softmax; downstream `meta['val']` refreshed via FakeTensor re-run |
| QK-norm (per-head RMSNorm on Q,K) | Gemma-3 | RMSNorm is pointwise — left replicated; the pass walks through it to find the SDPA core |
| Sliding-window vs global attention | Gemma-3 | mask-agnostic; pass shards both identically |
| MQA `n_kv=1` | 270M, 1B | `slice_kv` would zero-size socket 1; pass falls back to K/V replication automatically |
| Fused `qkv_proj` (single linear) | Some Gemma variants | the matcher recognises both fused and split forms; fused is sliced into three contiguous chunks per socket |
| `index_copy_` K/V write | cached-prefill path | incompatible with `slice_kv=True`; use `use_kv_cache=False` for the column-sharded K/V demo |
| Post-attention / post-FFW RMSNorm | Gemma-3 | sits between the row-parallel output and the residual; treated as pointwise after the cross-socket sum |

---

## 10. Where to look next

| You want to… | Read |
|---|---|
| Find a flag or config knob | `NUMA_TP_GEMMA_ARCHITECTURE.md` §15 |
| See exact pattern-matcher predicates | `numa_attention_tp.py::_find_attention_blocks`, `numa_mlp_tp.py::_find_geglu_quintuples` |
| Understand the wrapper codegen contract | `torch/_inductor/codegen/wrapper.py` (search `numa_shard_plan`) |
| Understand the launcher thread-pinning | `/home/kumara/triton-cpu-private/third_party/cpu/backend/driver.py::make_launcher` |
| See historical speedups & commits | `NUMA_TP_GEMMA_ARCHITECTURE.md` §1 and §18 |
| Run unit tests | `test/inductor/test_numa_attention_tp_pass.py`, `test/inductor/test_numa_mlp_tp_pass.py` |
