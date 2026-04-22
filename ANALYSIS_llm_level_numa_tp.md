# Analysis: Kernel-Level vs LLM-Level NUMA TP Architecture

## Current Implementation (Kernel-Level)

### Architecture Overview
The current approach operates **per-kernel** during the Inductor post-grad pass:

```
torch.compile(model)
  ↓
Dynamo trace → AOT Autograd → post-grad FX graph
  ↓
numa_mlp_tp_pass / numa_attention_tp_pass (kernel-level)
  • Pattern match: linear→gelu→linear (MLP) or Q/K/V+SDPA+out (attention)
  • Insert per-shard slice nodes (aten.slice.Tensor)
  • Annotate with numa_shard_plan metadata
  • Rewrite kernel operands to sliced versions
  ↓
Inductor lowering → scheduling → kernel emission
  ↓
PythonWrapperCodegen._emit_numa_kernel_call()
  • Read numa_shard_plan from kernel metadata
  • Emit per-NUMA clone calls: clone(tensor, numa_node=sid)
  • Build numa_args=[[n0_ptrs], [n1_ptrs], ...]
  ↓
aten.slice + clone execution happens **per kernel invocation**
  ↓
Triton CPU launcher reads numa_args → pins threads → runs kernel
```

### Key Characteristics
1. **Distributed cloning**: Each kernel independently clones its inputs/outputs
2. **Metadata-driven**: `numa_shard_plan` dict (generalized schema):
   - `column_parallel_nodes`: [(name, dim), ...] — slice along dim
   - `row_parallel_nodes`: [(name, dim), ...] — slice along dim
   - `replicated_nodes`: [name, ...] — full clone
   - `zero_on_nonzero_nodes`: [name, ...] — replicated but zeroed on non-zero sockets
3. **Graph-level rewriting**: Slices inserted in FX graph, but actual clones only in wrapper codegen
4. **Caching**: `_numa_tensor_cache` reuses per-node outputs across kernels

### Current Limitations
- **Redundant cloning**: Same weight tensor may be cloned multiple times if used by multiple kernels
- **SDPA decomposition dependency**: Attention pass assumes `SDPBackend.MATH` (primitive ops visible)
- **Stride complexity**: Column-parallel slices require full clone-then-slice for dim≠0 to preserve kernel's baked strides
- **No global weight categorization**: Pass doesn't know global importance or priority of weights

---

## Proposed: LLM-Level Unified Pass

### Architecture Overview
Operate **once at module entry** before any kernel dispatch:

```
torch.compile(llm_module)
  ↓
Dynamo trace → AOT Autograd → post-grad FX graph
  ↓
numa_llm_tp_pass (NEW: module-level, runs once)
  1. Traverse full graph; categorize ALL weights:
     • Fully replicated (x, attention biases, layer norms)
     • Column-parallel (Q/K/V/MLP.w_in matrices + their biases)
     • Row-parallel (Attention.Wo, MLP.w_out matrices)
  2. Extract weight tensor nodes + dtypes
  3. Build global clone plan:
     {
       "replicated": {
         "layer_norm.weight": {
           "full_size": (d_model,),
           "clone_vars": [cloned_var0, cloned_var1, ...]
         },
         ...
       },
       "column_parallel": {
         "attn.Wq": {
           "full_size": (d_qkv, d_model),
           "shard_d": d_qkv / world_size,
           "slice_dim": 1,
           "clones_per_node": [
             [(node0_clone0, slice(0:d/2, :)), ...],
             [(node1_clone0, slice(d/2:d, :)), ...],
           ]
         },
         ...
       },
       "row_parallel": { ... }
     }
  4. Emit pre-execution cloning code (Python, not in graph):
     ```python
     # Replicated
     _llm_ln_weight_n0 = torch.ops.aten.clone.default(ln.weight, numa_node=0)
     _llm_ln_weight_n1 = torch.ops.aten.clone.default(ln.weight, numa_node=1)
     
     # Column-parallel
     _llm_Wq_n0_shard = torch.ops.aten.slice.Tensor(Wq, 1, 0, 512, 1)
     _llm_Wq_n0 = torch.ops.aten.clone.default(_llm_Wq_n0_shard, numa_node=0)
     _llm_Wq_n1_shard = torch.ops.aten.slice.Tensor(Wq, 1, 512, 1024, 1)
     _llm_Wq_n1 = torch.ops.aten.clone.default(_llm_Wq_n1_shard, numa_node=1)
     ...
     ```
  5. Return cloned tensor dict + metadata
  ↓
Kernel-level passes become **metadata only** (register cloned vars + reference metadata)
  ↓
Kernels read pre-cloned tensors directly from module state
  • numa_shard_plan still guides which args need all-reduce, but no clone emission
  ↓
Execution: weights already on sockets, just run kernels
```

### Advantages
1. **Single cloning pass**: Each weight cloned exactly once, upfront
2. **Full-graph context**: Categorize all weights together (replicated vs col/row parallel)
3. **Memory efficiency**: Column-parallel can use true slice-then-clone (dim=0 strategy) because strides are NOT baked by this point
4. **Stride safety**: Cloning happens before kernel specialization, so the cloned tensor can be compact
5. **Module fusion**: Weights stay in the model class as buffers (optional), or return as context dict
6. **Cleaner kernel passes**: MLP/attention passes become thin wrappers that just register pre-cloned operands
7. **Debuggability**: All clones emitted in one place; easier to profile memory usage

### Design Questions
1. **Where does the pre-cloning code live?**
   - Option A: Wrap in a context manager / initializer called before `forward()`
   - Option B: Emit as setup function + update all kernel dispatch to read from shared dict
   - Option C: Register cloned buffers on the module itself (`self._numa_wq_n0`, etc.)
   
2. **How to handle dynamic batch sizes / sequence lengths?**
   - Activations (x) must still be cloned per-kernel (batch-dependent)
   - Only static weights pre-clone

3. **Attention pattern matching**: still need to detect Q/K/V, but just to annotate, not to rewrite?

4. **Integration with existing wrapper codegen?**
   - Fall back to current path if no LLM-level clones available
   - Or require both passes (llm-level mandatory, kernel-level optional metadata)

---

## Comparative Analysis

| Aspect | Current (Kernel-Level) | Proposed (LLM-Level) |
|--------|------------------------|----------------------|
| **Cloning overhead** | Per-kernel × num_kernels | Once per model |
| **Memory spike** | Local per kernel | Global upfront |
| **Stride handling** | Clone-full-then-slice for dim≠0 | Slice-then-clone (true per-node) |
| **Pattern matching** | MLP + attention (limited) | Any pattern (full graph context) |
| **Debuggability** | Clones spread across kernel codegen | Centralized pre-cloning pass |
| **Backward compat** | Works with any kernel | Requires new wrapper integration |
| **Complexity** | Medium | High (initial), simpler downstream |
| **Memory profile** | Unpredictable (per-kernel) | Predictable (one alloc phase) |

---

## Recommendation

### Best Path Forward: **Hybrid Approach**

1. **Phase 1**: Keep current kernel-level passes (MLP + attention).
   - Proven, works with existing codegen.
   - Reduced cloning via `_numa_tensor_cache`.

2. **Phase 2**: Add optional LLM-level pre-cloning pass.
   - Triggered by config flag: `config._numa_llm_tp_preclone = True`.
   - Emits weight cloning upfront (replicated + col/row parallel).
   - Kernel passes fall back to metadata-only (register cloned operands, no clone emission).
   - Activations still cloned per-kernel (batch-dependent).

3. **Phase 3**: Profile & measure.
   - Compare memory usage, latency, stride correctness.
   - Decide if LLM-level becomes default or stays optional.

### Why Hybrid?
- **Lower risk**: Current path still works; LLM-level is additive.
- **Incremental gains**: Reduce weight-cloning redundancy without full rewrite.
- **Experimental**: Can A/B test on real models (LLaMA, Mistral, etc.).
- **Cleaner backward compat**: Old passes can be deprecated gradually.

---

## Implementation Sketch for LLM-Level Pass

```python
# torch/_inductor/fx_passes/numa_llm_tp.py

def numa_llm_tp_preclone_pass(graph: torch.fx.Graph, module) -> Dict[str, Any]:
    """
    Single pass before any kernel dispatch. Returns a dict of pre-cloned tensors.
    
    Returns:
    {
        "replicated": {
            "layer_norm.weight": [clone_node0, clone_node1, ...],
            ...
        },
        "column_parallel": {
            "attn.Wq": [
                [shard_node0, shard_node1, ...],  # per node shards
                ...
            ],
            ...
        },
        "row_parallel": { ... },
        "metadata": {
            "world_size": 2,
            "shard_d_by_weight": { ... },
            ...
        }
    }
    """
    world_size = get_numa_world_size()
    if world_size < 2:
        return {}
    
    # Step 1: Traverse graph, categorize weights
    weight_catalog = {
        "replicated": {},
        "column_parallel": {},
        "row_parallel": {},
    }
    
    # Scan for patterns: layer norms (replicated), Q/K/V (col), Wo/fc2 (row)
    for node in graph.nodes:
        if is_layer_norm_weight(node):
            weight_catalog["replicated"][node.name] = node
        elif is_qkv_weight(node):
            weight_catalog["column_parallel"][node.name] = node
        elif is_output_weight(node):
            weight_catalog["row_parallel"][node.name] = node
    
    # Step 2: Build clone plan (replicated + col/row)
    # ... (detail elided)
    
    # Step 3: Emit cloning code
    clones_by_role = {}
    for role, weights in weight_catalog.items():
        clones_by_role[role] = _emit_clones_for_role(
            graph, weights, world_size, role
        )
    
    return clones_by_role
```

---

## Risk Assessment

### High Risk
- **Graph structure**: Inserting/removing clone nodes requires careful handling of FX node insertion points
- **Activation cloning**: Still per-kernel, so "pre-clone everything" claim is false for activations

### Medium Risk
- **Pattern detection**: Categorizing all weights correctly requires robust heuristics
- **Buffer tracking**: Ensuring no double-cloning of intermediate activations

### Low Risk
- **Backward compat**: Can coexist with current passes behind a config flag
- **Correctness**: Cloning is idempotent; wrong categorization is safer than current stride issues

---

## Conclusion

The **LLM-level unified pass is architecturally cleaner and more efficient**, but:

1. **Current kernel-level passes already work** with reduced overhead via caching.
2. **Hybrid approach is safest**: Add LLM-level as optional, measure impact.
3. **Key benefit**: Stride safety + single cloning phase (especially for weights).
4. **Key challenge**: Activations remain per-kernel (batch-dependent); can't truly eliminate this.

**Recommended next step**: Prototype LLM-level pass with a few weights (layer norms), measure cloning overhead reduction, then expand to attention/MLP weights.
