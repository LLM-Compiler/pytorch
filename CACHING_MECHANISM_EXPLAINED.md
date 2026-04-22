# NUMA Tensor Caching Mechanism — Deep Dive

## Overview

The caching mechanism in `wrapper.py` (`_numa_tensor_cache`) is a **deduplication system** that prevents cloning the same tensor multiple times across sequential NUMA kernels. This is critical for multi-stage operations like softmax decomposition where intermediate tensors are produced by one kernel and consumed by the next.

---

## The Cache Data Structure

### Definition (wrapper.py:535)
```python
self._numa_tensor_cache: Dict[str, List[str]] = {}
```

**Structure:**
```
{
  "buf1": ["_numa_p0_buf1_n0", "_numa_p0_buf1_n1"],  # per-NUMA-node cloned var names
  "buf2": ["_numa_p1_buf2_n0", "_numa_p1_buf2_n1"],
  ...
}
```

**Key**: Logical buffer name (e.g., `"buf1"`, `"arg1_1"`)  
**Value**: List of per-NUMA-node tensor variable names produced for that buffer

---

## Problem It Solves: Softmax Decomposition Example

### Scenario: Softmax in Attention Block

```
Kernel 1 (bmm_sum_kernel):      input: query  → output: qk_sum  (all threads write)
                                 ↓
Kernel 2 (softmax_kernel):      input: qk_sum → output: softmax_out  (read existing, write)
                                                                ↑
                                 This is an "in_out_ptr" (both read AND write)
```

Under NUMA TP:
- **Kernel 1**: Produces per-node copies of `qk_sum`
  - `_numa_p0_qk_sum_n0`, `_numa_p0_qk_sum_n1` (one per socket)
- **Kernel 2**: Needs to read the SAME per-node copies from Kernel 1
  - Must reuse `_numa_p0_qk_sum_n0`, `_numa_p0_qk_sum_n1`
  - **NOT** clone the shared buffer again!

### Without Caching (Incorrect Behavior)
```python
# Kernel 1 output
_numa_p0_qk_sum_n0 = empty_strided(...)    # allocate on socket 0
_numa_p0_qk_sum_n1 = empty_strided(...)    # allocate on socket 1
kernel1.run(..., _numa_p0_qk_sum_n0, _numa_p0_qk_sum_n1, ...)

# Kernel 2 input — WITHOUT CACHE
_numa_p1_qk_sum_n0 = clone(qk_sum, numa_node=0)  # WRONG! Clones shared buffer again
_numa_p1_qk_sum_n1 = clone(qk_sum, numa_node=1)  # Wastes memory, uses stale data
kernel2.run(..., _numa_p1_qk_sum_n0, _numa_p1_qk_sum_n1, ...)

# Result: kernel2 reads from wrong memory (shared buffer), not per-node updates from kernel1
```

### With Caching (Correct Behavior)
```python
# Kernel 1 output
_numa_p0_qk_sum_n0 = empty_strided(...)
_numa_p0_qk_sum_n1 = empty_strided(...)
kernel1.run(..., _numa_p0_qk_sum_n0, _numa_p0_qk_sum_n1, ...)

# Cache stores per-node var names
_numa_tensor_cache["qk_sum"] = ["_numa_p0_qk_sum_n0", "_numa_p0_qk_sum_n1"]

# Kernel 2 input — WITH CACHE
# Check: is "qk_sum" in cache?
if "qk_sum" in _numa_tensor_cache:
    # YES! Reuse the per-node names
    cached_n0 = _numa_tensor_cache["qk_sum"][0]  # "_numa_p0_qk_sum_n0"
    cached_n1 = _numa_tensor_cache["qk_sum"][1]  # "_numa_p0_qk_sum_n1"
    kernel2.run(..., cached_n0, cached_n1, ...)  # Read from correct per-node buffers

# Result: kernel2 reads from socket-local copies updated by kernel1
```

---

## Caching Workflow: Step-by-Step

### Context: Multi-Kernel Execution

```
Compiled Function (single wrapper.py file)
  ├─ def forward(arg_0_1, arg_1_1, ...):  ← shared Python buffers
  │    self._numa_tensor_cache = {}  ← cleared on entry
  │
  ├─ Kernel 1 (Q projection)
  │   ├─ Check: is input "X" in cache? NO
  │   ├─ Clone: _numa_p0_X_n0, _numa_p0_X_n1
  │   ├─ Run kernel with per-node args
  │   └─ Cache output: _numa_tensor_cache["qk_out"] = [_numa_p0_qk_out_n0, ...]
  │
  ├─ Kernel 2 (softmax)
  │   ├─ Check: is input "qk_out" in cache? YES ✓
  │   ├─ Reuse: _numa_p0_qk_out_n0, _numa_p0_qk_out_n1 (from kernel 1)
  │   ├─ Run kernel with same per-node args
  │   └─ Cache output: _numa_tensor_cache["softmax_out"] = [...]
  │
  └─ Kernel 3 (V projection × output)
      ├─ Check: is input "softmax_out" in cache? YES ✓
      ├─ Reuse: [per-node names from kernel 2]
      ├─ Run kernel
      └─ Cache output: _numa_tensor_cache["attn_out"] = [...]
```

---

## Cache Population Logic (wrapper.py:2232–2255)

### Step 1: Cache Outputs (wrapper.py:2242–2246)
```python
# After a kernel runs, cache its output pointers for reuse by next kernel
if (sig_name.startswith("out_ptr") or sig_name.startswith("in_out_ptr")):
    if arg_name in generated_by_arg:
        # Store per-node variable names
        self._numa_tensor_cache[arg_name] = generated_by_arg[arg_name]
```

**Effect**: Every kernel output (or in_out buffer) is stored with its per-NUMA-node variable names.

### Step 2: Cache Inputs (wrapper.py:2248–2255)
```python
# Also cache input args (replicated/sliced) for potential buffer-reuse scenarios
for arg_name, per_node in generated_by_arg.items():
    if arg_name not in self._numa_tensor_cache and all(per_node):
        self._numa_tensor_cache[arg_name] = per_node
```

**Effect**: Replicated or sliced inputs are cached in case they're outputs from a *previous* kernel and get reinterpreted (buffer-reuse optimization).

---

## Cache Lookup Logic (wrapper.py:2164–2177)

### When Processing Next Kernel's Args
```python
for arg_name, arg_type in zip(call_args, arg_types):
    if not isinstance(arg_name, str) or not isinstance(arg_type, torch_dtype):
        continue
    
    for sid in range(world_size):
        # **CACHE HIT**: Check if this arg was produced by a previous kernel
        if arg_name in self._numa_tensor_cache:
            cached_name = self._numa_tensor_cache[arg_name][sid]
            per_node_arg_names[sid].append(cached_name)
            self._numa_reclone_consumed.add(cached_name)
            continue  # ← SKIP cloning; reuse the cached per-node variable
        
        # CACHE MISS: Clone the tensor (default behavior)
        # ... clone logic (emit_replicate, emit_slice_clone, etc.)
```

**Key behavior:**
- **Cache hit** → Reuse per-node variable names (no clone)
- **Cache miss** → Generate fresh clones

---

## Concrete Example: Attention with QKV + SDPA + Out

### Generated Python Code

```python
def forward(arg_0_1, arg_1_1, arg_2_1, arg_3_1):
    # arg_0_1 = input X
    # arg_1_1 = Wq
    # arg_2_1 = intermediate (QK matmul output)
    # arg_3_1 = attention output
    
    # ============================================================
    # Kernel 1: Q projection (addmm)
    # ============================================================
    def run_q_proj(...):
        # Input X is replicated → clone both sockets
        _numa_p0_arg0_n0 = torch.ops.aten.clone.default(arg_0_1, numa_node=0)
        _numa_p0_arg0_n1 = torch.ops.aten.clone.default(arg_0_1, numa_node=1)
        
        # Wq is column-parallel → clone-then-slice
        _numa_p0_arg1_full_n0 = torch.ops.aten.clone.default(arg_1_1, numa_node=0)
        _numa_p0_arg1_n0 = torch.ops.aten.slice.Tensor(_numa_p0_arg1_full_n0, 1, 0, 512, 1)
        
        _numa_p0_arg1_full_n1 = torch.ops.aten.clone.default(arg_1_1, numa_node=1)
        _numa_p0_arg1_n1 = torch.ops.aten.slice.Tensor(_numa_p0_arg1_full_n1, 1, 512, 1024, 1)
        
        # Output Q is in_out (will be read by softmax)
        _numa_p0_q_out_n0 = empty_strided_cpu_numa(...)  # allocate on socket 0
        _numa_p0_q_out_n1 = empty_strided_cpu_numa(...)  # allocate on socket 1
        
        _numa_p0_numa_args = [
            (_numa_p0_arg0_n0, _numa_p0_arg1_n0, _numa_p0_q_out_n0),
            (_numa_p0_arg0_n1, _numa_p0_arg1_n1, _numa_p0_q_out_n1),
        ]
        q_proj_kernel.run(..., numa_args=_numa_p0_numa_args, ...)
        
        # CACHE OUTPUT Q
        _numa_tensor_cache["q_proj_out"] = ["_numa_p0_q_out_n0", "_numa_p0_q_out_n1"]
    
    run_q_proj()
    
    # ============================================================
    # Kernel 2: Softmax (takes Q as input, produces softmax_out)
    # ============================================================
    def run_softmax(...):
        # Input Q: CHECK CACHE
        if "q_proj_out" in _numa_tensor_cache:
            # CACHE HIT! Reuse per-node Q from kernel 1
            q_input_n0 = _numa_tensor_cache["q_proj_out"][0]  # "_numa_p0_q_out_n0"
            q_input_n1 = _numa_tensor_cache["q_proj_out"][1]  # "_numa_p0_q_out_n1"
        else:
            # CACHE MISS (shouldn't happen)
            q_input_n0 = torch.ops.aten.clone.default(q_proj_out, numa_node=0)
            q_input_n1 = torch.ops.aten.clone.default(q_proj_out, numa_node=1)
        
        # Output: softmax result (new buffer)
        _numa_p1_softmax_out_n0 = empty_strided_cpu_numa(...)
        _numa_p1_softmax_out_n1 = empty_strided_cpu_numa(...)
        
        _numa_p1_numa_args = [
            (q_input_n0, _numa_p1_softmax_out_n0),
            (q_input_n1, _numa_p1_softmax_out_n1),
        ]
        softmax_kernel.run(..., numa_args=_numa_p1_numa_args, ...)
        
        # CACHE OUTPUT softmax
        _numa_tensor_cache["softmax_out"] = ["_numa_p1_softmax_out_n0", "_numa_p1_softmax_out_n1"]
    
    run_softmax()
    
    # ============================================================
    # Kernel 3: Output linear (row-parallel)
    # ============================================================
    def run_out_linear(...):
        # Input softmax: CHECK CACHE
        if "softmax_out" in _numa_tensor_cache:
            # CACHE HIT! Reuse per-node softmax from kernel 2
            softmax_input_n0 = _numa_tensor_cache["softmax_out"][0]
            softmax_input_n1 = _numa_tensor_cache["softmax_out"][1]
        else:
            softmax_input_n0 = torch.ops.aten.clone.default(softmax_out, numa_node=0)
            softmax_input_n1 = torch.ops.aten.clone.default(softmax_out, numa_node=1)
        
        # Wo is row-parallel (reduce dimension)
        _numa_p2_wo_n0 = torch.ops.aten.slice.Tensor(Wo, 0, 0, d_model//2, 1)
        _numa_p2_wo_n0_clone = torch.ops.aten.clone.default(_numa_p2_wo_n0, numa_node=0)
        
        _numa_p2_wo_n1 = torch.ops.aten.slice.Tensor(Wo, 0, d_model//2, d_model, 1)
        _numa_p2_wo_n1_clone = torch.ops.aten.clone.default(_numa_p2_wo_n1, numa_node=1)
        
        _numa_p2_out_n0 = empty_strided_cpu_numa(...)
        _numa_p2_out_n1 = empty_strided_cpu_numa(...)
        
        _numa_p2_numa_args = [
            (softmax_input_n0, _numa_p2_wo_n0_clone, _numa_p2_out_n0),
            (softmax_input_n1, _numa_p2_wo_n1_clone, _numa_p2_out_n1),
        ]
        out_linear_kernel.run(..., numa_args=_numa_p2_numa_args, ...)
        
        # Final output (may be all-reduced and re-cloned for non-TP consumption)
        attn_out_n0 = all_reduce(_numa_p2_out_n0)
        attn_out_n1 = all_reduce(_numa_p2_out_n1)
        
        # CACHE OUTPUT
        _numa_tensor_cache["attn_final_out"] = [attn_out_n0, attn_out_n1]
```

---

## Memory Impact Analysis

### Without Cache (Redundant Cloning)
```
Kernel 1 output: clone Q twice (inputs + outputs for K, V, softmax, etc.)
Kernel 2 input:  clone Q AGAIN from shared buffer
Kernel 3 input:  clone softmax AGAIN from shared buffer

Total clones ∝ (num_kernels × num_buffers)
```

### With Cache (Deduplication)
```
Kernel 1 output: clone Q once → cache
Kernel 2 input:  REUSE Q from cache (no clone) ✓
Kernel 3 input:  REUSE softmax from cache (no clone) ✓

Total clones ∝ (num_unique_buffers)  // Much smaller!
```

### Example: Softmax Decomposition (4 kernels)
```
Kernel 1: bmm_sum         → produces qk_sum
Kernel 2: scale_softmax   → reads qk_sum (CACHE HIT), outputs softmax_out
Kernel 3: bmm_attn        → reads softmax_out (CACHE HIT)
Kernel 4: output_linear   → reads attn_out (CACHE HIT)

Without cache: clone 4 buffers × 4 kernels = 16 clone calls
With cache:    clone 4 buffers × 1 kernel  = 4 clone calls  ✓ 4× reduction
```

---

## Cache Invalidation & Scope

### Cache Lifetime: Single Forward Pass
```python
class PythonWrapperCodegen:
    def __init__(self):
        self._numa_tensor_cache = {}  # Created per wrapper instance
```

The cache is **local to a single compiled function's execution**, not global:
- Fresh cache per forward pass
- No cross-call pollution
- Handles different batch sizes automatically (activations re-cloned per batch)

### Cache Invalidation Scenarios
1. **Buffer reuse**: If Inductor reuses memory for a different tensor, the cache entry is cleared (see `_numa_propagate_buffer_alias`)
2. **Terminal outputs**: Final outputs are re-cloned for non-NUMA paths (via `_NumaReCloneLine` deferred logic)
3. **All-reduce**: After row-parallel reduction, outputs are re-cloned for next-layer consumption

---

## Related Mechanisms

### 1. `_numa_reclone_consumed` (wrapper.py:541)
```python
self._numa_reclone_consumed: Set[str] = set()
```
Tracks which post-all-reduce re-clones are actually used. Eliminates dead re-clone code at flush time.

### 2. `_numa_suppressed_allocs` (wrapper.py:546)
```python
self._numa_suppressed_allocs: Set[str] = set()
```
Suppresses `empty_strided` allocations that are superseded by buffer-reuse aliases.

### 3. `_NUMAFreshAllocLine` (wrapper.py:81–99)
A deferred line that only emits if the allocation isn't suppressed (buffer wasn't reused).

---

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Deduplicate NUMA cloning across sequential kernels |
| **Data structure** | `Dict[tensor_name] → List[per_node_var_names]` |
| **Scope** | Single forward pass (fresh per call) |
| **Lookup** | Before cloning, check cache for per-node tensor names |
| **Hit** | Reuse names (no clone); mark as consumed |
| **Miss** | Generate fresh clones |
| **Benefit** | 4–10× reduction in clone calls for softmax/multi-kernel ops |
| **Critical for** | Softmax decomposition, multi-stage kernels with in_out_ptr args |

---

## Why This Matters for Your LLM-Level Pass

In the **LLM-level pre-cloning approach**, you wouldn't need this caching because:
- All weights are cloned **once upfront**, not per-kernel
- Activations (batch-dependent) are still cloned per-kernel (where caching helps)

However, **activations** benefit hugely from caching:
- Q/K/V outputs from attention → softmax inputs (cache hit)
- Softmax outputs → attention weights (cache hit)
- MLP intermediate (gelu output) → linear2 input (cache hit)

So your **hybrid approach** would:
1. **Pre-clone weights** (no kernel-level cloning for weights)
2. **Keep activation caching** (per-kernel cloning for x, intermediate activations)
3. Reduce overall clone overhead by ~50–70%

