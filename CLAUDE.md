# CLAUDE.md — pytorch-private (NUMA TP / Triton CPU branch)

## Project goal

CPU-only NUMA tensor-parallel (TP) inference acceleration using Inductor FX passes +
Triton CPU codegen. **No GPU code.** Triton codegen is kept because the target is the
Triton CPU backend (not CUDA).

The feature inserts per-NUMA-socket weight shards into the Inductor post-grad FX graph
and tags kernels with `numa_args` metadata. The Triton CPU launcher
(`/home/kumara/triton-cpu-private/third_party/cpu/backend/driver.py`) reads those tags
and pins one `std::thread` group per socket with CPU affinity set from
`/sys/devices/system/node/nodeN/cpulist`.

---

## Active files (this branch)

### Inductor FX passes — core feature code

| File | Role |
|------|------|
| `torch/_inductor/fx_passes/numa_attention_tp.py` | Attention-block NUMA TP pass: matches Q/K/V + out-linear pattern, inserts per-shard `aten.slice` nodes, rewires operands, propagates `numa_shard_plan` metadata. |
| `torch/_inductor/fx_passes/numa_mlp_tp.py` | MLP NUMA TP pass: matches `linear→gelu→linear`, same slice/rewire pattern. |
| `torch/_inductor/fx_passes/numa_tp_common.py` | Shared helpers for both passes: `get_numa_world_size`, `set_numa_meta`, `resize_node_val_and_tensor_meta`, `ensure_generalized_lists`, generalized shard-plan schema. |
| `torch/_inductor/fx_passes/post_grad.py` | Registers both passes in the Inductor post-grad pass pipeline. |
| `torch/_inductor/codegen/wrapper.py` | Wrapper codegen: reads `numa_shard_plan` from kernel metadata and emits per-NUMA-node clone/empty-strided allocations + `numa_args` list passed to the launcher. |
| `torch/_inductor/config.py` | Config flags: `_numa_mlp_tp`, `_numa_attention_tp`, `_numa_*_world_size`, `_numa_*_compute_shard`, `_numa_*_debug`, `_numa_*_shard_plan`. |
| `torch/_inductor/utils.py` | NUMA utility helpers used by the passes. |
| `torch/csrc/cpu/Module.cpp` | Exposes `torch._C._cpu._is_numa_enabled()`, `_num_numa_nodes()`, `_set_numa_enabled()` to Python. |

### ATen / core extension

| File | Role |
|------|------|
| `aten/src/ATen/native/TensorFactories.cpp` | Dense `clone` extended with optional `numa_node` placement via `c10::NUMAMove`. |
| `aten/src/ATen/native/native_functions.yaml` | Schema: `clone(..., int? numa_node=None)`. |
| `aten/src/ATen/functorch/BatchRulesUnaryOps.cpp` | Forwards `numa_node` through functorch batch rules. |
| `torch/_refs/__init__.py` | Python-level `clone` forwards `numa_node`. |

### Tests

| File | Role |
|------|------|
| `test/inductor/test_numa_attention_tp_pass.py` | Unit tests for the attention FX pass (graph rewrite, metadata, shape correctness). |
| `test/inductor/test_numa_mlp_tp_pass.py` | Unit tests for the MLP FX pass. |
| `test/test_torch.py` | `clone_numa_node` API tests, NUMA page-placement checks. |

### Demo / diagnosis scripts

| Path | Purpose |
|------|---------|
| `scripts/inductor/attention_block_pytorch_demo.py` | Eager (non-compiled) attention reference. |
| `scripts/inductor/attention_block_torch_compile_demo.py` | End-to-end `torch.compile` attention with NUMA TP. |
| `scripts/inductor/check_numa_attention_output_correctness.py` | Numerics check: NUMA vs standard compiled attention. |
| `scripts/inductor/debug_output_code_intermediate_finite.py` | Diagnoses intermediate NaN/Inf in generated output code. |
| `scripts/inductor/output_codes/attention/…` | Captured Inductor output_code.py snapshots (numa vs standard). |
| `scripts/inductor/output_codes/mlp/…` | Captured output_code.py for MLP variants. |

---

## Shard plan schema (`numa_shard_plan` dict)

All FX pass nodes that are sharded carry `node.meta["numa_shard_plan"]` with:

```python
{
  "kind": "mlp" | "attention",
  "plan_id": int,
  "world_size": int,
  "shard_d": int,              # hidden-dim chunk per socket
  "compute_shard": int,        # which socket runs the compute path
  "x_node": str,               # replicated activation input
  "column_parallel_nodes": List[Tuple[str, int]],   # (arg_name, slice_dim)
  "row_parallel_nodes":    List[Tuple[str, int]],
  "replicated_nodes":      List[str],
  "zero_on_nonzero_nodes": List[str],   # row-parallel bias: zero on non-zero sockets
}
```

`ensure_generalized_lists()` in `numa_tp_common.py` converts legacy MLP plans
(with `a_node`/`b_node` keys) to this schema.

---

## Triton CPU launcher (external repo)

**File:** `/home/kumara/triton-cpu-private/third_party/cpu/backend/driver.py`

The launcher's `make_launcher()` generates a C extension (`__triton_cpu_launcher`) that
is the bridge from Python kernel dispatch to C++ thread execution.

### NUMA path in the launcher

When `kernel_metadata.numa_args` is present (a list-of-lists of tensor pointers, one
per socket), the launcher:

1. Reads `kernel_metadata.numa_list` / `numa_nodes` / `numa_node_list` to get NUMA
   node IDs per shard.
2. Reads `/sys/devices/system/node/nodeN/cpulist` to get CPU IDs for each node.
3. Spawns `total_threads` `std::thread`s split across sockets; each thread calls
   `bind_current_thread_to_cpu()` via `pthread_setaffinity_np`.
4. Each thread group iterates its grid slice with `per_node_ptrs[node_i]` (the
   socket-local weight copies).

If `numa_args` is absent or invalid, it falls back to `run_omp_kernels()` (standard
OpenMP parallel for).

### Key metadata attributes read by the launcher

| Attribute | Source | Meaning |
|-----------|--------|---------|
| `kernel_metadata.numa_args` | `wrapper.py` codegen | `[[ptr0_n0, ptr1_n0, …], [ptr0_n1, …], …]` |
| `kernel_metadata.numa_list` / `numa_nodes` | `wrapper.py` codegen | NUMA node IDs per shard |
| `kernel_metadata.num_threads` | Triton compiler | Total thread count hint |

---

## Codegen flow (end-to-end)

```
torch.compile(model)
  → Dynamo trace → AOT Autograd → post-grad FX graph
  → numa_mlp_tp_pass / numa_attention_tp_pass      ← this branch
      inserts aten.slice nodes, sets numa_shard_plan metadata
  → Inductor lowering → scheduling
  → PythonWrapperCodegen._emit_numa_kernel_call()  ← wrapper.py
      reads numa_shard_plan, emits per-socket clone() calls
      builds numa_args=[[n0_ptrs], [n1_ptrs]], numa_nodes=[0,1,…]
  → kernel metadata object attached to triton kernel call
  → CPULauncher.__call__() in driver.py
      detects numa_args → spawn threads → bind CPUs → run kernel
```

---

## Build (CPU-only, NUMA-enabled)

```bash
conda activate triton_cpu_integrated
cd /home/kumara/pytorch-private

export CMAKE_PREFIX_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages;${CONDA_PREFIX}"
export CMAKE_INCLUDE_PATH="${CONDA_PREFIX}/include"
export LIBRARY_PATH="${CONDA_PREFIX}/lib"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export USE_CUDA=0
export USE_ROCM=0
export USE_XPU=0
export USE_MKL=0
export USE_MKLDNN=0
export BLAS=OpenBLAS
export USE_NUMA=1
export MAX_JOBS=16

python -m pip install -e . -v --no-build-isolation \
  --config-settings=cmake.define.USE_MKL=OFF \
  --config-settings=cmake.define.USE_MKLDNN=OFF \
  --config-settings=cmake.define.BLAS=OpenBLAS \
  --config-settings=cmake.define.USE_NUMA=ON
```

Verify NUMA hooks after rebuild:
```bash
cd /tmp
python -c "import torch; cpu=torch._C._cpu; print(cpu._is_numa_enabled(), cpu._num_numa_nodes())"
```

---

## Run tests

For each run/compile always use conda activate triton_cpu_integrated

When it's needed to run the generated output code with realted to NUMA. Use this. Otherwise normal python run is sufficeint. 

python - <<'PY'
import torch, runpy
torch._C._cpu._set_numa_enabled(True)
print("numa enabled:", torch._C._cpu._is_numa_enabled(), "nodes:", torch._C._cpu._num_numa_nodes())
runpy.run_path("/path/to/outputcode.py", run_name="__main__")
PY


```bash
# Attention pass unit tests
python test/inductor/test_numa_attention_tp_pass.py

# MLP pass unit tests
python test/inductor/test_numa_mlp_tp_pass.py

# End-to-end correctness (attention)
python scripts/inductor/check_numa_attention_output_correctness.py

# End-to-end correctness (MLP)
python scripts/inductor/check_numa_output_correctness.py
```

---

## Config flags (quick reference)

```python
import torch._inductor.config as config

# Enable passes
config._numa_mlp_tp = True
config._numa_attention_tp = True

# Override NUMA world size (sockets) for testing on single-socket machines
config._numa_mlp_tp_world_size = 2
config._numa_attention_tp_world_size = 2

# Which shard runs the compute path (0-indexed)
config._numa_mlp_tp_compute_shard = 0
config._numa_attention_tp_compute_shard = 0

# Debug prints
config._numa_mlp_tp_debug = True
config._numa_attention_tp_debug = True
```

---

## Key invariants / gotchas

- **No clones in FX graph.** Passes insert only `aten.slice.Tensor` nodes. Clone calls
  happen entirely in wrapper.py codegen (Python string emission), not in the FX graph.
- **`in_out_ptr` buffers** (softmax intermediate, etc.) must be treated the same as
  `out_ptr` on first allocation — pre-existing cached per-node copies are reused on
  subsequent kernels via `self._numa_tensor_cache`.
- **Row-parallel bias zeroing.** `bo` (output linear bias) is replicated but zeroed on
  sockets ≠ 0 to avoid double-counting after the post-kernel all-reduce.
- **Shape refresh after rewiring.** After slicing Q/K/V weight operands, the chain of
  view/reshape/expand/softmax nodes has stale `meta['val']` shapes. The pass re-runs
  each op on FakeTensors via `_refresh_chain_vals_via_fake_tensor()`.
- **SDPA decomposition.** The attention pass assumes `SDPBackend.MATH` decomposition so
  all heavy ops (bmm + softmax + bmm) appear as primitive aten ops in the graph.
