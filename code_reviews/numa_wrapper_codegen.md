# NUMA Wrapper Codegen Overview

The NUMA wrapper codegen process in PyTorch Inductor bridges the high-level Graph IR representation with lower-level physical memory allocations and kernel metadata. This process ensures tensors are mapped to the correct Non-Uniform Memory Access (NUMA) sockets physically, eliminating expensive cross-memory fetches.

## How `numa_shard_plan` Links to Codegen

The `numa_shard_plan` metadata serves as the foundational communication mechanism between the graph transformation passes (like `numa_mlp_tp.py` and `numa_attention_tp.py`) and the actual code emission in `wrapper.py`:

1.  **Metadata Injection Pipeline**: During the `post_grad` FX graph stage, standard ATen `mm`/`addmm` nodes are matched and sliced. As nodes are modified, they are tagged with a `numa_shard_plan` (a Python dictionary containing configuration like `world_size`, `shard_d`, and logical mapping lists such as `column_parallel_nodes` and `row_parallel_nodes`).
2.  **Codegen Trigger**: When `wrapper.py` needs to generate the Triton kernel call via `PythonWrapperCodegen.wrap_kernel_call` or its Triton counterpart, it introspects the kernel `call_args` and `triton_meta` arrays. If it finds `numa_shard_plan` attached to the kernel metadata, it invokes the specific NUMA shard handler (`generate_numa_shard_allocations()`).
3.  **Parsing the Plan**: Inside `generate_numa_shard_allocations()`, the configuration dictionary is normalized. Nodes are identified as `replicated_nodes`, `column_parallel_nodes`, `row_parallel_nodes`, or `zero_on_nonzero_nodes`.
4.  **Translating Logical Shards to Physical Allocations**: The wrapper codegen generates explicit Python code containing `.clone(numa_node=sid)` calls. 
    *   **Weights (`_emit_slice_clone`)**: Weights are dynamically sliced and placed correctly via `aten.clone` and `numa_node` bindings.
    *   **Bias Terms (`_emit_zero_bias_clone`)**: Any bias associated with the `zero_on_nonzero_nodes` configuration gets initialized normally for node `0` but is populated with `zeros_like` for `node > 0`.
    *   **Outputs (`_emit_numa_empty_strided`)**: Uninitialized kernel output buffers are constructed directly on matching NUMA architectures without first-touch faults.
5.  **Injecting Metadata to the Dispatcher**: The codegen compiles all per-socket pointer variables into a grouped list (e.g., `_numa_args = [(_numa_x_0, _numa_a_0, ...), (_numa_x_1, _numa_a_1, ...)]`). This list is directly patched into the `TritonKernel.run` invocation natively over `numa_args=_numa_args`.
6.  **Outputs Reassembly (Row-Parallel)**: For final operations requiring data accumulation (`should_allreduce_outputs = True`), `wrapper.py` generates in-place M-split block copying so the runtime driver concatenates per-socket slices linearly into user-consumable format without `torch.cat`.

## Architecture Flow overview

```mermaid
flowchart TD
    subgraph Inductor [PyTorch Inductor Codegen Stack]
        Plan[Graph Node Meta: numa_shard_plan] --> Trigger[wrapper.py checks and triggers NUMA codegen]
        Trigger --> Normalizer[Normalize Plan Lists: col_parallel, row_parallel, replicated]
        
        subgraph Allocate [Data Allocation & Physical Mapping Generation]
            Normalizer --> Rep[Replicated Inputs<br>emit aten.clone NUMA X]
            Normalizer --> Slices[Sliced Weights<br>emit slice & clone NUMA X]
            Normalizer --> Biases[Biases<br>emit zeros_like for non-zero node]
            Normalizer --> Outs[Outputs<br>emit empty_strided_cpu_numa]
        end
        
        Slices --> Cache[Store static bindings in _numa_weight_cache]
        Rep --> Cache
        Biases --> Cache
        
        Allocate --> Builder[Construct _numa_args List of Lists]
        Builder --> Emission[Emit Kernel Call:]
        Emission --> Call[kernel_name.run(..., numa_args=_numa_args)]
        
        Call --> Reassemble{Is this a row-parallel output?}
        Reassemble -->|Yes| MSplit[Emit M-split Reassembly Code]
        Reassemble -->|No| Continue[Continue Graph Flow]
    end
    
    subgraph Runtime [Triton CPU Runner driver.py]
        Call -- numa_args received via PyBind --> NativeDriver[C++ Kernel Launcher]
        NativeDriver --> SocketBinding[Bind std::threads per NUMA Socket]
        SocketBinding --> Execute[Execute partitioned kernel grid]
    end
```