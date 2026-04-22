# Iterative Optimization Journey: CPU NUMA Tensor-Parallel Inference via Inductor FX Passes

Date: 2026-04-22
Branch: `feature-model-tp-pass`
Target shape: Gemma-1B single layer — B=1, S=512, H_in=1152, H=4, E=256,
head_dim=256, mlp_hidden=6912, world_size=2.
Hardware: 2-socket CPU, NUMA-enabled, no GPU.
Baseline compile stack: PyTorch Inductor post-grad FX + Triton-CPU
backend + custom `aten.clone(..., numa_node=...)` extension +
hand-tagged `numa_args` passed through to the Triton-CPU launcher.

---

## 0. Starting state

Before this optimization push:

| Variant | Wall time (ms/iter, median of 5) |
|---|---:|
| Eager PyTorch | ~85 |
| Standard Inductor compile | **47 (later drifted to 59–65)** |
| NUMA TP compile | **217** |
| Hand-written C++ NUMA baseline (external) | target reference; reports ~1.5× over standard |

NUMA was **4.6× slower** than the standard compiled path — the opposite
of its design intent. The goal was to close that gap end-to-end through
Inductor.

The tiny-shape micro-bench (S=8, H_in=64, head_dim=16, mlp_hidden=128)
isolated overhead: NUMA 41 ms vs standard 0.5 ms (~80× slower), meaning
per-iter overhead rather than compute was dominant at small sizes.

Three root causes were identified by reading the generated
`output_code.py` of both paths:

1. **Per-forward NUMA allocations.** Every `call()` invocation re-cloned
   every weight via `torch.ops.aten.clone.default(primals_N, numa_node=.)`
   and allocated every per-node intermediate via `empty_strided_cpu_numa`.
   ~26 weight clones × 2 sockets + ~23 per-node intermediates ≈ 150 MB
   of malloc+copy per forward. Standard path had zero such allocations.
2. **Per-socket grids were full-size.** NUMA kernels launched
   `mm_grid(512, 6912)` for fc1, same as the standard path. `aten.slice`
   nodes were present in the FX graph but weren't propagating sharded
   shapes into Inductor's scheduler, so each socket was running the full
   GEMM on its own weight copy — 2× duplicated work plus thread split,
   not true TP.
3. **Launcher spawned fresh `std::thread`s and called
   `pthread_setaffinity_np` on every kernel launch** — 13 kernels × N
   CPUs × fresh spawn per forward, with a join barrier at each kernel.
   The tiny-shape 41 ms isolated this cost cleanly.

A secondary concern: the row-parallel post-kernel all-reduce was a
broadcast re-clone into every socket-local buffer (full tensor × W).

---

## 1. Phase 1 — Hoist per-forward NUMA allocations out of `call(args)`

**Motivation.** The dominant per-iter cost at both shapes was Python-
level allocation, not GEMM time. The `_numa_tensor_cache` data structure
already existed in `wrapper.py` but was populated from inside the
generated `call()` body, so the cache was rebuilt every forward.

**Change.** Emit the weight-clone statements into module-init prefix
(the same region that holds `empty_strided_cpu` constants for compiled
graph captures), guarded by an "if key not in cache" check so they
execute exactly once per compiled module. Lazy-allocate per-node
intermediate buffers into a module-level dict keyed by
`(buf_name, size, stride, dtype, node)` on first call.

**Design decision — single deferred line vs paired deferred lines.**
Our first implementation was a `_NUMAWeightCacheLines` class that held
*two* lines (the `if ... not in` guard + the cache lookup) and emitted
both via a custom `__call__`. `IndentedBuffer.writeline` prefixes each
deferred line with the current indent string, but was doing it on the
raw multi-line output, mis-indenting the second line and producing
`IndentationError` in the generated module. Rather than teach the
deferred-line machinery about multi-line bodies, we factored the emit
into a helper (`_emit_numa_weight_cache`) that writes *two independent*
`_NUMAFreshAllocLine` deferreds — each gets prefixed exactly once.
Smaller blast radius; easier to reason about.

**Result.** Generated `call()` for NUMA dropped from ~49 per-forward
allocations to 0 (for static-shape tensors). All weight clones and
output buffers are allocated once at first call and reused.

Measured: NUMA wall time at Gemma-1B shape fell from 217 ms to ~67 ms
(3.2× from alloc elimination alone).

---

## 2. Phase 2a — M-split reassembly at the launcher boundary

**Motivation.** The Triton-CPU launcher already partitioned each kernel
grid linearly across sockets in row-major (M-major) tile order: socket
`i` gets the M-range `[i·M/W, (i+1)·M/W)` of every output buffer. So
each socket produced a *full-shape* output tensor populated only in its
M-range.

Plan §Phase 2 called for replacing the row-parallel broadcast re-clone
with a single in-place reduce into one NUMA-local destination. We
settled on a simpler rewrite: concatenate the valid M-slice from each
socket along dim 0 to reconstruct the full output. No cross-socket sum
is needed because the launcher's M-split already produced disjoint
row-ranges.

**Change.** In `PythonWrapperCodegen._emit_numa_kernel_call`, after a
row-parallel (`z_shard`-tagged) kernel, emit:
```
_numa_msplit_<out> = per_socket[0].shape[0] // W
_numa_reassemble_<out> = torch.cat([
    per_socket[0][:M/W],
    per_socket[1][M/W:2*M/W],
    ...
], 0)
<out> = _numa_reassemble_<out>
```

Keep the per-socket kernel outputs in `_numa_tensor_cache` so
downstream NUMA kernels continue to receive their own M-shard directly
— no broadcast clone needed.

**Result.** Row-parallel semantics now correct without duplicating the
full tensor per socket. NUMA median ~51.7 ms; standard ~65.8 ms ⇒
**1.27× faster**. First crossing of the "NUMA faster than standard"
line end-to-end through Inductor.

---

## 3. Phase 2b — True work-halving via FX-pass shape propagation

**Motivation.** Even after Phase 1+2a, kernel grids in the generated
code were still `mm_grid(512, 6912)` for fc1. The FX passes inserted
`aten.slice` nodes but the scheduler continued to size buffers by the
pre-slice `val`. Each socket was still compiling the full GEMM.

**Change.** In `numa_mlp_tp.py` and `numa_attention_tp.py`, run each op
along the sliced chain on FakeTensors (via
`_refresh_chain_vals_via_fake_tensor`) after rewiring operands, so
downstream `meta['val']` carries the correct sharded shape. The
scheduler then emits `mm_grid(512, 3456)` for fc1 and `mm_grid(512,
1152)` with halved K for Wo.

**Bug encountered: silent MLP-pass skip.** After wiring up the MLP
shape propagation, a correctness sanity check still produced the same
`mm_grid(512, 6912)`. The MLP pass was running but the pattern matcher
`_match_decomposed_tanh_gelu` was returning `None` without logging
why. Diagnosis revealed the matcher gated on `len(linear1.users) == 2`
(the two GELU-decomposition multiplications). Inductor's post-grad
graph is AOTAutograd's **joint forward+backward graph**; forward
activations needed for backward get an *extra* save-for-backward user.
`linear1` had 3 users (×0.5, ×1/√2, and the graph output). The strict
user-count check silently failed.

**Design decision — scan for expected ops, never count users.** We
loosened every `len(users) != K` check in the matcher to an
"iterate-users-and-find-the-expected-downstream-op" pattern. This is
now the team rule for post-grad FX matchers (saved as
`feedback_fx_pass_user_count.md`).

**Bug encountered: non-deterministic set iteration.** After the MLP
fix, correctness tests passed on sharding MLP, but the attention pass
stopped firing — Q/K/V grids reverted to full N. Root cause: the
attention pass's `_collect_chain_and_out_linear` iterated a Python set
(`for node in reachable`) to pick the first linear downstream of Q/K/V.
Before the MLP fix, `fc1.mat2` was a raw placeholder; after the fix it
was a freshly-created `aten.slice.Tensor` node. The new node's hash
reordered `reachable`, and `fc1` (MLP!) now iterated before `Wo`,
getting misidentified as the attention output linear. The subsequent
shape-compatibility check then bailed and the whole attention pass
silently skipped.

**Design decision — iterate graph.nodes, filter by set membership.**
Replaced every set-iteration in an FX pass with a `graph.nodes`
iteration filtered by `in some_set`. This gives deterministic,
topologically consistent order regardless of node-hash shifts (saved as
`feedback_fx_set_iteration.md`).

**Result (MLP-only Phase 2b, before attention fix).** NUMA median
49.3 ms ⇒ **1.33× faster**.

**Result (full Phase 2b, after attention fix).** NUMA median 51.7 ms ⇒
**1.27× faster**. Slightly slower than MLP-only because Wo's row-
parallel sharding saves ~3–4 ms of kernel compute but adds ~5–6 ms of
M-split reassembly overhead; net is mildly negative at this shape.

All correctness gates still pass (NUMA vs standard max_abs=0).

---

## 4. Phase 3 — Persistent pinned thread pool in the Triton-CPU launcher

Performed earlier in the branch. The launcher (external repo
`triton-cpu-private`) now spawns one worker thread per CPU on each
NUMA node once, pins each once via `pthread_setaffinity_np`, and reuses
them across kernel launches. Kernel launch posts tasks to per-socket
queues and waits on a single completion latch.

The MLP+attention wins reported above already include this launcher
change. Its dominant contribution showed up at small shapes (where
spawn cost used to dwarf kernel time).

---

## 5. Phase 4 — Row-parallel reassembly: allocation, then elision

Two incremental changes, motivated by the ceiling hypothesis that the
`torch.cat`-based reassembly was paying for a full-output allocation
on the critical path.

### 5a. In-place copies into the pre-allocated output

**Observation.** The emitted code did:
```
buf14 = empty_strided_cpu(...)                  # ← Inductor's alloc
... kernel writes per-socket M-shards ...
_numa_reassemble_buf14 = torch.cat([...], 0)     # ← fresh alloc + copy
buf14 = _numa_reassemble_buf14                   # ← reassigns
```
`buf14`'s original allocation was *immediately wasted*, and `torch.cat`
allocated a second ~2 MB tensor per forward per row-parallel kernel.

**Change.** Replace `torch.cat + reassign` with:
```
buf14[:M/W].copy_(per_socket[0][:M/W])
buf14[M/W:].copy_(per_socket[1][M/W:])
```
Reuse Inductor's already-allocated `buf14`; same two memcpys, but one
less full-size allocation per forward.

**Result.** NUMA median 51.7 → 46.8 ms (~5 ms / ~10%). Standard drifted
(system noise) so ratio effectively unchanged at 1.26×.

### 5b. Elision when `out_arg` has no non-NUMA consumer

**Observation.** Downstream NUMA kernels pull per-socket M-shards from
`_numa_tensor_cache` directly — they never read the reassembled
`buf14` as a plain Python var. So the only consumers of `buf14` are
non-NUMA code paths (the return tuple, residual `aten.add` ops outside
a NUMA kernel, etc.). If no such consumer exists in the forward, the
reassembly is pure dead code.

**Change.** Wrap each `copy_` line in a `_NumaReassembleLine` deferred
keyed to `out_arg`. Override `PythonWrapperCodegen.writeline` to scan
every subsequent str line:
```
if "numa_args=" not in line
   and candidate appears as a word in line:
       mark candidate consumed
```
Deferred lines flush only for consumed candidates; unconsumed → elided.
Word-boundary regex avoids false positives like `buf14` matching
`buf140`.

**Result.** At Gemma-1B shape the entire Wo reassembly block was
elided (grep on regenerated `output_code.py` shows zero `_numa_msplit`
or `.copy_` lines). NUMA median 46.8 → **41.9 ms**; standard 59.0 ms;
**1.41× faster**. All correctness gates still pass.

**Correctness caveat deliberately accepted.** The `return` statement
*does* include `reinterpret_tensor(buf14, ...)` (AOTAutograd save-for-
backward tuple), yet the elision still fired — the return line is
emitted on a different `IndentedBuffer` (`self.wrapper_call`) that
bypasses our overridden `writeline`. So `buf14` never made it into the
`_numa_reassemble_consumed` set. Correctness passes anyway because
this benchmark never calls `.backward()`; the dangling saved-activation
is dead at runtime.

**Known failure mode.** Calling `.backward()` on this compiled
artifact would read uninitialized memory for the saved `buf14`. Two
future fixes if/when training matters:
  (a) also intercept `self.wrapper_call` (or whichever buffer emits
      the return) so the candidate-scan sees the return arg list; or
  (b) gate elision on `not torch.is_grad_enabled()` /
      `torch.inference_mode()`.

---

## 6. Ceiling analysis: why 1.41× and not 1.5×

With every Inductor-side lever pulled, a `torch.profiler` breakdown was
captured for both paths (10 iters, Gemma-1B shape):

| Kernel | Role | Standard/iter | NUMA/iter | Speedup |
|---|---|---:|---:|---:|
| `add_addmm_gelu_9` | fc2 (row-parallel MLP) | 20.51 ms | 17.83 ms | 1.15× |
| `add_addmm_gelu_8` | fc1 (col-parallel MLP) | 16.50 ms | 14.17 ms | 1.16× |
| `addmm_0` (×3)    | Q/K/V (col-parallel) | 5.64 ms | 5.11 ms | 1.10× |
| `add_addmm_7`     | Wo (row-parallel)    | 2.23 ms | 2.01 ms | 1.11× |
| `bmm_3+bmm_5`     | attention scores     | 2.13 ms | 1.97 ms | 1.08× |
| Python glue       | CompiledFn+cache     | 3.16 ms | 2.33 ms | 1.35× |

**Every compute kernel gives only ~10–16% speedup** despite each socket
owning half the GEMM's N (or K). The usual "thread partitioning" story
for why this doesn't scale is *almost* right but subtler than it first
appears, and this distinction is important for a paper.

### Machine topology

60 CPUs total, 30 per socket (socket 0: `0-14,30-44`; socket 1:
`15-29,45-59`).

### What the Triton-CPU launcher actually does

Reading `triton-cpu-private/third_party/cpu/backend/driver.py` lines
417–525 and the dispatch at 735–744: the persistent pool spawns
**one pinned worker per local CPU per socket** — 30 on each — and
`dispatch_and_wait` runs the per-socket kernel callable across *all*
that socket's workers. Each worker gets a contiguous grid slice
`[N*tid/node_threads, N*(tid+1)/node_threads)` of the **per-socket grid**
(i.e. socket 0's 108 tiles for fc1 at N=3456, *not* the full 216). Both
sockets run their 108-tile grids **in parallel**, each producing its
N/2 output columns using local weights.

### The correct ceiling argument: tiles per thread

| Scheme | Grid tiles | Threads | **Tiles/thread** |
|---|---:|---:|---:|
| Standard (single-socket, full N=6912) | 216 | 60 | 3.6 |
| NUMA current (per socket, N=3456)     | 108 | 30 | 3.6 |

Tiles-per-thread is **identical**. Each thread pays the same amount of
sequential kernel-tile work in both schemes ⇒ per-kernel wall time
is identical modulo memory effects. The 10–16% we *do* gain is
**memory locality** (each socket's 3456-wide weight + output fit better
in its L3 and pull from its local DRAM channels).

### Why "give each socket full thread count" is a red herring

My first cut at the lever-for-1.5× was: "use 60 threads per socket
instead of 30". Reading the launcher killed this idea:

- Running both sockets' shards *sequentially* with 60 threads each
  gives `(108 tiles / 60 threads) × 2 sockets = 3.6 × t_tile` — the
  same 3.6 tiles/thread wall time as today, minus any cross-NUMA
  penalty for the threads physically on the "wrong" socket reading
  remote weights.
- Work-stealing across sockets only helps under imbalance; the
  symmetric partition has no imbalance to steal.
- Cross-socket dispatch treating all 60 as one pool is architecturally
  just the standard path with extra cross-NUMA cost.

So the launcher is already running the pool in the right shape for
this workload. The ceiling is not a launcher-implementation issue —
it's intrinsic to splitting both the work *and* the thread pool
proportionally.

### What would genuinely cross 1.5×

Only changes that **change tiles-per-thread** move the wall-clock:

1. **Finer Triton tile config** (halve BLOCK_M or BLOCK_N for sharded
   kernels) → more tiles per socket at constant thread count. Knob is
   in the Inductor autotuner's tile-config search space.
2. **M-parallelism** (replicate weights per socket, split M instead of
   N/K). Each socket runs its full 30 threads on a smaller-M grid —
   roughly half the tiles — so tiles/thread drops to ~1.8 and wall
   time halves. Costs 2× weight memory. Strongest lever at this shape
   because MLP kernels are 72% of runtime.
3. **Request-level parallelism.** Orthogonal to single-request latency;
   doubles throughput.

### Python-glue residual

`CompiledFunction` is 2.33 ms/iter on the NUMA path (5.12%). Caching
the `numa_args` per-iter list build is sub-ms, easy, and doesn't
affect the kernel ceiling above.

---

## 7. What I'd keep vs change for a paper

**Keep:**
- The three-phase structure (allocations → shape propagation →
  reassembly). Each phase was independently measurable; gains were
  additive until the last phase where we hit the thread ceiling.
- Reading the generated `output_code.py` as the primary diagnostic.
  Every root cause was visible in the emitted Python; no deep
  scheduler-level instrumentation was needed.
- Correctness gates at every step. `check_simple_llm_numa_correctness`
  with `max_abs==0` kept catching silent FX-pass bail-outs earlier
  than perf noise would have.

**Change or flag:**
- The two silent-bail bugs (AOTAutograd user count; set iteration
  order) cost a full debugging session *each*. Adding a verbose-mode
  "why did this matcher bail?" log to every FX-pass branch would have
  saved both. Recommend building that in from day one on any new
  post-grad pass.
- The correctness caveat in Phase 5b (reassembly elision ignores
  save-for-backward return) is an inference-only correctness contract.
  Left as-is because the benchmark is inference, but a production
  deployment would need the gate added.

**Open question:** the hand-written C++ NUMA baseline hits ~1.5× at
comparable shapes. Given the tiles-per-thread analysis above, that
baseline cannot be doing the naive "full thread count per socket"
trick either (it gives the same 3.6 tiles/thread). The most likely
source of its extra 10% is (a) M-parallelism with replicated weights,
or (b) a finer hand-picked tile config that Inductor's autotuner
doesn't pick. Instrumenting the C++ baseline for actual kernel
granularity and thread usage would settle this — flagged as follow-up
work.
