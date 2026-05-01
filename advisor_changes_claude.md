# Intel Advisor Roofline Setup — Changes, Mechanism, and Coverage

This report covers (1) the change made to the Triton-CPU driver, (2) Intel
Advisor's internal benchmarking mechanism for the cache-aware roofline,
(3) which files/loops actually contribute to the chart, and (4) the explicit
question: under the current setup, does Advisor measure the *full* NUMA-mode
execution (including the wrapper-emitted clones) or only the kernel `.so`
files?

---

## 1. The change made to `triton-cpu/.../driver.py`

### 1.1 File and location

The patched method lives in two hardlinked copies (same inode):

- `/home/shavin/llm-cpu-compiler/triton-cpu/python/triton/backends/cpu/driver.py`
- `/home/shavin/llm-cpu-compiler/triton-cpu/third_party/cpu/backend/driver.py`

### 1.2 Original code (before)

```python
def load_binary(self, name, kernel, shared_mem, device):
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".so") as f:
        f.write(kernel)
        f.flush()
        import ctypes
        lib = ctypes.cdll.LoadLibrary(f.name)
        fn_ptr = getattr(lib, name)
        fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
        return (lib, fn_ptr_as_void_p, 0, 0)
```

### 1.3 The bug

`tempfile.NamedTemporaryFile(... suffix=".so")` defaults to `delete=True`.
The `with` block unlinks the file the moment `load_binary` returns. The
`ctypes.cdll.LoadLibrary(f.name)` call has already resolved an in-memory
mapping for the kernel via `dlopen`, so the kernel keeps running, but the
on-disk path `/tmp/tmpXXXXXX.so` is gone immediately afterward.

When Intel Advisor finalizes its trace (after the Python process exits),
Pin walks every module recorded in the trace and tries to re-open the
underlying `.so` to read DWARF debug info, build loop tables, attribute
FLOPs to source lines, and so on. That re-open fails → the warning
`Cannot locate file '/tmp/tmpXXXXXX.so'` fires for every kernel → no loop
boundaries identified inside the kernel → no FLOP attribution → empty
roofline chart.

### 1.4 The fix

```python
def load_binary(self, name, kernel, shared_mem, device):
    dump_dir = os.environ.get(
        "TRITON_KERNEL_DUMP_DIR",
        os.path.join(tempfile.gettempdir(), "triton_cpu_kernels"),
    )
    os.makedirs(dump_dir, exist_ok=True)
    kernel_hash = hashlib.md5(kernel).hexdigest()
    so_path = os.path.join(dump_dir, f"{name}_{kernel_hash}.so")
    if not os.path.exists(so_path):
        tmp_path = so_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(kernel)
        os.replace(tmp_path, so_path)
    import ctypes
    lib = ctypes.cdll.LoadLibrary(so_path)
    fn_ptr = getattr(lib, name)
    fn_ptr_as_void_p = ctypes.cast(fn_ptr, ctypes.c_void_p).value
    return (lib, fn_ptr_as_void_p, 0, 0)
```

Properties of the new path:

- **Persistent**: the file is *not* deleted; Pin/Advisor can re-open it
  at finalization to read DWARF.
- **Deterministic name**: `<kernel_name>_<md5(kernel_bytes)>.so`. Two
  invocations of the same kernel produce the same path, so subsequent
  runs reuse the file and avoid recompilation.
- **Atomic write**: `tmp_path` then `os.replace` — avoids a partial-write
  race when multiple processes / threads load the same kernel
  concurrently.
- **Override-able**: `TRITON_KERNEL_DUMP_DIR=/some/path` lets you redirect
  the dump for archiving or for a tmpfs-backed dir.

The `-g` flag was already on the `_build` command line in
`triton/runtime/build.py:58` (`cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-g", "-o", so]`),
so debug info was being produced; it just had nowhere to live. The
content-hash-named persistent path is what makes the existing `-g` info
reachable to Advisor.

### 1.5 Resulting kernel set on disk

After one full standard + NUMA collection, `/tmp/triton_cpu_kernels/`
contains 23 unique `.so` files — 13 from the standard pass, 10 from the
NUMA pass. They are partitioned cleanly by content hash (no collisions).
Several kernel name prefixes (`triton_tem_fused_bmm_3`,
`triton_tem_fused_add_addmm_gelu_8`, `triton_per_fused__safe_softmax_4`,
…) appear twice with *different* hashes — once compiled against the full
weight shape (standard mode) and once against the per-socket sliced
weight shape (NUMA mode).

---

## 2. Intel Advisor's roofline mechanism — what it actually measures

The cache-aware roofline produced by `advixe-cl --collect=roofline`
(equivalently the survey + tripcounts pair we run by hand) requires *two*
distinct passes because they need different kinds of measurement:

### 2.1 Pass 1 — `--collect=survey`

Lightweight statistical sampling, low (~1–5×) overhead. What it captures:

- **PMU sampling** of instruction pointers at a fixed rate (typically
  ~1 ms or ~1000 samples/sec). For each sample, Advisor records:
  - the program counter,
  - the call stack,
  - the wall-clock time bucket.
- **Static disassembly** of every module Pin loads. From the disassembly
  Advisor classifies each loop body as `Scalar`, `Vectorized`,
  `Vectorized (Body)`, etc., and records the dominant ISA (SSE2 / AVX /
  AVX2 / AVX-512 / …).
- **Loop discovery**: Advisor walks the symbolized debug info (DWARF) to
  build a tree of functions and loops. The "Function Call Sites and
  Loops" column in your survey TXT is the output of this step.
- **Time per loop**: from the PMU sample histogram, Advisor attributes
  wall time to each loop / function based on which loop's address range
  was sampled.

Survey alone does not count FLOPs or model caches.

### 2.2 Pass 2 — `--collect=tripcounts --enable-cache-simulation --flop`

Heavy dynamic instrumentation via Pin (~30–100× overhead, more with
cache simulation). Three independent counters run simultaneously, all
implemented as Pin pintools:

1. **Trip-count counter** (`tripcounts`): for every loop identified in
   the survey, Pin inserts a counter at the loop header; the counter is
   incremented on each entry.

2. **FLOP counter** (`--flop`): Pin inspects every executed instruction.
   When the opcode matches a recognized FP-arithmetic pattern
   (`add{ss,sd,ps,pd}`, `mul{ss,sd,ps,pd}`, `vfmadd*`, scalar/packed
   variants for both single and double precision, plus `div`, `sqrt`,
   etc.), Pin increments a per-thread counter weighted by lane width and
   FMA multiplicity (FMA = 2 FLOPs per element). Integer arithmetic is
   counted separately in `GINTOPS`.

3. **Cache simulator** (`--enable-cache-simulation`): a software model
   of the target machine's L1 / L2 / L3 / DRAM hierarchy. Every memory
   reference Pin observes is fed through the model; the model produces
   per-loop hit/miss counts and byte traffic per level. From those
   numbers Advisor derives per-loop arithmetic intensity:

   $$
     AI_\text{level} = \frac{FLOPs_\text{loop}}{Bytes_\text{level}}
   $$

   where `Bytes_level` is the byte traffic crossing that cache level
   (L1 misses go to L2, L2 misses go to L3, L3 misses go to DRAM, etc.).

   The "cache-aware" part of the chart comes from the simulator: each
   loop ends up with *four* AI values (one per level), plotted as a
   single dot whose horizontal position depends on which roofline tier
   you are reading. The dot above the L3 ceiling at AI_L3 vs. above the
   DRAM ceiling at AI_DRAM tells you *where* the bottleneck sits.

### 2.3 Why the chart is per-loop, not per-function

A "loop" to Advisor is a back-edge in the control-flow graph identified
either statically (from DWARF loop info) or dynamically (from repeated
trip counts at the same PC). The roofline plots one dot per loop, with:

- Y-axis: GFLOPs/s achieved by that loop = `FLOPs_loop / SelfTime_loop`
- X-axis: Arithmetic Intensity at the chosen cache level

Loops with `SelfTime_loop ≈ 0` or `FLOPs_loop = 0` cannot be plotted —
their X or Y is undefined. This is the GUI's "no floating point
operations detected" message: it has loop time but no FLOP counts.

### 2.4 Roofline ceilings

Two more pieces complete the chart:

- **Compute ceilings** (horizontal lines at the top): peak GFLOPs/s of
  the CPU for SP / DP / FMA / no-FMA. Advisor measures these once with a
  micro-benchmark or reads them from a built-in CPU database; they are
  hardware capabilities, not workload data.
- **Bandwidth ceilings** (diagonal lines): peak memory bandwidth for
  L1 / L2 / L3 / DRAM. Measured by Advisor's bandwidth micro-benchmarks
  using uncore PMU events (`uncore_imc/cas_count_*` for DRAM, etc.) on
  first project load.

A loop that lands on the DRAM diagonal is DRAM-bound; one that lands on
the L3 diagonal is L3-bound but well-fit-in-L3. Above all diagonals and
below the compute ceiling means compute-bound.

---

## 3. Which files / regions are actually instrumented

### 3.1 Process layout under Advisor

When `advixe-cl --collect=tripcounts -- python advisor_bench_*.py` runs,
the OS process tree at the moment the bench's measurement loop runs
looks like:

```
advixe-cl
  └── pinbin              (Pin's launcher, replaces `python`)
        └── python3.9     (with all DSOs below mapped in)
              ├── /usr/lib/python3.9/...   (Python interpreter C code)
              ├── libtorch_cpu.so          (PyTorch ATen dispatcher + kernels)
              ├── libtorch_python.so       (Python ↔ ATen glue)
              ├── libc10.so                (Tensor metadata)
              ├── libtriton.so             (Triton runtime)
              ├── __triton_cpu_launcher.so (Triton-CPU C++ launcher; per kernel)
              ├── /tmp/triton_cpu_kernels/triton_*.so   ← the SIMD kernels
              ├── libnuma.so.1             (NUMA bind / move syscalls)
              ├── libgomp.so.1             (OpenMP runtime)
              ├── libtbb.so.12, libtbbmalloc.so.2  (allocator / TBB)
              ├── libopenblas.so.0, libsleef.so.3
              ├── libstdc++.so.6, libgcc_s.so.1, libc.so.6
              └── …
```

Pin sees and instruments **every native instruction executed in any of
those modules** (subject to `--module-filter-mode=exclude`). PMU samples
in survey can land in any of them too.

### 3.2 What survives the measurement window

We launch with `--start-paused`. The bench calls `__itt_resume()` only
after warmup. So instrumentation is *recorded* only for the 20-iteration
measurement loop. Within that window the executed code paths include:

- The Inductor-generated `output_code.py` (the wrapper), running as
  Python bytecode in `python3.9`.
- The wrapper's calls into `libtorch_cpu.so` for `aten.empty_strided`,
  `aten.clone`, `aten.copy_`, NUMA move helpers, etc.
- `__triton_cpu_launcher.so::launch()` — the C++ entry point Triton
  generated. It dispatches the grid to OpenMP / `std::thread` workers.
- The `triton_*_<hash>.so` kernel functions — the actual SIMD math.
- Standard-library glue: `libgomp` task scheduling, `libc` mem ops,
  `libnuma` `move_pages` calls (only in NUMA mode and only on first
  iteration).

### 3.3 Module filter currently in place

`run_advisor_roofline.sh` excludes the following modules from
instrumentation (they are only touched during torch.compile, which we
already exclude via `--start-paused`):

```
cc1plus, x86_64-linux-gnu-as, x86_64-linux-gnu-ld.bfd, collect2,
x86_64-linux-gnu-gcc-13, libcrypto.so.3, libssl.so.3, libgmp.so.10,
libpindwarf.so, libbfd-2.42-system.so, libbfd-2.42.so,
_ssl.cpython-39-…, _hashlib.cpython-39-…, _ctypes.cpython-39-…,
_json.cpython-39-…, _elementtree.cpython-39-…
```

Notably *not* excluded: `libtorch_cpu.so`, `libgomp.so.1`,
`libnuma.so.1`, `libtbb*`, `libopenblas.so.0`, `libsleef.so.3`,
`__triton_cpu_launcher.so`, every `triton_*.so`, and `python3.9` itself.

---

## 4. Explicit answer — does the current setup measure the *full* NUMA execution including the wrapper.py-emitted clones?

The user's question, phrased precisely:

> NUMA-mode codegen in `wrapper.py` emits clone instructions (cloning
> weights/buffers per NUMA socket) that do not exist in standard mode.
> In the current Advisor benchmarking, is that full execution
> performance measured, or just the generated `.so` files?

### 4.1 What the NUMA wrapper actually emits

The relevant emission sites in
`/home/shavin/llm-cpu-compiler/pytorch/torch/_inductor/codegen/wrapper.py`:

- Line 700–701: imports `empty_strided_cpu_numa` (a helper that
  allocates an uninitialized CPU tensor and binds its storage to a
  specified NUMA node via `c10::NUMAMove`).
- Lines 1790–1796 (`_ensure_numa_weight_cache_header`): emits
  `_numa_weight_cache = {}` once per compiled graph, a module-level
  dict that caches per-socket weight clones across forward calls.
- Lines 2098–2422 (`generate_numa_shard_allocations`): for every kernel
  call whose operands are tagged with a `numa_shard_plan`, emits the
  per-socket clone of weights and the per-socket allocation of output
  buffers, builds the `numa_args=[[ptr_n0,…],[ptr_n1,…],…]` list, and
  rewrites the `kernel.run(…)` call to pass `numa_args=…`.
- Lines 2113, 2168, 2229: literal `torch.ops.aten.clone.default(arg,
  numa_node=sid)` lines emitted into `output_code.py` for the
  per-socket weight clone.
- Lines 2256–2319 (`_emit_numa_empty_strided`): for output / activation
  buffers, emits `empty_strided_cpu_numa(shape, stride, numa_node=sid)`
  per socket — these allocations happen on every forward pass.

Two distinct cost categories show up at runtime:

| Category | Where it runs | First call only? |
|---|---|---|
| Per-socket **weight** clone | `aten::clone(x, numa_node=sid)` → memcpy in `libtorch_cpu.so` + `c10::NUMAMove` (`move_pages` syscall in `libnuma.so.1`) | **Yes** — cached in `_numa_weight_cache` dict; subsequent forward calls do a Python dict lookup |
| Per-socket **activation / output** allocation | `empty_strided_cpu_numa(...)` → `torch.empty_strided` + `c10::NUMAMove` | **No** — happens every forward call |
| All-reduce / sum-reduce of row-parallel outputs | aten add / copy_ in `libtorch_cpu.so` | Every forward call |

### 4.2 The crisp answer

**Yes — Advisor measures the full execution, including everything the
NUMA wrapper.py emits.** It is *not* limited to the `.so` kernels. Pin
instruments every native instruction reached during the resumed
measurement window, regardless of which DSO it lives in. So `aten.clone
(numa_node=sid)`, `empty_strided_cpu_numa`, the `move_pages` syscall in
`libnuma.so.1`, the OpenMP / `std::thread` orchestration in
`__triton_cpu_launcher.so`, and the row-parallel reduction all execute
under Pin and contribute to the **time axis** of the survey and to the
**byte traffic** of the cache simulator.

There are, however, two important caveats about *which view* shows that
overhead:

#### Caveat A — the **roofline scatter plot** only shows FLOP-producing loops

The roofline plots `GFLOPs/s` vs `Arithmetic Intensity`. Both axes
require non-zero FLOPs. The NUMA clone / allocation / reduction code is
**byte traffic, not floating-point math** — so those code paths
contribute zero to the FLOP counter, are excluded from the roofline
scatter, and won't appear as dots. They will still:

- bloat the byte traffic seen by the cache simulator → push compute
  loops *closer to the bandwidth ceilings* than they would be on the
  standard path,
- consume wall-clock time → reduce the GFLOPs/s achieved by the FLOP
  loops (because total measurement time is in the denominator of
  GFLOPs/s).

So NUMA overhead **does affect the roofline indirectly**: if cloning
dominates time, your kernel dot moves *down* the chart (lower
GFLOPs/s) and *left* (smaller AI because more bytes are touched per
FLOP) compared to the standard run.

#### Caveat B — to *see* the NUMA overhead explicitly, look at the survey table, not the roofline

The survey's "Total Time / Self Time" table includes every loop and
function Pin sampled. NUMA-specific entries you should expect:

- `loop in run_omp_kernels._omp_fn.0 at main.cpp:404` — the launcher's
  parallel-for body executing the kernel; dominant in standard mode.
- `loop in operator() at main.cpp:555` — the launcher's NUMA-aware
  per-thread dispatch loop; visible in NUMA mode, hands work to
  per-socket thread groups.
- `loop in launch at main.cpp:608` / `:425` / `:548` / `:479` — the
  C++ launch entry, including the cross-socket fan-out for NUMA.
- `_M_run at std_thread.h:244`, `_M_run at condition_variable:104` —
  worker threads waiting on barriers between dispatches; these were a
  significant chunk of NUMA-mode time in your last run (the 6.07s line).
- `aten::clone` / `aten::copy_` / `aten::empty_strided` symbols inside
  `libtorch_cpu.so` — the Inductor-emitted clones / allocations.
- `move_pages` / `mbind` symbols inside `libnuma.so.1` — the NUMA
  binding syscalls, only on the first iteration's weight clones (then
  cached by `_numa_weight_cache`).

`topdown_<label>.csv` and `tripcounts_<label>.csv` (the new CSVs the
report script now emits) include those entries with their CPU time and
trip counts. They are the right place to read NUMA-overhead numbers.

#### Caveat C — `--start-paused` still excludes one thing on purpose

We start Advisor paused and only `__itt_resume` after warmup. So
torch.compile, Triton kernel JIT, gcc invocations, and FX graph
compilation are *not* measured — those are one-time setup costs we
explicitly want to leave out. But `_numa_weight_cache` is populated on
the *first call to `compiled(x)` inside the measurement loop*, not
during warmup, because the wrapper allocates the cache lazily on first
forward. The first measured iteration therefore pays the per-socket
clone cost; iterations 2 through 20 hit the cache and skip it. So in
your 20-iteration measurement loop the weight-clone cost is
amortized 20×. If you want the steady-state-only number, raise
`ADVISOR_BENCH_ITERS`.

To check this empirically after a run: search the survey TXT for
`aten::clone` — its `Call Count` column tells you how many times the
clone fired during the measurement window. Should be a small constant
(the number of distinct NUMA-sharded weights, fired once each), not
proportional to `ITERS`.

### 4.3 Summary table

| What you want to compare | Where to read it |
|---|---|
| GFLOPs/s of the SIMD math itself | Roofline chart Y-axis on the kernel dots |
| AI of the SIMD math at L1/L2/L3/DRAM | Roofline chart X-axis on the kernel dots |
| Wall time spent in NUMA cloning | Survey table → loops in `aten::clone`, `aten::copy_`, `empty_strided_cpu_numa`, lines in `topdown_<label>.csv` |
| Wall time spent in launcher orchestration | Survey table → `main.cpp:404`, `main.cpp:555`, `main.cpp:608` rows |
| Wall time worker threads spend idle on barriers | Survey table → `_M_run at condition_variable:104` |
| Bytes the NUMA clone causes the simulator to count | `tripcounts_<label>.csv` byte columns aggregated across non-kernel rows |
| Whether NUMA shifted kernels closer to DRAM ceiling | Compare kernel dot positions on the roofline chart, NUMA project vs. standard project |
| Whether the chart contains NUMA-clone dots themselves | **No** — clone has zero FLOPs, so it has no dot on the roofline; read from survey/topdown CSV |

### 4.4 If you want a single end-to-end "execution time" number including NUMA overhead

The simplest aggregate is `Program Elapsed Time` from the run log
(line "Program Elapsed Time: …s") *minus* the time spent paused
(warmup + finalization), or — more cleanly — instrument the
measurement loop with `time.perf_counter()` before `itt_resume` and
after `itt_pause` and print the delta. That number includes every
NUMA-related instruction Pin saw. Multiplying it by your CPU's peak
GFLOPs / peak DRAM BW lets you cross-check the roofline interpretation
against a single end-to-end value.

---

## 5. Bottom line

- The driver patch makes the kernel `.so` files persistent and
  hash-named, which is what made Pin able to symbolize them and
  attribute time to vectorized loops.
- Advisor's roofline plots **only FLOP-producing loops**, so the SIMD
  math kernels appear as dots, while NUMA cloning / allocation /
  thread-pool barriers do not.
- Despite that, **the full NUMA execution is being measured** in the
  trace; the overhead is captured in time and byte-traffic columns and
  in the survey table — just not on the roofline scatter itself. The
  roofline reflects the NUMA overhead *indirectly* (kernel dots shift
  down/left when cloning eats wall time and adds byte traffic).
- For a direct view of NUMA cost, read `topdown_<label>.csv`,
  `tripcounts_<label>.csv`, and the `aten::clone` / `_M_run` /
  `main.cpp:555` rows in `survey_<label>.txt`.
