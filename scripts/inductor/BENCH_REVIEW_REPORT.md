# NUMA-TP decoder-block benchmark review

**Scope.** Audit of the benchmark + roofline scripts you used and an explanation of
the suspicious ~2× speedup (and why the GCP c2-standard-60 run produced a much
smaller ~1.46×).

Files reviewed:

* `scripts/inductor/simple_llm_decoder_block_speedup_cli.py`
* `scripts/inductor/simple_llm_decoder_block_paper_bench.py`
* `scripts/inductor/run_advisor_roofline.sh`
* `scripts/inductor/advisor_bench_numa.py`
* `scripts/inductor/advisor_bench_standard.py`
* Spot checks in `torch/_inductor/fx_passes/numa_*.py`,
  `torch/_inductor/codegen/wrapper.py`, and the Triton CPU launcher.

---

## TL;DR — verdict

The **timing harness itself is correct** (warmup vs measurement separation, per-iter
`time.perf_counter()`, fixed seed, `model.eval()` + `torch.no_grad()`, fresh
`torch._dynamo.reset()` per mode, `fx_graph_cache=False`, distinct `inductor_config.patch`
contexts per mode).

But the **comparison itself is not apples-to-apples**, and that — not a measurement
bug — is what produces the suspicious numbers. The single most important fact:

> Your machine has **2 NUMA nodes** (`numactl --hardware`: `available: 2 nodes (0-1)`).
> With a 2-way TP world size, the **theoretical maximum speedup is 2×**.
> You are observing **2.66–2.81×**. Anything strictly greater than `world_size` cannot
> come from tensor parallelism alone — the baseline is being penalized.

The most plausible reason for the >2× is a **thread-pinning asymmetry**: the NUMA
launcher path pins worker threads to socket-local CPUs via
`pthread_setaffinity_np` (driver.py), while the "standard" Triton-CPU path runs
through OpenMP with no explicit binding. On a 2-socket E5-2620 v3 with hyper-threading
and only 6 physical cores per socket, OS-scheduler hopping causes large remote-NUMA
penalties and L1/L2 thrashing in the baseline. So you are partially measuring
"pinned vs unpinned", not "TP vs no-TP".

The cache-clear experiment confirms there is also non-trivial **measurement
non-determinism** (2.15× cold cache vs 2.7–2.8× warm cache, same parameters — a
~30 % swing).

The cross-machine gap is consistent with this story:

| Machine | Sockets | Cores/socket | Speedup |
|---|---|---|---|
| Your local box (Xeon E5-2620 v3, 24 vCPU) | 2 | 6 phys / 12 logical | 2.15–2.81× |
| GCP c2-standard-60 | 2 | 30 | 1.46–1.54× |

With 30 cores per socket, the OpenMP baseline more easily saturates socket-local
bandwidth even without explicit pinning, so the TP-vs-baseline gap shrinks toward
the true ceiling. With only 12 logical cores per socket, the baseline is much more
sensitive to thread placement, and "fix placement via TP" reaps a disproportionate
gain that exceeds 2×.

---

## System ground-truth (this box)

```
$ numactl --hardware
available: 2 nodes (0-1)
node 0 cpus: 0 2 4 6 8 10 12 14 16 18 20 22
node 1 cpus: 1 3 5 7 9 11 13 15 17 19 21 23

$ lscpu (selected)
Model name:        Intel(R) Xeon(R) CPU E5-2620 v3 @ 2.40GHz
CPU(s):            24      Socket(s): 2     Core(s)/socket: 6     Threads/core: 2
L3 cache:          30 MiB (2 instances)    L2: 3 MiB (12 instances)

$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
schedutil               # ← dynamic frequency, not "performance"
$ env | grep -E 'OMP|KMP|MKL|TRITON' | sort
(empty)                 # ← OMP_NUM_THREADS / OMP_PROC_BIND / OMP_PLACES all unset
$ ls /tmp/torchinductor_ubuntu | head
22  2q  2r  2u  …       # ← hot Inductor on-disk cache present
```

Important consequences of the above:

1. **NUMA world size is 2** on this box. Auto-shard logic in
   `simple_llm_decoder_block_paper_bench.py::_auto_num_tp_shards` will pick `2`
   (= `min(num_heads=4, nodes=2)`). Speedup ceiling = 2×.
2. **Governor is `schedutil`**, so per-iteration latency is partly a function of
   frequency ramp. The first 1–2 measured iterations of each run are systematically
   different from the steady state. This adds variance.
3. **OpenMP is unpinned** (no `OMP_PROC_BIND`, no `OMP_PLACES`). The 24 OpenMP
   workers will hop between the two sockets at the kernel scheduler's discretion;
   this is precisely what NUMA pinning fixes.

---

## Issue list — `simple_llm_decoder_block_paper_bench.py`

### 1. Speedup > world_size is structurally impossible from TP — the baseline is unfair

This is the headline issue. The "standard" mode you compare against:

* Uses Triton-CPU codegen for the same kernels (because
  `inductor_config.max_autotune_gemm_backends = "TRITON"` is set at module import,
  line 76 — affects **both** modes). It is **not** the OpenBLAS / MKLDNN baseline a
  reader of the numbers might expect.
* Runs those Triton-CPU kernels through the launcher's **OpenMP fallback** path
  (`run_omp_kernels()` in `driver.py`), with the host's default OMP environment.
* Has **no pinning** because OMP_PROC_BIND/OMP_PLACES are unset on the host.

Then the NUMA mode runs those same kernels but through the launcher's
`std::thread` group + `pthread_setaffinity_np` path, which binds each worker to a
specific CPU read from `/sys/devices/system/node/nodeN/cpulist`.

So the numerator and denominator differ by **(a)** TP sharding and **(b)** thread
binding policy. (b) alone can be worth a substantial factor on 12-core sockets.

**What "fair" looks like:**

```bash
# Pin the standard baseline too, so the comparison isolates TP.
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=24 \
  python scripts/inductor/simple_llm_decoder_block_speedup_cli.py …

# Or run both under numactl with an explicit interleave / single-node policy:
numactl --interleave=all python … speedup_cli.py …

# Cross-check by also running standard with --cpunodebind=0 --membind=0 (forces
# all OMP work + memory onto one socket, eliminating remote-node hits but keeping
# only 12 logical cores). If this run is much faster than the unpinned default,
# the prior 2.7× was inflated by NUMA-hopping in the baseline.
```

You should expect the "fair" speedup on this box to be **< 2×**, probably in the
1.4–1.8× range — closer to what the GCP run reports. If it isn't, the next
suspect is OMP barrier overhead being eliminated by the std::thread launcher,
which is a real (and legitimate) NUMA-path advantage but should be stated as
such, not lumped into "tensor parallelism".

### 2. Run order is fixed (standard first, NUMA second)

`_mode_run_order` (line 612) hard-orders `("standard", "numa_tp")`. The second run
benefits from:

* Warmer page cache, warmer code/uop caches, hotter cores under `schedutil`.
* Whatever heap fragmentation patterns the first run left.

This is also why `--warmup 1 vs --warmup 10` shifted the number visibly
(2.73 → 2.81 for the 1024-seq case): more warmup of the standard run means a
longer "favor-numa" period of process state, not strictly a more-converged
measurement.

**Mitigation:** alternate or randomize order across configs and report median over
≥3 trials of each ordering.

### 3. Result varies ~30 % with disk-cache state

Your own experiment:

| Cache | Speedup |
|---|---|
| Warm `/tmp/torchinductor_ubuntu` + `~/.triton/cache` | 2.66 / 2.73 / 2.81 |
| `rm -rf` both, then run | 2.15 |

Same parameters, same box, same script. A 30 % swing from cache state alone is
the clearest evidence that the harness, while internally consistent, is not
producing a stable point estimate.

Causes:

* `fx_graph_cache=False` is set, but the **Inductor on-disk artifact cache** under
  `/tmp/torchinductor_ubuntu` and the **Triton kernel binary cache** under
  `~/.triton/cache` are still active. Different first-iter behavior between the
  cold and warm cases produces different background-thread load during warmup.
* `inductor_config.compile_threads` is left at the default — Inductor will spawn
  many parallel compile workers that may still be finishing during the early
  warmup iterations. The colder the cache, the more compile work overlaps with
  early warmup.

**Mitigations:**

* Force a clean state per benchmark run: clear both caches and start with
  `inductor_config.compile_threads = 1` (as the advisor scripts already do).
* Increase warmup. 10 iterations is small for a Triton-CPU first run with cold
  caches; the per-iter time is dominated by background compile finishing.
* Drop the first N measured iterations (e.g. `latencies[5:]`) before computing
  mean/percentiles so any trailing JIT spillover is excluded.
* Run multiple trials and report the median of medians.

### 4. `compile_proxy` saves only iter 0

`time_forward_per_iter` (line 303) records `compile_proxy = dt` only for `i == 0`.
That's fine as a "first-run" diagnostic, but it is not a sufficient compile-time
estimate when Inductor `compile_threads > 1`: lazy autotune / Triton kernel
compile can still be in flight on subsequent iterations. Without an explicit
join, you cannot tell from the recorded `latencies_ms` whether iter 1 included
residual compile cost.

Recommendation: either (a) set `inductor_config.compile_threads = 1` like the
advisor benches do, or (b) call `torch._dynamo.eval_frame.compile_threadpool` /
the Triton compile pool's join hook before starting measurement, or (c) simply
warm with many more iterations and discard early ones.

### 5. `inductor_config.max_autotune_gemm_backends = "TRITON"` is set at import

Line 76 in `paper_bench.py`. This means `simple_llm_decoder_block_speedup_cli.py`
inherits it the moment it imports `paper_bench`. The implication is:

* The "standard" baseline is **also** Triton-CPU GEMMs, not the conventional
  OpenBLAS path that `USE_MKLDNN=0 BLAS=OpenBLAS` would otherwise give you.
* That baseline therefore exercises the same launcher and the same OMP-fallback
  thread fan-out as the NUMA path — which is good for isolating "NUMA shard plan"
  from "GEMM backend" but **bad for telling a paper-style story of "compared to
  the default torch.compile CPU path"**, because that default does **not** force
  Triton-only.

If your story is "NUMA TP vs default torch.compile on CPU", you need a third
control with `max_autotune_gemm_backends` left at default. If your story is
"NUMA TP launcher path vs OMP launcher path holding GEMM backend constant",
state that explicitly.

### 6. Both modes run inside the same Python process, with shared global state

Even with `torch._dynamo.reset()` and `fx_graph_cache=False`:

* The Inductor on-disk artifact cache is shared.
* The Triton kernel binary cache is shared.
* Process-level allocators (`malloc`, the Python free-list), the libtorch tensor
  allocator's free pools, and any thermal/frequency state are shared.

For a **publication-grade** claim, run each mode in a **fresh subprocess** (e.g.,
`subprocess.run([...])` per mode), with caches cleared in between, and aggregate
externally. The current setup is fine for development sanity but invites the
exact "warm vs cold" swing you observed.

### 7. NUMA path's `compute_shard=0` is an invariant assumption

`speedup_cli` and `paper_bench` always set `_numa_*_compute_shard = 0`. That is
fine, but the launcher is then the only thing balancing work across sockets via
the per-shard `numa_args[i]` lists. If for any reason the launcher silently
dropped a shard (e.g., `numa_list` missing in `kernel_metadata`), the kernel
would fall back to OpenMP with the *full* (un-sharded) inputs and you would still
see *some* speedup from pinning alone — without TP actually firing. Worth
sanity-checking once with `_numa_mlp_tp_debug = True` /
`_numa_attention_tp_debug = True` on a single config, and confirming the
generated `output_code.py` carries the expected `numa_args=[…]` keyword on every
hot kernel call. The codegen entry points are at
`torch/_inductor/codegen/wrapper.py:1812` and `:1882`.

### 8. Per-iter sink

```python
def run() -> None:
    nonlocal y
    y = compiled(x)
```

Correct (`y` keeps the latest output alive long enough for the next iter to start
without hitting use-after-free, while not forcing materialization beyond what
inductor already does). No issue.

---

## Issue list — `simple_llm_decoder_block_speedup_cli.py`

This is a thin wrapper over `paper.run_one_mode`. It inherits all of the issues
above. One additional thing to know:

* It calls `run_one_mode("standard", …)` then `run_one_mode("numa_tp", …)` and
  prints `std_mean / numa_mean`. Because `run_one_mode` is order-fixed and
  in-process, **every observation from this CLI inherits the warm-second-run
  bias**.

Otherwise the wrapper is sound: argparse maps to the exact kwargs `paper`
expects, NUMA setup errors abort with non-zero exit, and only the speedup goes
to stdout.

---

## Issue list — roofline scripts

### `run_advisor_roofline.sh`

* Line 97 (`echo "  TP shards (NUMA)    : ${ADVISOR_BENCH_TP_SHARDS}"`) **runs
  before** the auto-detect block (line 144). Combined with `set -euo pipefail`
  on line 60, invoking the script **without** exporting
  `ADVISOR_BENCH_TP_SHARDS=…` will trip `set -u` ("unbound variable") and the
  script will exit before ever reaching the auto-detect. Either move the echo
  below the auto-detect, or make it `${ADVISOR_BENCH_TP_SHARDS:-(auto)}`.
* The two-pass strategy (survey → tripcounts+cache-sim) is correct and follows
  Intel's recommendation. `--start-paused` plus ITT pause/resume around the
  measurement loop only is the right way to keep `torch.compile`, gcc, autotune
  and Triton-JIT out of the trace.
* `--static-instruction-mix` on survey is the correct workaround for the long
  "Dynamic Instruction Mix" stall — fine.
* The default module filter excludes the right things (gcc/binutils/Python crypto)
  while keeping `libtriton.so`, `__triton_cpu_launcher.so`, `tmp*.so`, and
  `libgomp.so` instrumented.

### `advisor_bench_numa.py` / `advisor_bench_standard.py`

* `inductor_config.compile_threads = 1` (and `TORCHINDUCTOR_COMPILE_THREADS=1`
  via env) — good, this is what you want under Advisor.
* `_load_itt()` discovers the already-mapped `libittnotify*.so` from
  `/proc/self/maps` rather than dlopening a fresh copy. Important — the long
  comment at the top of `advisor_bench_standard.py:39-47` correctly explains why
  (without this, tripcounts stays paused → GFLOPS=0). Keep it.
* The numa script defaults to `TP_SHARDS=4`. On *this* box (2 nodes) running
  `python advisor_bench_numa.py` directly without env override would either
  fail in `run_one_mode` validation or produce a misconfigured run. The shell
  driver overrides via env, so going through `run_advisor_roofline.sh` is safe;
  invoking the python script directly is not.
* Warmup defaults are `WARMUP=2 ITERS=20`. That is enough for the *roofline*
  measurement (you only care about steady-state arithmetic intensity), but it is
  too low for *latency* claims. Don't reuse these defaults to derive speedup
  numbers.
* Both scripts call `_set_numa_enabled(True)` only in the NUMA bench (correct);
  the standard bench leaves NUMA off (correct). And both call
  `torch._dynamo.reset()` before `torch.compile` and disable `fx_graph_cache`,
  so the compile is genuinely fresh.

The roofline harness is methodologically the cleanest of the three — it is the
right tool to verify whether the measured wall-time speedup is actually backed
by changes in arithmetic intensity / memory traffic, which is what would
distinguish "real TP gain" from "thread-pinning gain".

---

## Spot check — NUMA implementation files (no issues that explain the speedup)

* `numa_tp_common.py::get_numa_world_size` first reads `_numa_*_world_size` from
  `inductor_config` (which `run_one_mode` always sets to `shards`), then falls
  back to `_num_numa_nodes()`. Correct.
* `numa_mlp_tp.py` and `numa_attention_tp.py` both gate on `world_size >= 2` and
  set `numa_shard_plan` on the relevant nodes; the wrapper consumes it (see
  `wrapper.py:1812 / :1882`) only when `world_size >= 2`. So when standard mode
  patches `_numa_*_tp = False`, neither pass fires and codegen does not emit
  `numa_args=` — which means the standard run cannot accidentally execute the
  pinned-thread path. Verified.
* The launcher (`driver.py`) only takes the std::thread + affinity path when
  `kernel_metadata.numa_args` is present, otherwise falls through to
  `run_omp_kernels()`. So your "standard" run is genuinely the OMP path.

There is no implementation bug that would invalidate the comparison; the issue
is purely that the comparison is **between two runtime regimes that differ in
more than just TP**.

---

## Why is the speedup higher on this box than on c2-standard-60?

Both machines have 2 NUMA nodes, so the *theoretical* ceiling is the same (2×).
The *measured* gap differs because the baseline efficiency differs:

| Aspect | E5-2620 v3 (this box) | c2-standard-60 |
|---|---|---|
| Logical CPUs / socket | 12 (6 cores × 2 SMT) | 30 |
| L3 / socket | 15 MiB | larger (Cascade Lake) |
| Memory channels | 4 (DDR4-1866) | 6 (DDR4-2933) |
| Inter-socket interconnect | QPI 8 GT/s | UPI 10.4 GT/s |

The NUMA penalty for an *unpinned* baseline is largest where
(remote-vs-local memory ratio is worst) × (per-socket bandwidth headroom is
small) × (cross-socket interconnect is slow) — all three favour a *bigger
unpinned-vs-pinned gap* on this box. So the "TP" speedup ratio you see here is
inflated by everything that pinning fixes. On the c2 the OS scheduler has more
headroom to keep workers near their data even without explicit pinning, the
unpinned baseline is closer to optimal, and the gap to the pinned NUMA-TP path
shrinks toward the true ceiling.

Both numbers are likely "real" — they just answer **different questions**:

* **2.7× on this box** ≈ "what does (NUMA-TP launcher + pinning + sharding) win
  vs. (unpinned OMP fallback)?"
* **1.46× on c2** ≈ closer to "what does TP sharding alone win vs. baseline?"

---

## Recommendations (no code changes yet — for your approval)

In rough priority:

1. **Add a fair baseline.** Run standard with explicit OMP pinning
   (`OMP_PROC_BIND=close OMP_PLACES=cores`, optionally with
   `numactl --interleave=all`). Report both:
   * "NUMA-TP vs unpinned standard" (what you have now), and
   * "NUMA-TP vs pinned standard" (the speedup attributable to TP itself).
2. **Stabilise the harness.** Pin governor to `performance` for the duration of
   the bench, set `OMP_NUM_THREADS` explicitly, set
   `inductor_config.compile_threads = 1`, and discard the first N measured
   iterations.
3. **Run each mode in a fresh subprocess** with caches cleared in between, so
   warm-vs-cold cache state never enters the speedup ratio.
4. **Alternate run order** (standard, numa, numa, standard) and report the
   median across orderings to remove first-vs-second bias.
5. **Cross-validate with the roofline run.** If the Advisor cache-aware roofline
   shows the NUMA-TP kernels at noticeably higher achieved bandwidth or at
   higher arithmetic intensity than the standard kernels, the speedup is
   real-and-from-TP. If they're at the same point on the roofline but standard
   simply runs longer, the gap is from pinning / scheduling, not from TP.
6. **Cosmetic:** fix the `set -u` / `ADVISOR_BENCH_TP_SHARDS` ordering bug in
   `run_advisor_roofline.sh`.

If you want, once you've reviewed this report I can implement any subset of (1)–(6)
as concrete edits.
