# NUMA-TP Reconciliation Defect

Investigated 2026-09-25/26 on `hpgen9-2`. Branch `fix/numa-tp-row-parallel-allreduce`.

## Summary

The NUMA-TP path never combines per-socket results, so it returns an unwritten
output buffer. On GPT-2 117M the returned tensor is 98,303 of 98,304 elements
exactly zero, while 29 NUMA-tagged kernels run normally.

This is not the 4-socket regression we set out to investigate. It reproduces at
`world_size=2` — the paper's configuration — and in every archived benchmark
artifact on this machine.

| Check | Result |
| --- | --- |
| Paper branch `54581fbdb4`, W=2, all four passes | all-zeros output, **0** reconciliation lines |
| Paper branch, W=4 | identical failure signature |
| Archived `numa_tp` runs from May 2026 sweeps | 26 of 26 NUMA-active, **0** with any reconciliation |

The kernels execute, so the recorded timings measured real compute. What they
did not include is the cross-socket reduction — the communication cost tensor
parallelism exists to trade against.

The defect sits in hardware-independent code generation, not in anything
specific to this server's socket count.

## How reconciliation is meant to work

NUMA-TP shards each transformer block across sockets and runs one kernel
instance per socket. Because each socket computes only part of the answer, the
pieces must be recombined before the value leaves the parallel region. Section
II-E of the paper describes two paths, selected by a semantic output tag
attached to each generated value:

| Output kind | Produced by | Reconciliation |
| --- | --- | --- |
| Terminal partial sum | Row-parallel projections (MLP down-proj, attention out-proj) | All-reduce: `Z = Σᵢ Zⁱ` |
| Intermediate activation | Column-parallel projections, fused pointwise ops | None — stays socket-local |

In the generated wrapper this decision is one predicate:

```python
should_allreduce_outputs = numa_role == "z_shard"
```

When the tag is absent, a fallback heuristic infers it from the kernel's
arguments — a kernel qualifies if it consumes a row-parallel weight and no
column-parallel weight:

```python
should_allreduce_outputs = has_row and not has_col
```

Both mechanisms must fail for a partial sum to be silently dropped. Both do.

## The defect: the output tag never arrives

`numa_role` is `None` for every kernel. The primary mechanism never fires.

Instrumenting the reconciliation decision per kernel on a 2-layer GPT-2 with
only `_numa_mlp_tp` enabled:

```
[RECON] kernel=triton_tem_fused_addmm_gelu_9  role=None allreduce=False outs=['buf23']
[RECON] kernel=triton_tem_fused_addmm_1       role=None allreduce=False outs=['buf24']   <- layer 0 MLP down-proj
...
[RECON] kernel=triton_tem_fused_addmm_1       role=None allreduce=True  outs=['buf50']   <- layer 1 MLP down-proj
[RECON] kernel=triton_per_fused_..._12        role=None allreduce=False outs=['buf54']   <- graph output
```

Two failures compound:

1. **The tag is lost.** `role=None` on all 29 kernels — the FX passes'
   partitioning metadata never reaches the wrapper code generator.
2. **The fallback is unreliable.** It fires for `buf50` and nothing else,
   including `buf24`, layer 0's structurally identical MLP down-projection.

Layer 0's partial sums are discarded, and every later layer computes on
corrupted input. The error compounds through the stack.

A third gap sits at the end: the final replicated LayerNorm is the graph's
return value. It is an intermediate activation by the tag's logic, so it stays
socket-local — but there is no next local kernel to forward it to, and nothing
writes the buffer the caller receives.

## Evidence 1: live reproduction on the paper branch

Paper branch `wrapper.py` (`54581fbdb4`) with the benchmark harness's exact
flag set:

| Configuration | max_abs | max_rel | Output |
| --- | --- | --- | --- |
| GPT-2 117M, B=1, S=128, **W=2** | 3.861e+00 | 1.000e+00 | all zeros |
| GPT-2 117M, B=1, S=128, **W=4** | 3.861e+00 | 1.000e+00 | all zeros |
| BERT base, B=1, S=128, W=4 | 5.258e+00 | 1.059e+04 | wrong |

A `max_rel` of exactly `1.000` is the all-zeros fingerprint: every element gives
`|0 - ref| / |ref| = 1`. Confirmed directly — 98,303 of 98,304 elements exactly
zero, against a reference with none.

The NUMA path is genuinely active: 29 `numa_args=` call sites at 2 layers, 181
at the full 12 layers. **Zero reconciliation lines.**

Two controls rule out alternatives:

- **Not caused by the all-reduce fix.** Swapping the pre-fix `wrapper.py` in
  produces an identical `max_abs=3.861e+00`. The defect predates `24f44419b9`.
- **Not caused by the rebuild.** The absence of reconciliation is visible by
  reading the generated Python, independent of which binary is loaded.

The repository's own correctness scripts pass **vacuously**.
`check_numa_preLN_block_correctness.py` reports `PASS` with
`max abs diff = 0.000e+00`, but emits **0** `numa_args` call sites — it never
fires the NUMA path, so it compares the standard path against itself. Exactly
zero difference was the tell: a real tensor-parallel reduction changes summation
order and would produce small floating-point differences.

## Evidence 2: archived benchmark artifacts

The sweeps saved generated Inductor code from actual benchmark runs under
`torch_compile_debug_runs/`, dated 4-5 May 2026. These are the artifacts
produced while the timing data was collected, not reproductions.

```
archived numa_tp output_code.py files : 26
  with NUMA path active (numa_args>0)  : 26
  with ANY reconciliation line         :  0
```

| Archived run | numa_args | reductions | copies |
| --- | --- | --- | --- |
| M345m_B1_S128 .. S2048 (5 runs) | 337 | 0 | 0 |
| M117m_B2_S128 .. S4096 (6 runs) | 181 | 0 | 0 |
| M117m_B4_S128 .. S2048 (5 runs) | 181 | 0 | 0 |
| M117m_B8_S128 .. S1024 (4 runs) | 181 | 0 | 0 |
| M117m_B1_S128 .. S4096 (6 runs) | 181 | 0 | 0 |

Every benchmarked configuration ran 181-337 NUMA-tagged kernel launches and
combined the per-socket results zero times.

## Impact on the measurements

The baseline arm is unaffected — ordinary Inductor, correct output. The NUMA arm
executed all its kernels, so timings measured real compute on correctly shaped
tensors. What they omitted is the reconciliation work.

A correct implementation must add, per row-parallel output:

- `W - 1` full-tensor `add_` operations
- `W` full-tensor clones to redistribute the reduced value

At 12 layers and `W=2` that is roughly 24 additional full-activation passes per
forward that were never measured. This is the cross-socket communication cost
that tensor parallelism trades compute against, so the reported speedups are
optimistic by an unquantified margin.

### A corroborating anomaly

`BENCH_REVIEW_REPORT.md` flagged a result it could not explain:
`gemma270m_sweep_results.csv` records **2.119x** at `world_size=2`, above the 2x
ceiling that 2-way tensor parallelism can deliver even with perfect scaling and
zero communication.

```
B,S,std_ms,numa_ms,speedup
1,8192,165161.42,77930.30,2.119
```

A path that performs the sharded compute but skips the reduction is a concrete
mechanism for exceeding that ceiling. The report attributed it to an
unpinned-baseline artifact; missing reduction work is a simpler explanation and
consistent with everything else here.

## Defect status

| # | Defect | Status |
| --- | --- | --- |
| 1 | Row-parallel reassembly used concatenation, not a sum | Fixed in `24f44419b9` — but masked, since reductions were never emitted |
| 2 | Graph output never reconciled -> returns unwritten memory | **Fixed**, uncommitted, 42 lines |
| 3 | `numa_role` lost between FX passes and wrapper | **Open — blocking** |

### Defect 2, as fixed

A deferred line class reconciles a replicated result that is also the graph's
return value, using a copy from node 0 rather than a sum — every socket holds a
complete copy of a replicated result, so summing would scale it by the socket
count.

```python
class _NumaGraphOutputCopyLine(DeferredLineBase):
    def __call__(self):
        return self.line if self.name in V.graph.get_output_names() else None
```

Effect: the returned tensor is now genuinely written (`zeros overall` went from
98,303/98,304 to 0/98,304), the catastrophic `2.6e+13` garbage values
disappeared, and pass combinations became distinguishable. Correctness still
fails, because defect 3 corrupts the values upstream.

Patch: `graph_output_copy_fix.patch` in this directory.

### Defect 3 is not a patch

`numa_role` has to be threaded from the FX passes through Inductor scheduling
into `codegen_kernel_call`, and the fallback heuristic needs to become reliable
enough to catch every row-parallel output. That is a design question about how
the passes tag their outputs.

One scoping question to settle first: with `_numa_mlp_tp` alone, **all 29
kernels** are NUMA-sharded, including attention and the layer norms. Whether
that is intended — whole-graph per-socket execution, reconciling only at
row-parallel boundaries — or a leak in the plan lookup changes what fixing
defect 3 means.

## Reproducing this

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd ~/llm-cpu-compiler/pytorch
```

The `LD_LIBRARY_PATH` line is needed only in non-interactive shells, which skip
the `.bashrc` that sources oneAPI `setvars.sh`. Without it Triton fails on
`libdnnl.so.3`.

Minimal failing case (exits 1):

```bash
python scripts/inductor/gpt2_multi_size_speedup_cli.py \
  --model 117m --batch-size 1 --seq-length 128 --world-size 2 \
  --warmup 2 --iters 5 --check-correctness
```

To confirm the NUMA path is active and unreconciled: the harness's `--profile`
counter is unreliable (it scans for `output_code.py`, which exists only under
compile-debug, and emits a false `TP did NOT fire` warning). Count in the
Inductor cache instead:

```bash
grep -rho "numa_args=" $CACHE_DIR | wc -l          # NUMA active
grep -rhE "\.add_\(_numa_|\.copy_\(_numa_" $F      # reconciliation (expect none)
```

Scripts written for this investigation, in this directory:

| Script | Purpose |
| --- | --- |
| `numa_pass_isolation.py` | One pass combination per invocation; reports `numa_args` sites so a combination that never fires is marked `NOFIRE` rather than scored as a pass |
| `run_pass_isolation.sh` | Driver — one subprocess per combination, so a segfault kills only that cell |
| `numa_diff_pattern.py` | Characterises the difference: zero vs non-finite, per-shard column distribution |
| `graph_output_copy_fix.patch` | Defect 2 fix |

Note: `scripts/inductor/gemma270m_bs_seq_sweep.sh` runs `git add`, `git commit`
and `git push` after every row. Use `gemma_multi_size_compile_debug_sweep.py`.

## What this does not establish

**The paper's evaluation platforms were not tested.** Table I lists Haswell
(Xeon E5-2620 v3) and Cascade Lake (Xeon Gold 6253CL), both dual-socket. Every
measurement here — live and archived — comes from `hpgen9-2`, a 4-socket Xeon
E7-4820 v3. No artifacts from the paper's own sweeps survive on this machine.

There is no known mechanism by which a different CPU would change the outcome:
the reconciliation decision is made in Python code generation, driven by FX
metadata that arrives as `None`, and nothing in that path reads the CPU model or
socket count. But that is reasoning, not measurement.

Closing the gap needs one of:

- A single `--check-correctness` run on one of the two evaluation platforms
- Archived `output_code.py` from those sweeps, checked the same way

Either is cheap. The first is about four minutes of machine time.

Also unverified: whether any non-output intermediate buffer hits the same gap as
defect 2 (the isolation runs prove only the output case), and whether the
`attention`-only path has a distinct problem, since it produced `2.643e+13`
before any changes were made.

## Open question

The correctness picture is separable from the idea. Nothing here says NUMA-aware
tensor parallelism does not work on multi-socket CPUs; it says this
implementation did not compute the right answer in the configuration that was
measured. Whether the approach delivers its claimed speedups is untested — the
experiment that would answer it has not yet been run correctly.
