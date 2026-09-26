# Stage 2 -- speedup spot-check

Host hpgen9-2 (4 sockets, 4 NUMA nodes, 80 logical CPUs, ~16 GB/node).
Branch fix/numa-tp-row-parallel-allreduce @ 9cfcfaed7b + 5 uncommitted
correctness fixes (bench_2026-09-25/numa_tp_correctness_fixes.patch).
Protocol: --warmup 3 --iters 10. W=2 cells run under
`numactl --cpunodebind=0,1 --membind=0,1` so both arms get the same 40 CPUs.
Baseline is the harness baseline (max_autotune_gemm_backends="TRITON"),
not stock torch.compile. Run 2026-09-26 18:01-19:10 IST, 69 min.

speedup = std_mean_ms / numa_mean_ms.  ">1.0 = NUMA-TP wins"

## World size 4 (all four sockets)

| Model        | B | S    | speedup | std ms  | numa ms | correctness | numa_args |
|--------------|---|------|---------|---------|---------|-------------|-----------|
| GPT-2 117M   | 1 | 1024 | 0.70x   |  1161.5 |  1661.7 | PASS        | 338       |
| GPT-2 117M   | 1 | 2048 | 1.31x   |  4282.4 |  3273.7 | PASS        | 338       |
| GPT-2 345M   | 1 | 2048 | 1.25x   | 12676.5 | 10153.5 | PASS        | 674       |
| BERT large   | 1 | 2048 | 1.23x   | 11792.0 |  9549.4 | PASS        | 674       |
| Gemma 270M   | 1 | 2048 | (0.76x) | 10023.1 | 13150.6 | **FAIL**    | fires     |

## World size 2 (nodes 0-1, both arms confined by numactl)

| Model        | B | S    | speedup | std ms  | numa ms | correctness | numa_args |
|--------------|---|------|---------|---------|---------|-------------|-----------|
| GPT-2 117M   | 1 | 1024 | 0.69x   |  1332.9 |  1924.4 | PASS        | 338       |
| GPT-2 117M   | 1 | 2048 | 0.97x   |  4345.3 |  4478.1 | PASS        | 338       |
| GPT-2 345M   | 1 | 2048 | 0.95x   | 13326.0 | 14075.2 | PASS        | 674       |
| BERT large   | 1 | 2048 | 0.99x   | 13368.5 | 13480.1 | PASS        | 674       |
| Gemma 270M   | 1 | 2048 | (0.82x) | 12592.1 | 15355.1 | **FAIL**    | fires     |

Standard arm emitted 0 numa_args in every cell, as required.

## Gate assessment (plan's Stage 2 criteria)

PRIMARY -- "majority of configs show speedup > 1.0 at W=4":
  MET for the three valid transformer cells at S=2048 (1.31x, 1.25x, 1.23x).
  NOT met at S=1024 (0.70x).
SANITY CEILING -- "speedup must not exceed world size": all well under 4. OK.
DIRECT COMPARISON -- configs 2 and 3 had recorded pre-fix values on this exact
  branch at W=2 of 0.93x and 0.95x. They now read 0.97x and 0.95x: W=2 is
  essentially UNCHANGED. The gain is confined to W=4.

## Reading

1. Sequence length is the whole story at W=4. 117M goes 0.70x -> 1.31x from
   S=1024 to S=2048; the paper reports the same monotone trend. Below some
   crossover the fixed per-reduction cost (W-1 full-tensor add_ passes, W
   re-clones, W accumulator clones) exceeds the locality gain.

2. W=2 gains nothing on this box. This is consistent with the launcher-thread
   asymmetry found earlier: the launcher ignores the requested thread count
   and spawns one worker per CPU in the selected nodes' cpulist, so a W=2 NUMA
   arm uses 40 threads. Confining the baseline to the same 40 CPUs (as done
   here) removes the unfair comparison -- and with it, the speedup. Every
   archived manifest on this machine ran W=2 unconfined, which is why the
   original "no speedup on 4 sockets" result looked worse than it was AND why
   correcting the comparison does not rescue it.

3. The paper's dual-socket numbers are not reproduced at W=2 here. What
   reproduces is the shape of the effect at W=4, at roughly half the paper's
   magnitude (1.23-1.31x vs 1.87-2.63x).

## Gemma: results VOID

The two Gemma cells are measured on numerically wrong output and must not be
quoted. The Gemma CLI cannot detect this itself: it builds a FRESH randomly
initialised model per arm with no seed, so the two arms never share weights
and no output comparison is possible.

Verified separately (bench_2026-09-25/gemma_check.py, 2 layers, S=128, shared
weights, vs standard Inductor):

  gemma attention TP only, W=4   corr=0.541  max_abs=9.2e+00
  gemma GeGLU MLP TP only, W=4   corr=0.767  max_abs=7.1e+00
  both, W=4                      corr=0.541  max_abs=9.2e+00
  both, W=2                      corr=0.498  max_abs=8.9e+00

Two independent defects, both distinct from the five already fixed:
  (a) the Gemma attention path (numa_gemma_attention_tp) is wrong on its own;
  (b) the GeGLU MLP path (_apply_geglu_plan) is wrong on its own.
And a third signature: "both" is BIT-IDENTICAL to attention-only, so with
both passes on the GeGLU MLP contributes nothing -- the same shape as the
defect-5 shadowing, though not the same cause (both Gemma passes already
write into the two config keys that merged_global_shard_plan unions).

GPT-2 and BERT are unaffected: they pass at every world size and layer count.
