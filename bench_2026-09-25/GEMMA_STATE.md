# Gemma NUMA-TP — diagnosis state (2026-10-09)

Status: **NOT FIXED**. Narrowed considerably; next steps below.

## Measured (gemma_check.py, 1152/4h/1kv/256hd/6912mlp, S=128, vs standard Inductor)

| config | 1 layer | 2 layers |
|---|---|---|
| GeGLU MLP TP alone, W=4 | corr 0.770 | corr 0.767 |
| attention TP alone, W=4 | corr 0.844 | corr 0.541 |
| both, W=4                | -         | corr 0.541 (BIT-IDENTICAL to attn-only) |
| both, W=2                | -         | corr 0.498 |

Two INDEPENDENT defects, each distinct from the six fixed on this branch.

### Defect G1 — GeGLU MLP, single-block
Error does NOT compound with depth (0.770 at 1 layer vs 0.767 at 2). So this is
a per-block bug, NOT the multi-block plan-merge class (defect 3).

RULED OUT by inspection of the generated code and the pass:
  - shard sizes correct: 6912/4 = 1728, slices [0:1728][1728:3456][3456:5184][5184:6912]
  - gate_w/up_w column-parallel dim 0 -> slice-then-clone (compact, strides match)
  - down_w row-parallel dim 1 -> clone-full-then-slice (parent strides, matches)
  - all-reduce IS emitted: 6 add_ lines for 2 layers x W=4 (= 2 x (W-1)) and
    8 _numa_ar_ reclones (= 2 x W)
  - reclones ARE consumed; buffer aliasing off them works (defect 4/6 fixes OK)
  - plan carries x_node AND all four generalized lists
  - reshape target-shape LITERALS are updated alongside meta (args[1] rewrite
    present in _apply_geglu_plan), so that gap is already handled

STILL UNCHECKED (next steps, cheapest first):
  1. Hypothesis discrimination on the GeGLU math, the technique that cracked the
     GPT-2 MLP defect: build the 1-layer model in eager, compute candidate wrong
     forms, correlate each against the NUMA output. Candidates: gate/up shard
     mispairing (gelu(gate_i) * up_j, i != j); only shard 0 contributing;
     down_w sliced on the wrong dim; gelu applied pre- rather than post-slice.
  2. The `len(old_shape_arg) == len(new_shape)` guard in the reshape rewrite --
     if the lengths ever differ the literal is silently left at full d while
     meta is resized.
  3. Whether `view` and `view_2` (the two separate reshapes of x feeding gate_mm
     and up_mm) both resolve to x_node; only one name is recorded.

### Defect G2 — Gemma attention, compounds per layer
0.844 at 1 layer -> 0.541 at 2. Per-layer accumulation. Not investigated yet.

### Defect G3 — shadowing when both passes run
"both" output is BIT-IDENTICAL to attention-only, so the GeGLU MLP contributes
nothing once attention TP is on. Same SHAPE as defect 5, but NOT the same cause:
both Gemma passes already write into the two config keys that
merged_global_shard_plan unions (_numa_attention_tp_shard_plan and
_numa_mlp_tp_shard_plan), so that fix does not reach this.

## Reproduce
    python bench_2026-09-25/gemma_check.py {std|mlp|attn|numa} {2|4}
    NLAYERS=1 ... for the single-layer variant
    python bench_2026-09-25/cmp.py /tmp/gc_std_w4_L1_s128.pt /tmp/gc_mlp_w4_L1_s128.pt

NOTE: scripts/inductor/simple_gemma_decoder_block_speedup_cli.py CANNOT be used
for correctness -- it builds a fresh UNSEEDED model per arm, so the two arms
never share weights. gemma_check.py exists because of that.
