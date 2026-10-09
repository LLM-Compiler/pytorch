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

## ROOT CAUSE FOUND (2026-10-09) — rotary embedding, not the TP passes

Disabling `apply_rotary_emb` in BOTH arms makes every Gemma path correct:

| config (1 layer, W=4, S=128) | rotary ON | rotary OFF |
|---|---|---|
| GeGLU MLP TP vs standard | corr 0.770 | **corr 1.000000, max_abs 1.55e-05** |
| attention TP vs standard | corr 0.844 | **corr 1.000001, max_abs 9.78e-06** |

So there is ONE defect, not the two (G1/G2) hypothesised earlier, and it is not
in the GeGLU or attention sharding at all.

### Evidence chain
1. No GeGLU arithmetic hypothesis matched the NUMA output (8 candidates tried,
   incl. shard mispairing, wrong slice dim, gelu-on-fuse, W-overcount). H0 (the
   correct sharded form) reproduced STANDARD exactly, so the harness was sound.
2. Zeroing `down_proj` so the MLP contributes NOTHING still left NUMA differing
   from standard (corr 0.822). The error was therefore outside the MLP.
3. Disabling rotary fixes both paths. Rotary is the shared root.

### Why this fits every symptom
- MLP-only wrong even with MLP output zeroed -> error never was in the MLP
- attention error COMPOUNDS with depth (0.844 @1L -> 0.541 @2L): rotary runs per layer
- GPT-2 / BERT unaffected: neither has rotary
- "both" bit-identical to attention-only: both perturbed by the same upstream op

### Mechanism
`apply_rotary_emb` uses complex ops:
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x..., 2, -1), -1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
Inductor logs "does not support code generation for complex operators" and falls
back for these. The wrapper's per-socket `numa_args` path does not give the
fallback the per-socket treatment the surrounding Triton kernels get, so the
sockets diverge from the master.

### Next step
Inspect the generated code around the rotary region: confirm whether the
complex-op fallback is (a) run only on the master with per-socket buffers left
stale, or (b) run per socket but reading/writing the wrong buffers. Fix is in
wrapper codegen, NOT in numa_mlp_tp.py / numa_gemma_attention_tp.py.

### Reproduce
    NOROPE=1 NLAYERS=1 python bench_2026-09-25/gemma_check.py {std|mlp|attn} 4
    python bench_2026-09-25/cmp.py /tmp/gc_std_w4_L1_s128_norope.pt /tmp/gc_mlp_w4_L1_s128_norope.pt

## REFINED (2026-10-09, same session) — it is the rotary CHAIN, not the complex ops

Two further experiments sharpen the root cause:

### 1. Real-arithmetic rotary does NOT fix it
Reformulated `apply_rotary_emb` without complex ops, using
(a+bi)(c+di) = (ac-bd) + (ad+bc)i. Verified mathematically exact:
    standard(real-rope) vs standard(complex-rope): max_abs 8.58e-06, corr 1.000000

But NUMA stays broken, and one case gets worse:
| config (1 layer, W=4) | complex rope | real rope |
|---|---|---|
| attention TP | corr 0.844 | corr 0.770 |
| GeGLU MLP TP | corr 0.770 | **NaN** |

So the defect is NOT the complex-operator fallback. Removing rotary entirely
(NOROPE) gives corr 1.000000; expressing the SAME rotation in real arithmetic
does not. The problem is the rotary region's buffer structure, not its op type.

### 2. The rotary chain is never NUMA-tracked
In the generated code the rotary input is a PLAIN allocation with no per-socket
versions at all:
    buf6 = empty_strided_cpu((1,1,128,128,2), (32768,32768,256,2,1), float32)
    buf4 = reinterpret_tensor(buf6, ...)  # alias
    buf5 = reinterpret_tensor(buf6, ...)  # alias
    grep -c "_numa_p._buf6_n" -> 0
The chunk/stack/alias pattern feeding rotary sits outside the NUMA machinery,
so per-socket execution reads and writes inconsistent buffers -> divergence,
and NaN once the shapes interact badly.

### 3. The pass walks PAST rotary instead of sharding it
`numa_gemma_attention_tp.py` references rotary in exactly one place -- a BFS
comment about traversing "reshape/permute/clone/expand/mul/cat/unsqueeze/
split/getitem/view_as_complex/view_as_real/index_put" to locate the Q/K mm
upstream. It never tags, resizes, or shards the rotary chain nodes.

Compare `numa_mlp_tp.py`, which DOES handle its activation chain: it collects
`gelu_chain` and runs `_resize_node_val_and_tensor_meta` over every node plus
rewrites reshape target-shape literals. The Gemma attention pass has no
equivalent for the rotary chain.

### THE FIX
Extend `numa_gemma_attention_tp_pass` to treat the rotary chain the way
`numa_mlp_tp_pass` treats `gelu_chain`: collect the chain nodes between the Q/K
mm and the attention matmul, resize their meta to the sharded head count,
rewrite any reshape literals, and tag them with the shard plan so wrapper
codegen emits per-socket buffers for them.

### Also in the tree (unvalidated)
`PythonWrapperCodegen._numa_extern_per_socket` replicates extern/fallback aten
calls across sockets. Correct by construction (it only fires when an operand
already has per-socket versions) but NOT exercised by this bug, because the
rotary operands are not NUMA-tracked in the first place. Keep it: it closes a
real gap for extern ops whose inputs ARE tracked. Remove it if it ever gets in
the way.

## EXACT MECHANISM + IMPLEMENTATION PLAN (2026-10-09, end of session)

### The precise bug, from the generated code

    buf6 = empty_strided_cpu((1,1,128,128,2), (32768,32768,256,2,1), float32)
    buf4 = reinterpret_tensor(buf6, (1,1,128,128,1), (...), 0)   # alias
    buf5 = reinterpret_tensor(buf6, (1,1,128,128,1), (...), 1)   # alias

    _numa_p1000_buf4_n0 = empty_strided_cpu_numa(buf4.size(), ...)   # FRESH
    _numa_p1000_buf5_n0 = empty_strided_cpu_numa(buf5.size(), ...)   # FRESH
    triton_poi_fused_stack_3.run(buf2, buf3, arg5_1, buf4, buf5, numa_args=[...])

    buf7 = torch.ops.aten.view_as_complex.default(buf6)   # MASTER ONLY

`stack` writes the real/imag pair as TWO INTERLEAVED ALIASES into one parent
(offsets 0 and 1, stride 2). Per socket the wrapper allocates those aliases as
two INDEPENDENT buffers, so (a) the interleaving is destroyed and (b) `buf6`
itself never gets a socket-local copy. Everything downstream of `buf6` --
view_as_complex, the complex mul, view_as_real -- therefore runs on master data
only, and sockets 1..W-1 keep un-rotated q/k.

NOTE the FX side is FINE: shapes are already sharded correctly in the generated
code (1,1,128,128,2) -- H_total 4 -> H_shard 1. The value-based resize in
`_apply_plan` works. This is purely a WRAPPER CODEGEN gap, not an FX-pass gap.
(Supersedes the earlier "give the pass a rotary-chain handler" suggestion.)

### Attempted and REVERTED
Made `_emit_numa_empty_strided` alias-aware (carve the per-socket buffer out of
a per-socket parent instead of allocating standalone), plus
`_numa_extern_per_socket` to replicate extern aten calls. Result: attention went
from corr 0.844 to **NaN**. Reverted; verified back to 0.844.

Partial replication is WORSE than none. The rotary region spans FOUR codegen
paths and all must be handled together or sockets read a mix of master and
socket-local data:
  1. aliased allocation      buf4/buf5 = reinterpret_tensor(buf6, ...)  # alias
                             -> codegen_deferred_allocation
  2. extern/fallback calls   buf7 = torch.ops.aten.view_as_complex(buf6)
                             -> generate_extern_kernel_alloc
  3. plain alias assignment  buf8 = buf7, buf10 = buf9, buf12 = buf11
                             -> emitted directly, no NUMA hook at all
  4. reinterpret reuse       buf13 = reinterpret_tensor(buf6, ...)  # reuse
                             -> make_buffer_reuse (ALREADY handled)

### Implementation plan (do all four, then test; do not land partially)
1. `codegen_deferred_allocation`: record name -> (parent, size, stride, offset)
   and make `_emit_numa_empty_strided` carve from a per-socket parent,
   allocating the parent per socket on first use and registering it in
   `_numa_tensor_cache`.
2. `generate_extern_kernel_alloc`: replicate the call per socket, substituting
   each operand's socket-local buffer; register the output in the cache.
   (`_numa_extern_per_socket`, already committed, does this -- it currently
   declines only because its operands are untracked, which step 1 fixes.)
3. Find where `bufN = bufM` pass-through assignments are emitted and propagate
   the cache entry across them. This is the step that was MISSING and is the
   likely cause of the NaN: the chain broke at `buf8 = buf7`.
4. Verify `make_buffer_reuse` still resolves `buf13 = reinterpret_tensor(buf6)`
   now that buf6 is cached.

### Test loop (fast)
    rm -rf /tmp/gc_cache_*
    NLAYERS=1 python bench_2026-09-25/gemma_check.py std  4
    NLAYERS=1 python bench_2026-09-25/gemma_check.py attn 4
    python bench_2026-09-25/cmp.py /tmp/gc_std_w4_L1_s128.pt /tmp/gc_attn_w4_L1_s128.pt
~4 min per compile. Targets, in order:
    attn  corr 0.844 -> 1.000000
    mlp   corr 0.770 -> 1.000000
    numa (both)      -> 1.000000
Then re-run the GPT-2/BERT matrix (bench_2026-09-25/matrix.sh) to prove no
regression, since all four hooks are shared code.

### Ground truth to check against
With rotary disabled (NOROPE=1) every Gemma path is already correct
(corr 1.000000 / 1.000001). That is the target the fix must reach with rotary on.

## FIXED (2026-10-09) — all four alias-chain hooks, together

Gemma is numerically correct with rotary ENABLED, under `NUMA_ALIAS_CHAIN=1`:

| config (S=128, vs standard Inductor) | before | after |
|---|---|---|
| L1 GeGLU MLP TP, W=4 | corr 0.770 | **1.000000** (max_abs 1.53e-05) |
| L1 attention TP, W=4 | corr 0.844 | **1.000000** (max_abs 9.12e-06) |
| L1 both, W=4         | corr 0.844 | **1.000000** (max_abs 1.57e-05) |
| L2 GeGLU MLP TP, W=4 | corr 0.767 | **1.000001** (max_abs 1.76e-05) |
| L2 attention TP, W=4 | corr 0.541 | **1.000001** (max_abs 1.50e-05) |
| L2 both, W=4         | corr 0.541 | **1.000001** (max_abs 1.79e-05) |
| L2 both, W=2         | corr 0.498 | **1.000001** (max_abs 1.86e-05) |

The residuals are float reduction-order noise, the same order as GPT-2/BERT.

### What the fix is
Four wrapper-codegen hooks that must ALL be active together. Enabling a subset
leaves sockets reading a mix of master and socket-local data, which is strictly
worse than none (measured: attention 0.844 -> NaN with only two of them).

1. `codegen_deferred_allocation` records `name -> (parent, size, stride, offset)`
   for buffers that are VIEWS into a larger parent.
2. `_emit_numa_empty_strided` carves such a view out of a PER-SOCKET parent
   (allocating the parent per socket on first use and caching it) instead of
   allocating it standalone. This is what preserves the interleaved real/imag
   layout that `stack` builds for `view_as_complex`.
3. `codegen_multi_output` carries the per-socket cache entry across bare
   `bufN = bufM` renames. THIS was the step missing in the failed attempt; the
   replication chain broke at `buf8 = buf7`.
4. `_numa_extern_per_socket` replicates extern/fallback aten calls
   (`view_as_complex`, complex `mul`, `view_as_real`) per socket. It only fires
   once (1) and (2) have made its operands socket-local.

`make_buffer_reuse` already handled the fourth path (`reinterpret_tensor` reuse).

### Why only Gemma needed it
GPT-2 and BERT never build interleaved aliases and never hit an op Inductor
cannot codegen, so their buffers were always tracked. Rotary is the only place
in these benchmarks that combines `stack`-built aliasing with extern fallbacks.
