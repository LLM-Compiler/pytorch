# Defect 9 — GEMM templates compute element offsets in int32, so operands above 2 GiB segfault

Status: **fixed** in `torch/_inductor/kernel/bmm.py` + `torch/_inductor/kernel/mm_common.py`.
Unblocks 12 of the 24 grid cells that previously carried no number.

## Symptom

`Segmentation fault` in the standard (non-NUMA) arm, deterministic, for every
configuration with `B·H·S² · 4 bytes > 2³¹`. The NUMA arm was unaffected
because it shards the head axis, so each socket's attention intermediate is
4× smaller and stays under the limit. That asymmetry is why the failure looked
like a NUMA-only quirk for so long: it is the exact opposite.

## Reproduction

One transformer layer is enough — the overflow depends on a single attention
intermediate, not on depth:

```
python bench_2026-09-25/ovf_repro.py --model 117m --batch-size 4 \
    --seq-length 4096 --num-layers 1
```

Before the fix, with `OMP_NUM_THREADS=1` and `faulthandler` armed, the fault
lands in `triton_tem_fused_bmm_7` — the `softmax(QKᵀ) @ V` BMM, whose A operand
is `(4, 12, 4096, 4096)` float32 = 805,306,368 elements = **3.0 GiB**:

```
Current thread (most recent call first):
  File ".../triton/backends/cpu/driver.py", line 872 in __call__
  File ".../torch/_inductor/runtime/triton_heuristics.py", line 945 in run
  File "/tmp/.../cz3gpeeq...py", line 1181 in call
```

Note which kernels *survive* at the same tensor size: the QKᵀ BMM that **writes**
that buffer, and the softmax reduction that reads it. Both are lowered through
the contiguous `getelementptr` form. Only the **gather load of a tensor of
pointers inside the BMM template's K-loop** takes the overflowing path.

## Root cause

Two layers are each individually defensible, and together they are wrong.

**1. Triton-CPU scales the byte offset in the offset's own width.**
`triton-cpu/third_party/cpu/lib/TritonToTritonCPU/ConvertPtrOps.cpp`,
`AddPtrOpConversion::matchAndRewrite`:

```cpp
Value scale = /* splat of sizeof(elem), in offsetTy */;
offset = rewriter.create<arith::MulIOp>(loc, offset, scale);   // <-- in i32
if (offsetTy.getElementTypeBitWidth() < ptrTy.getElementTypeBitWidth())
  offset = rewriter.create<arith::ExtSIOp>(loc, ptr.getType(), offset);
rewriter.replaceOpWithNewOp<arith::AddIOp>(op, ptr.getType(), ptr, offset);
```

The multiply by the element size happens **before** the sign-extend. For the
worst-case element index 805,306,367 the product `805306367 << 2` evaluated in
i32 is `0xBFFF_FFFC` = −1,073,741,828; sign-extending *that* gives a huge
negative i64, and `base + offset` lands on an unmapped page. This is the only
scale-multiply site in the whole CPU backend, confirmed by grep.

**2. Inductor's index-width choice bounds element counts, not bytes.**
`SIMDScheduling.can_use_32bit_indexing` compares `storage_size()` against
`INT32_MAX`. A 3 GiB float32 tensor is only 805M elements, so int32 is selected
and is *correct at Triton level* — element indices really do fit. The bound
that binds on this backend is the byte offset, which is 4× larger. And the GEMM
templates do not consult `index_dtype` at all: they write their own index
arithmetic, and `TritonTemplate.generate` simply refuses anything it considers
out of int32 range with `"64-bit indexing is not yet implemented for triton
templates"`.

## The fix applied here

`needs_int64_offsets(node)` in `mm_common.py` asks the question that actually
matters — does this operand's **byte** extent leave int32 range — and the bmm
template widens its base offsets to int64 when it does:

```python
if A_LARGE:
    A = A + (ram[:, None].to(tl.int64) * stride_am + ... )
else:
    A = A + (ram[:, None] * stride_am + ... )
```

Three properties worth stating:

- **It is conditional.** `A_LARGE`/`B_LARGE` are false for every operand below
  2 GiB, so the generated kernel body is byte-identical to before for all 120
  cells already measured. Verified in the generated source: for the repro, the
  `softmax@V` BMM gets `A_LARGE=True, B_LARGE=False` and every other BMM gets
  both `False`.
- **Only the base offsets are widened.** The in-loop advance is
  `BLOCK_K * stride_ak`, bounded by the block size, and cannot overflow.
- **Numerics are unchanged**, checked by forcing the int64 path on an operand
  small enough to run both ways (`INDUCTOR_FORCE_I64_OFFSETS=1`, see
  `bench_2026-09-25/i64_equiv.py`). gpt2-117m B=1 S=2048, standard arm, both
  widths, same seed:

  ```
  shape=(1, 2048, 768) max_abs=0 mean_abs=0
  bit_identical=True both_finite=True
  ```

  Not merely close — the same bits.

## A second defect this turned up, in the benchmark harness

`timeout -k` escalates to SIGKILL, which `/usr/bin/time` reports as
`terminated by signal 9` — indistinguishable from an OOM kill by signal number
alone. The earlier probe tested the signal before the return code, so a cell
that ran out of *time* was recorded as having run out of *memory*. Two cells
carry that label wrongly: bert-large B=8 S=4096 and gpt2-762m B=8 S=4096, both
"OOM" at a peak of 15-16 GB on a 62 GB machine, which cannot be memory
exhaustion.

This matters more than a mislabel: a cell abandoned as unfixable-memory gets no
further attempt, whereas a cell that hit a time cap just needs a longer cap.
`final_sweep.sh` now decides from elapsed wall clock against the cap, and
records the elapsed seconds for both arms so the attribution is auditable
rather than asserted.

## The upstream fix, not applied here

The real repair belongs in Triton-CPU: sign-extend before scaling, so the byte
offset is 64-bit throughout.

```cpp
VectorType offsetTy = cast<VectorType>(offset.getType());
VectorType ptrTy = cast<VectorType>(ptr.getType());
// Widen to pointer width BEFORE scaling by the element size, otherwise the
// byte product wraps for operands above 2 GiB.
if (offsetTy.getElementTypeBitWidth() < ptrTy.getElementTypeBitWidth()) {
  offset = rewriter.create<arith::ExtSIOp>(loc, ptrTy, offset);
  offsetTy = ptrTy;
}
Value scale = rewriter.create<arith::ConstantOp>(
    loc, offsetTy,
    SplatElementsAttr::get(offsetTy,
        rewriter.getIntegerAttr(offsetTy.getElementType(),
                                (elemBitWidth + 7) / 8)));
offset = rewriter.create<arith::MulIOp>(loc, offset, scale);
rewriter.replaceOpWithNewOp<arith::AddIOp>(op, ptr.getType(), ptr, offset);
```

It was deliberately **not** applied during this study. It changes address
arithmetic for every kernel on the backend, which would mean relinking
`libtriton.so` (647 MB, and the configured build tree has never produced an
artifact) and re-running all 120 measured cells to keep the dataset
homogeneous. The template fix achieves the same unblocking with no effect on
already-measured configurations.

## Same construct, not touched

`ram[:, None] * stride_am + ...` appears in five other templates: `mm.py`,
`mm_plus_mm.py`, `mm_scaled.py`, `unpack_mixed_mm.py`. None can be reached with
a >2 GiB operand in this sweep — the largest 2-D operand is
`B·S × d_ffn = 32768 × 5120` = 671 MB, computed over the whole grid, not
estimated. They carry the same latent bug for larger models.
