"""Fix the >2 GiB segfault: GEMM templates must compute element offsets in
int64 when an operand's BYTE extent leaves int32 range.

Touches only torch/_inductor/kernel/mm_common.py (new helper) and
torch/_inductor/kernel/bmm.py (template + its two call sites). The flags
default to False, so every operand below 2 GiB generates a byte-identical
kernel body and stays comparable with results already measured.
"""
import io, sys

REPO = "/home/shavin/llm-cpu-compiler/pytorch"

HELPER = '''

def needs_int64_offsets(node) -> bool:
    """Whether a GEMM template must compute this operand's offsets in int64.

    Triton-CPU lowers ``tt.addptr`` on a tensor of pointers by multiplying the
    element offset by the element size *in the offset's own width*, and only
    sign-extends the product afterwards (triton-cpu
    ``third_party/cpu/lib/TritonToTritonCPU/ConvertPtrOps.cpp``,
    ``AddPtrOpConversion``). With int32 offsets that byte product wraps
    negative as soon as an operand exceeds 2 GiB, so the template's gather
    load reads an unmapped address -- a segfault, not a wrong number.

    Inductor's own ``can_use_32bit_indexing`` does not catch this because it
    bounds element *counts*: a 3 GiB float32 tensor is 805M elements, well
    inside int32. The bound that actually binds here is the byte offset.
    """
    int_max = torch.iinfo(torch.int32).max
    try:
        size = node.get_layout().storage_size()
    except (AttributeError, NotImplementedError):
        size = node.get_numel()
    try:
        size = int(V.graph.sizevars.size_hint(size))
    except Exception:
        # Extent not known at compile time: widen rather than risk the fault.
        return True
    return size * node.get_dtype().itemsize > int_max
'''

# ---------------------------------------------------------------- mm_common
p = f"{REPO}/torch/_inductor/kernel/mm_common.py"
s = io.open(p, encoding="utf-8").read()
if "def needs_int64_offsets" in s:
    print("mm_common: already patched")
else:
    anchor = "def mm_options(config, sym_m, sym_n, sym_k, layout, b_prologue_cast_type=None):"
    assert s.count(anchor) == 1, "mm_options anchor not unique"
    s = s.replace(anchor, HELPER.strip("\n") + "\n\n\n" + anchor)
    io.open(p, "w", encoding="utf-8").write(s)
    print("mm_common: added needs_int64_offsets")

# ---------------------------------------------------------------- bmm
p = f"{REPO}/torch/_inductor/kernel/bmm.py"
s = io.open(p, encoding="utf-8").read()

if "A_LARGE" in s:
    print("bmm: already patched")
    sys.exit(0)

# 1. import the helper
old = "from .mm_common import addmm_epilogue, mm_args, mm_configs, mm_options"
new = ("from .mm_common import (\n"
       "    addmm_epilogue,\n"
       "    mm_args,\n"
       "    mm_configs,\n"
       "    mm_options,\n"
       "    needs_int64_offsets,\n"
       ")")
assert s.count(old) == 1, "import anchor not unique"
s = s.replace(old, new)

# 2. template body: widen the initial pointer offsets when the flag is set.
#    Only the base offsets need widening -- the in-loop advance is
#    BLOCK_K * stride, which is bounded by the block size and cannot overflow.
old = """    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)
"""
new = """    idx_q = tl.program_id(1)  # batch dimension for BMM
    # A_LARGE / B_LARGE are set when the operand's byte extent leaves int32
    # range. Triton-CPU scales an element offset by the element size in the
    # offset's own width before sign-extending it, so int32 offsets wrap for
    # operands above 2 GiB and the gather below reads an unmapped address.
    if A_LARGE:
        A = A + (ram[:, None].to(tl.int64) * stride_am + rk[None, :].to(tl.int64) * stride_ak + idx_q.to(tl.int64) * stride_aq)
    else:
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    if B_LARGE:
        B = B + (rk[:, None].to(tl.int64) * stride_bk + rbn[None, :].to(tl.int64) * stride_bn + idx_q.to(tl.int64) * stride_bq)
    else:
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)
"""
assert s.count(old) == 1, "template anchor not unique"
s = s.replace(old, new)

# 3. tuned_bmm call site
old = """            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )"""
new = """            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
                A_LARGE=needs_int64_offsets(mat1),
                B_LARGE=needs_int64_offsets(mat2),
            )"""
assert s.count(old) == 1, "tuned_bmm call site not unique"
s = s.replace(old, new)

# 4. tuned_baddbmm call site
old = """            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(inp, mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
                prefix_args=1,
                epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
            )"""
new = """            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(inp, mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
                A_LARGE=needs_int64_offsets(mat1),
                B_LARGE=needs_int64_offsets(mat2),
                prefix_args=1,
                epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
            )"""
assert s.count(old) == 1, "tuned_baddbmm call site not unique"
s = s.replace(old, new)

io.open(p, "w", encoding="utf-8").write(s)
print("bmm: patched template + 2 call sites")
