# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.utils import empty_strided_cpu_numa
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import torch._inductor.kernel.mm_common

def get_raw_stream(_):
    return 0

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
import torch._inductor.kernel.bmm

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_kumara/er/cervotsmrugev5jc3iiztkvqntku3xd65hgh5y7b5zoizoiqr72g.py
# Topologically Sorted Source Nodes: [addmm], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   addmm => addmm
# Graph fragment:
#   %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg2_1, %arg1_1, %arg0_1), kwargs = {})
triton_tem_fused_addmm_0 = async_compile.triton('triton_tem_fused_addmm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=4,
    num_warps=2,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(3,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_0', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_addmm_0(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 6
    N = 32
    K = 12
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 12
    stride_ak = 1
    stride_bk = 32
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (32*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cpu')
meta0 = {'GROUP_M': 8, 'EVEN_K': False, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_kumara/zx/czxvmtncmyown2d2nogzuxog3jqdxjkzwskfzqvbhrer7eezvdbz.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   y => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute, 0.5946035575013605), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 6
    x2 = (xindex // 48)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*x2) + (32*x1)), xmask)
    tmp1 = 0.5946035575013605
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/ur/curp7okl5e3vp2hux6lei6fsy3rtdijkfyg2ymr6sb6trc6txefe.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   y => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_3, 0.5946035575013605), kwargs = {})
triton_poi_fused_mul_2 = async_compile.triton('triton_poi_fused_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[32, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 6
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.5946035575013605
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x1 + (6*y0)), tmp2, xmask & ymask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/ap/capd6dpx6llnzi4ee52jgnlv5pn4zux75deh7f7wyers63nn7dw6.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   y => bmm
# Graph fragment:
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_3, %view_4), kwargs = {})
triton_tem_fused_bmm_3 = async_compile.triton('triton_tem_fused_bmm_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=1,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_bmm_3', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_bmm_3(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 6
    N = 6
    K = 8

    stride_aq = 48
    stride_am = 8
    stride_ak = 1

    stride_bq = 48
    stride_bk = 6
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (6*idx_m) + (36*idx_q)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cpu')
meta1 = {'GROUP_M': 8, 'EVEN_K': False, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_kumara/wk/cwkvqoe2msqgc332tldtotp4j7ox7azmijukqdawweaq6zfaq7ub.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   y => amax, any_1, eq, logical_not
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_5, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_5, [-1], True), kwargs = {})
triton_poi_fused__safe_softmax_4 = async_compile.triton('triton_poi_fused__safe_softmax_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (2 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (3 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (4 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (5 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp1 = float("-inf")
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2 == 0
    tmp4 = tmp3.to(tl.int64)
    tmp5 = (tmp4 != 0)
    tmp7 = tmp6 == tmp1
    tmp8 = tmp7 == 0
    tmp9 = tmp8.to(tl.int64)
    tmp10 = (tmp9 != 0)
    tmp11 = tmp5 | tmp10
    tmp13 = tmp12 == tmp1
    tmp14 = tmp13 == 0
    tmp15 = tmp14.to(tl.int64)
    tmp16 = (tmp15 != 0)
    tmp17 = tmp11 | tmp16
    tmp19 = tmp18 == tmp1
    tmp20 = tmp19 == 0
    tmp21 = tmp20.to(tl.int64)
    tmp22 = (tmp21 != 0)
    tmp23 = tmp17 | tmp22
    tmp25 = tmp24 == tmp1
    tmp26 = tmp25 == 0
    tmp27 = tmp26.to(tl.int64)
    tmp28 = (tmp27 != 0)
    tmp29 = tmp23 | tmp28
    tmp31 = tmp30 == tmp1
    tmp32 = tmp31 == 0
    tmp33 = tmp32.to(tl.int64)
    tmp34 = (tmp33 != 0)
    tmp35 = tmp29 | tmp34
    tmp36 = triton_helpers.maximum(tmp0, tmp6)
    tmp37 = triton_helpers.maximum(tmp36, tmp12)
    tmp38 = triton_helpers.maximum(tmp37, tmp18)
    tmp39 = triton_helpers.maximum(tmp38, tmp24)
    tmp40 = triton_helpers.maximum(tmp39, tmp30)
    tl.store(out_ptr0 + (x0), tmp35, xmask)
    tl.store(out_ptr1 + (x0), tmp40, xmask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/yo/cyonufinb5ekyg5gqrnivijxcgp5tgcj72tj4j74k7jmjlbxdqs2.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   y => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_5, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__safe_softmax_5 = async_compile.triton('triton_poi_fused__safe_softmax_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 6)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/wk/cwkvu2xrpclp4ugatjubriguigkkqkvlrcfmee4bl5ya5pcz4dgt.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   y => sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
triton_poi_fused__safe_softmax_6 = async_compile.triton('triton_poi_fused__safe_softmax_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + (6*x0)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/vf/cvfck55mgky7rklwrunkdlespdenstz23tbtclw4xwoo6yclsfn5.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   y => div, full_default, logical_not_1, sum_1, where
# Graph fragment:
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 4, 6, 6], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default, %div), kwargs = {})
triton_poi_fused__safe_softmax_7 = async_compile.triton('triton_poi_fused__safe_softmax_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 6)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 == 0
    tmp4 = tmp2 / tmp3
    tmp5 = 0.0
    tmp6 = tl.where(tmp1, tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/sp/cspng54zqgruphtipo4y5luxpv32l4kdt7wxtkd7pttmwuykx2ji.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   y => bmm_1
# Graph fragment:
#   %bmm_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_6, %view_7), kwargs = {})
triton_tem_fused_bmm_8 = async_compile.triton('triton_tem_fused_bmm_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=4,
    num_warps=1,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_bmm_8', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_bmm_8(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 6
    N = 8
    K = 6

    stride_aq = 36
    stride_am = 6
    stride_ak = 1

    stride_bq = 8
    stride_bk = 32
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (8*idx_m) + (48*idx_q)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/n6/cn6xnpdeejfkwrzz4h5y3dwimxwosfooatfpzrrvqgegya56gr34.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 4
    x2 = (xindex // 32)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*x2) + (48*x1)), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/p6/cp6jtw5bha6maunixb3sjwcp54ntihwrkw6yycotle54k5tr67ji.py
# Topologically Sorted Source Nodes: [addmm_3], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   addmm_3 => addmm_3
# Graph fragment:
#   %addmm_3 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg8_1, %view_9, %arg7_1), kwargs = {})
triton_tem_fused_addmm_10 = async_compile.triton('triton_tem_fused_addmm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=2,
    num_warps=1,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_10', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_addmm_10(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 16
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 6
    N = 12
    K = 32
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 32
    stride_ak = 1
    stride_bk = 12
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (12*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cpu')
meta2 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    assert_size_stride(arg0_1, (12, 32), (32, 1))
    assert_size_stride(arg1_1, (6, 12), (12, 1))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (12, 32), (32, 1))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (12, 32), (32, 1))
    assert_size_stride(arg6_1, (32, ), (1, ))
    assert_size_stride(arg7_1, (32, 12), (12, 1))
    assert_size_stride(arg8_1, (12, ), (1, ))
    buf0 = empty_strided_cpu((6, 32), (32, 1), torch.float32)
    # Topologically Sorted Source Nodes: [addmm], Original ATen: [aten.addmm]
    streamNone = get_raw_stream(None)
    triton_tem_fused_addmm_0.run(arg2_1, arg1_1, arg0_1, buf0, grid=torch._inductor.kernel.mm_common.mm_grid(6, 32, meta0), stream=streamNone)
    del arg0_1
    del arg2_1
    buf1 = empty_strided_cpu((6, 32), (32, 1), torch.float32)
    # Topologically Sorted Source Nodes: [addmm_1], Original ATen: [aten.addmm]
    triton_tem_fused_addmm_0.run(arg4_1, arg1_1, arg3_1, buf1, grid=torch._inductor.kernel.mm_common.mm_grid(6, 32, meta0), stream=streamNone)
    del arg3_1
    del arg4_1
    buf2 = empty_strided_cpu((1, 4, 6, 8), (192, 48, 8, 1), torch.float32)
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul]
    triton_poi_fused_mul_1.run(buf0, buf2, 192, grid=grid(192), stream=streamNone)
    buf3 = reinterpret_tensor(buf0, (1, 4, 8, 6), (192, 48, 6, 1), 0); del buf0  # reuse
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul]
    triton_poi_fused_mul_2.run(buf1, buf3, 32, 6, grid=grid(32, 6), stream=streamNone)
    del buf1
    buf4 = empty_strided_cpu((4, 6, 6), (36, 6, 1), torch.float32)
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten.bmm]
    triton_tem_fused_bmm_3.run(buf2, buf3, buf4, grid=torch._inductor.kernel.bmm.bmm_grid(4, 6, 6, meta1), stream=streamNone)
    buf5 = empty_strided_cpu((1, 4, 6, 1), (24, 6, 1, 24), torch.bool)
    buf6 = empty_strided_cpu((1, 4, 6, 1), (24, 6, 1, 24), torch.float32)
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
    triton_poi_fused__safe_softmax_4.run(buf4, buf5, buf6, 24, grid=grid(24), stream=streamNone)
    buf7 = reinterpret_tensor(buf4, (1, 4, 6, 6), (144, 36, 6, 1), 0); del buf4  # reuse
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
    triton_poi_fused__safe_softmax_5.run(buf7, buf6, 144, grid=grid(144), stream=streamNone)
    buf8 = buf6; del buf6  # reuse
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
    triton_poi_fused__safe_softmax_6.run(buf7, buf8, 24, grid=grid(24), stream=streamNone)
    buf9 = reinterpret_tensor(buf3, (6, 32), (32, 1), 0); del buf3  # reuse
    # Topologically Sorted Source Nodes: [addmm_2], Original ATen: [aten.addmm]
    triton_tem_fused_addmm_0.run(arg6_1, arg1_1, arg5_1, buf9, grid=torch._inductor.kernel.mm_common.mm_grid(6, 32, meta0), stream=streamNone)
    del arg1_1
    del arg5_1
    del arg6_1
    buf10 = buf7; del buf7  # reuse
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
    triton_poi_fused__safe_softmax_7.run(buf10, buf5, buf8, 144, grid=grid(144), stream=streamNone)
    del buf5
    del buf8
    buf11 = reinterpret_tensor(buf2, (4, 6, 8), (48, 8, 1), 0); del buf2  # reuse
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten.bmm]
    triton_tem_fused_bmm_8.run(buf10, buf9, buf11, grid=torch._inductor.kernel.bmm.bmm_grid(4, 6, 8, meta1), stream=streamNone)
    del buf10
    buf12 = reinterpret_tensor(buf9, (1, 6, 4, 8), (192, 32, 8, 1), 0); del buf9  # reuse
    # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
    triton_poi_fused_clone_9.run(buf11, buf12, 192, grid=grid(192), stream=streamNone)
    del buf11
    buf13 = empty_strided_cpu((6, 12), (12, 1), torch.float32)
    # Topologically Sorted Source Nodes: [addmm_3], Original ATen: [aten.addmm]
    triton_tem_fused_addmm_10.run(arg8_1, buf12, arg7_1, buf13, grid=torch._inductor.kernel.mm_common.mm_grid(6, 12, meta2), stream=streamNone)
    del arg7_1
    del arg8_1
    return (buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((12, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((6, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((12, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((12, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((32, 12), (12, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
