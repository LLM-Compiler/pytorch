# AOT ID: ['0_forward']
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


# kernel path: /tmp/torchinductor_kumara/vg/cvgbfakjslluawjkzafvncp5ix7nq2kn2astcqt5ls6wji7elov7.py
# Topologically Sorted Source Nodes: [addmm], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   addmm => addmm
# Graph fragment:
#   %addmm : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_1, %primals_3, %primals_2), kwargs = {})
triton_tem_fused_addmm_0 = async_compile.triton('triton_tem_fused_addmm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=1,
    num_warps=2,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(3,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_0', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_addmm_0(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 512
    N = 1024
    K = 1152
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1152
    stride_ak = 1
    stride_bk = 1024
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
    xindex = idx_n + (1024*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cpu')
meta0 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_kumara/eb/cebslkrsph75rirsfwesgfx5qmp7svohcyp7ymgtwus2q5wedy25.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   y => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute, 0.25), kwargs = {})
#   %permute_14 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1]), kwargs = {})
triton_poi_fused_mul_transpose_1 = async_compile.triton('triton_poi_fused_mul_transpose_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_transpose_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_transpose_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 512
    x2 = (xindex // 131072)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x2) + (1024*x1)), None)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x0 + (256*x2) + (1024*x1)), tmp2, None)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/t4/ct4zha33r3va7djjblm6bg3owzwem3yngw2jnpyrlss4cgn7boo6.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   y => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_3, 0.25), kwargs = {})
triton_poi_fused_mul_2 = async_compile.triton('triton_poi_fused_mul_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x1)), xmask, eviction_policy='evict_last')
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x1 + (512*y0)), tmp2, xmask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/d4/cd4rutq6dguw32ry7yxef6j3gsekifdjcnpvhqont35d2scxcb7b.py
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
    num_stages=1,
    num_warps=2,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_bmm_3', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_bmm_3(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 512
    N = 512
    K = 256

    stride_aq = 131072
    stride_am = 256
    stride_ak = 1

    stride_bq = 131072
    stride_bk = 512
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
    xindex = idx_n + (512*idx_m) + (262144*idx_q)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/kt/ckt7cmqy523kjuyk2ey6ksb5ltswpxd5mvzae2nrksmkderjid6a.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   y => amax, any_1, div, eq, exp, full_default, logical_not, logical_not_1, sub, sum_1, where
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_5, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_5, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 4, 512, 512], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default, %div), kwargs = {})
triton_per_fused__safe_softmax_4 = async_compile.triton('triton_per_fused__safe_softmax_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_4', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__safe_softmax_4(in_ptr0, out_ptr4, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp1, 0))
    tmp4 = float("-inf")
    tmp5 = tmp0 == tmp4
    tmp6 = tmp5 == 0
    tmp7 = tmp6.to(tl.int64)
    tmp8 = (tmp7 != 0)
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(triton_helpers.any(tmp9, 0))
    tmp12 = tmp0 - tmp3
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp11 == 0
    tmp18 = tmp13 / tmp16
    tmp19 = 0.0
    tmp20 = tl.where(tmp17, tmp19, tmp18)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp20, None)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/yd/cydwznc2cwth2h2onouwcku5zi3iygbeaiuxqm7kwew2jlthcao4.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   y => bmm_1
# Graph fragment:
#   %bmm_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_6, %view_7), kwargs = {})
triton_tem_fused_bmm_5 = async_compile.triton('triton_tem_fused_bmm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=4,
    num_warps=8,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_bmm_5', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_bmm_5(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 512
    N = 256
    K = 512

    stride_aq = 262144
    stride_am = 512
    stride_ak = 1

    stride_bq = 256
    stride_bk = 1024
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
    xindex = idx_n + (256*idx_m) + (131072*idx_q)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cpu')
meta1 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_kumara/4b/c4bu7cvybqk5ge4kn65mvwkia7dkj55d3t5dtxzvxu5wi2iw4rgp.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 256
    x1 = (xindex // 256) % 4
    x2 = (xindex // 1024)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x2) + (131072*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/q6/cq64kwuolwkpizy7skq5ytmskcf3rigr75s46j4stnz3ubkdd6vi.py
# Topologically Sorted Source Nodes: [attn_out, h], Original ATen: [aten.addmm, aten.add]
# Source node to ATen node mapping:
#   attn_out => addmm_3
#   h => add
# Graph fragment:
#   %addmm_3 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_8, %view_9, %primals_9), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%addmm_3, %primals_3), kwargs = {})
triton_tem_fused_add_addmm_7 = async_compile.triton('triton_tem_fused_add_addmm_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=8,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(1, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_add_addmm_7', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_add_addmm_7(in_ptr0, arg_A, arg_B, in_ptr3, out_ptr1):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 512
    N = 1152
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1152
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
    xindex = idx_n + (1152*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tmp2 = tl.load(in_ptr3 + (tl.broadcast_to(xindex, acc.shape)), mask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), tmp3, mask)
''', device_str='cpu')
meta2 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}


# kernel path: /tmp/torchinductor_kumara/5y/c5yvzjfnnldfwrgml44i25vzhytq2xdv77ke3eruyclybtuy5kv3.py
# Topologically Sorted Source Nodes: [h, u, u_1], Original ATen: [aten.add, aten.addmm, aten.gelu]
# Source node to ATen node mapping:
#   h => add
#   u => addmm_4
#   u_1 => add_1, erf, mul_2, mul_3, mul_4
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%addmm_3, %primals_3), kwargs = {})
#   %addmm_4 : [num_users=3] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_10, %add, %primals_11), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_4, 0.5), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_4, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_3,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %add_1), kwargs = {})
triton_tem_fused_add_addmm_gelu_8 = async_compile.triton('triton_tem_fused_add_addmm_gelu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=2,
    num_warps=4,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(1, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_add_addmm_gelu_8', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_add_addmm_gelu_8(in_ptr0, arg_A, arg_B, out_ptr0, out_ptr1):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 16
    A = arg_A
    B = arg_B

    M = 512
    N = 6912
    K = 1152
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1152
    stride_ak = 1
    stride_bk = 6912
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
    xindex = idx_n + (6912*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), tmp9, mask)
''', device_str='cpu')
meta3 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_kumara/v6/cv6u4n2igpvow5nyxwcekque5tv3s2iiizj7ts5wdstuia4px72d.py
# Topologically Sorted Source Nodes: [u_1, mlp_out, x], Original ATen: [aten.gelu, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   mlp_out => addmm_5
#   u_1 => add_1, erf, mul_2, mul_3, mul_4
#   x => add_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_4, 0.5), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_4, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_3,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %add_1), kwargs = {})
#   %addmm_5 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%primals_12, %mul_4, %primals_13), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%addmm_5, %add), kwargs = {})
triton_tem_fused_add_addmm_gelu_9 = async_compile.triton('triton_tem_fused_add_addmm_gelu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=8,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(1, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_add_addmm_gelu_9', 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_add_addmm_gelu_9(in_ptr0, arg_A, arg_B, in_ptr3, out_ptr1):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = False
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 512
    N = 1152
    K = 6912
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 6912
    stride_ak = 1
    stride_bk = 1152
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
    xindex = idx_n + (1152*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tmp2 = tl.load(in_ptr3 + (tl.broadcast_to(xindex, acc.shape)), mask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), tmp3, mask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_kumara/h7/ch7pqmajkpot5hl4bxpkkenu4e3oqatuwyg7n2zwhu5flfeyj4p5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose]
# Source node to ATen node mapping:
# Graph fragment:
#   %permute_15 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1]), kwargs = {})
triton_poi_fused_transpose_10 = async_compile.triton('triton_poi_fused_transpose_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_cpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_transpose_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F62E01E27927831DD211807EEB30CA3E85443518C78A4F492A904AA8BCD314BD', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_transpose_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (256*y1) + (1024*y0)), tmp0, xmask)
''', device_str='cpu')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_2, (1152, 1024), (1024, 1))
    assert_size_stride(primals_3, (512, 1152), (1152, 1))
    assert_size_stride(primals_4, (1024, ), (1, ))
    assert_size_stride(primals_5, (1152, 1024), (1024, 1))
    assert_size_stride(primals_6, (1024, ), (1, ))
    assert_size_stride(primals_7, (1152, 1024), (1024, 1))
    assert_size_stride(primals_8, (1152, ), (1, ))
    assert_size_stride(primals_9, (1024, 1152), (1152, 1))
    assert_size_stride(primals_10, (6912, ), (1, ))
    assert_size_stride(primals_11, (1152, 6912), (6912, 1))
    assert_size_stride(primals_12, (1152, ), (1, ))
    assert_size_stride(primals_13, (6912, 1152), (1152, 1))
    buf0 = empty_strided_cpu((512, 1024), (1024, 1), torch.float32)
    # Topologically Sorted Source Nodes: [addmm], Original ATen: [aten.addmm]
    streamNone = get_raw_stream(None)
    _numa_p0_primals_1_n0 = torch.ops.aten.clone.default(primals_1, numa_node=0)
    _numa_p0_primals_1_n1 = torch.ops.aten.clone.default(primals_1, numa_node=1)
    _numa_p0_primals_3_n0 = torch.ops.aten.clone.default(primals_3, numa_node=0)
    _numa_p0_primals_3_n1 = torch.ops.aten.clone.default(primals_3, numa_node=1)
    _numa_p0_primals_2_n0 = torch.ops.aten.clone.default(primals_2, numa_node=0)
    _numa_p0_primals_2_n1 = torch.ops.aten.clone.default(primals_2, numa_node=1)
    _numa_p0_buf0_n0 = empty_strided_cpu_numa(buf0.size(), buf0.stride(), buf0.dtype, numa_node=0)
    _numa_p0_buf0_n1 = empty_strided_cpu_numa(buf0.size(), buf0.stride(), buf0.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_primals_1_n0, _numa_p0_primals_3_n0, _numa_p0_primals_2_n0, _numa_p0_buf0_n0), (_numa_p0_primals_1_n1, _numa_p0_primals_3_n1, _numa_p0_primals_2_n1, _numa_p0_buf0_n1)]
    triton_tem_fused_addmm_0.run(primals_1, primals_3, primals_2, buf0, grid=torch._inductor.kernel.mm_common.mm_grid(512, 1024, meta0), stream=streamNone, numa_args=_numa_p0_numa_args)
    del primals_1
    del primals_2
    buf1 = empty_strided_cpu((512, 1024), (1024, 1), torch.float32)
    # Topologically Sorted Source Nodes: [addmm_1], Original ATen: [aten.addmm]
    _numa_p0_primals_4_n0 = torch.ops.aten.clone.default(primals_4, numa_node=0)
    _numa_p0_primals_4_n1 = torch.ops.aten.clone.default(primals_4, numa_node=1)
    _numa_p0_primals_5_n0 = torch.ops.aten.clone.default(primals_5, numa_node=0)
    _numa_p0_primals_5_n1 = torch.ops.aten.clone.default(primals_5, numa_node=1)
    _numa_p0_buf1_n0 = empty_strided_cpu_numa(buf1.size(), buf1.stride(), buf1.dtype, numa_node=0)
    _numa_p0_buf1_n1 = empty_strided_cpu_numa(buf1.size(), buf1.stride(), buf1.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_primals_4_n0, _numa_p0_primals_3_n0, _numa_p0_primals_5_n0, _numa_p0_buf1_n0), (_numa_p0_primals_4_n1, _numa_p0_primals_3_n1, _numa_p0_primals_5_n1, _numa_p0_buf1_n1)]
    triton_tem_fused_addmm_0.run(primals_4, primals_3, primals_5, buf1, grid=torch._inductor.kernel.mm_common.mm_grid(512, 1024, meta0), stream=streamNone, numa_args=_numa_p0_numa_args)
    del primals_4
    del primals_5
    buf2 = empty_strided_cpu((512, 1024), (1024, 1), torch.float32)
    # Topologically Sorted Source Nodes: [addmm_2], Original ATen: [aten.addmm]
    _numa_p0_primals_6_n0 = torch.ops.aten.clone.default(primals_6, numa_node=0)
    _numa_p0_primals_6_n1 = torch.ops.aten.clone.default(primals_6, numa_node=1)
    _numa_p0_primals_7_n0 = torch.ops.aten.clone.default(primals_7, numa_node=0)
    _numa_p0_primals_7_n1 = torch.ops.aten.clone.default(primals_7, numa_node=1)
    _numa_p0_buf2_n0 = empty_strided_cpu_numa(buf2.size(), buf2.stride(), buf2.dtype, numa_node=0)
    _numa_p0_buf2_n1 = empty_strided_cpu_numa(buf2.size(), buf2.stride(), buf2.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_primals_6_n0, _numa_p0_primals_3_n0, _numa_p0_primals_7_n0, _numa_p0_buf2_n0), (_numa_p0_primals_6_n1, _numa_p0_primals_3_n1, _numa_p0_primals_7_n1, _numa_p0_buf2_n1)]
    triton_tem_fused_addmm_0.run(primals_6, primals_3, primals_7, buf2, grid=torch._inductor.kernel.mm_common.mm_grid(512, 1024, meta0), stream=streamNone, numa_args=_numa_p0_numa_args)
    del primals_6
    del primals_7
    buf3 = empty_strided_cpu((1, 4, 512, 256), (524288, 131072, 256, 1), torch.float32)
    buf19 = empty_strided_cpu((4, 256, 512), (256, 1, 1024), torch.float32)
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul, aten.transpose]
    _numa_p0_buf3_n0 = empty_strided_cpu_numa(buf3.size(), buf3.stride(), buf3.dtype, numa_node=0)
    _numa_p0_buf3_n1 = empty_strided_cpu_numa(buf3.size(), buf3.stride(), buf3.dtype, numa_node=1)
    _numa_p0_buf19_n0 = empty_strided_cpu_numa(buf19.size(), buf19.stride(), buf19.dtype, numa_node=0)
    _numa_p0_buf19_n1 = empty_strided_cpu_numa(buf19.size(), buf19.stride(), buf19.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_buf0_n0, _numa_p0_buf3_n0, _numa_p0_buf19_n0), (_numa_p0_buf0_n1, _numa_p0_buf3_n1, _numa_p0_buf19_n1)]
    triton_poi_fused_mul_transpose_1.run(buf0, buf3, buf19, 524288, grid=grid(524288), stream=streamNone, numa_args=_numa_p0_numa_args)
    _numa_p0_buf4_n0 = reinterpret_tensor(_numa_p0_buf0_n0, (1, 4, 256, 512), (524288, 131072, 512, 1), 0)
    _numa_p0_buf4_n1 = reinterpret_tensor(_numa_p0_buf0_n1, (1, 4, 256, 512), (524288, 131072, 512, 1), 0)
    buf4 = reinterpret_tensor(buf0, (1, 4, 256, 512), (524288, 131072, 512, 1), 0); del buf0  # reuse
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul]
    _numa_p0_numa_args = [(_numa_p0_buf1_n0, _numa_p0_buf4_n0), (_numa_p0_buf1_n1, _numa_p0_buf4_n1)]
    triton_poi_fused_mul_2.run(buf1, buf4, 1024, 512, grid=grid(1024, 512), stream=streamNone, numa_args=_numa_p0_numa_args)
    buf5 = empty_strided_cpu((4, 512, 512), (262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten.bmm]
    _numa_p0_buf5_n0 = empty_strided_cpu_numa(buf5.size(), buf5.stride(), buf5.dtype, numa_node=0)
    _numa_p0_buf5_n1 = empty_strided_cpu_numa(buf5.size(), buf5.stride(), buf5.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_buf3_n0, _numa_p0_buf4_n0, _numa_p0_buf5_n0), (_numa_p0_buf3_n1, _numa_p0_buf4_n1, _numa_p0_buf5_n1)]
    triton_tem_fused_bmm_3.run(buf3, buf4, buf5, grid=torch._inductor.kernel.bmm.bmm_grid(4, 512, 512, meta0), stream=streamNone, numa_args=_numa_p0_numa_args)
    buf10 = empty_strided_cpu((1, 4, 512, 512), (1048576, 262144, 512, 1), torch.float32)
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten._safe_softmax]
    _numa_p0_buf10_n0 = empty_strided_cpu_numa(buf10.size(), buf10.stride(), buf10.dtype, numa_node=0)
    _numa_p0_buf10_n1 = empty_strided_cpu_numa(buf10.size(), buf10.stride(), buf10.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_buf5_n0, _numa_p0_buf10_n0), (_numa_p0_buf5_n1, _numa_p0_buf10_n1)]
    triton_per_fused__safe_softmax_4.run(buf5, buf10, 2048, 512, grid=grid(2048), stream=streamNone, numa_args=_numa_p0_numa_args)
    del buf5
    _numa_p0_buf11_n0 = reinterpret_tensor(_numa_p0_buf3_n0, (4, 512, 256), (131072, 256, 1), 0)
    _numa_p0_buf11_n1 = reinterpret_tensor(_numa_p0_buf3_n1, (4, 512, 256), (131072, 256, 1), 0)
    buf11 = reinterpret_tensor(buf3, (4, 512, 256), (131072, 256, 1), 0); del buf3  # reuse
    # Topologically Sorted Source Nodes: [y], Original ATen: [aten.bmm]
    _numa_p0_numa_args = [(_numa_p0_buf10_n0, _numa_p0_buf2_n0, _numa_p0_buf11_n0), (_numa_p0_buf10_n1, _numa_p0_buf2_n1, _numa_p0_buf11_n1)]
    triton_tem_fused_bmm_5.run(buf10, buf2, buf11, grid=torch._inductor.kernel.bmm.bmm_grid(4, 512, 256, meta1), stream=streamNone, numa_args=_numa_p0_numa_args)
    _numa_p0_buf12_n0 = reinterpret_tensor(_numa_p0_buf1_n0, (1, 512, 4, 256), (524288, 1024, 256, 1), 0)
    _numa_p0_buf12_n1 = reinterpret_tensor(_numa_p0_buf1_n1, (1, 512, 4, 256), (524288, 1024, 256, 1), 0)
    buf12 = reinterpret_tensor(buf1, (1, 512, 4, 256), (524288, 1024, 256, 1), 0); del buf1  # reuse
    # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
    _numa_p0_numa_args = [(_numa_p0_buf11_n0, _numa_p0_buf12_n0), (_numa_p0_buf11_n1, _numa_p0_buf12_n1)]
    triton_poi_fused_clone_6.run(buf11, buf12, 524288, grid=grid(524288), stream=streamNone, numa_args=_numa_p0_numa_args)
    buf14 = empty_strided_cpu((512, 1152), (1152, 1), torch.float32)
    # Topologically Sorted Source Nodes: [attn_out, h], Original ATen: [aten.addmm, aten.add]
    _numa_p0_primals_8_n0 = torch.ops.aten.clone.default(primals_8, numa_node=0)
    _numa_p0_primals_8_n1 = torch.ops.aten.clone.default(primals_8, numa_node=1)
    _numa_p0_primals_9_n0 = torch.ops.aten.clone.default(primals_9, numa_node=0)
    _numa_p0_primals_9_n1 = torch.ops.aten.clone.default(primals_9, numa_node=1)
    _numa_p0_buf14_n0 = empty_strided_cpu_numa(buf14.size(), buf14.stride(), buf14.dtype, numa_node=0)
    _numa_p0_buf14_n1 = empty_strided_cpu_numa(buf14.size(), buf14.stride(), buf14.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_primals_8_n0, _numa_p0_buf12_n0, _numa_p0_primals_9_n0, _numa_p0_primals_3_n0, _numa_p0_buf14_n0), (_numa_p0_primals_8_n1, _numa_p0_buf12_n1, _numa_p0_primals_9_n1, _numa_p0_primals_3_n1, _numa_p0_buf14_n1)]
    triton_tem_fused_add_addmm_7.run(primals_8, buf12, primals_9, primals_3, buf14, grid=torch._inductor.kernel.mm_common.mm_grid(512, 1152, meta2), stream=streamNone, numa_args=_numa_p0_numa_args)
    del primals_8
    buf15 = empty_strided_cpu((512, 6912), (6912, 1), torch.float32)
    buf16 = empty_strided_cpu((512, 6912), (6912, 1), torch.float32)
    # Topologically Sorted Source Nodes: [h, u, u_1], Original ATen: [aten.add, aten.addmm, aten.gelu]
    _numa_p0_primals_10_n0 = torch.ops.aten.clone.default(primals_10, numa_node=0)
    _numa_p0_primals_10_n1 = torch.ops.aten.clone.default(primals_10, numa_node=1)
    _numa_p0_primals_11_n0 = torch.ops.aten.clone.default(primals_11, numa_node=0)
    _numa_p0_primals_11_n1 = torch.ops.aten.clone.default(primals_11, numa_node=1)
    _numa_p0_buf15_n0 = empty_strided_cpu_numa(buf15.size(), buf15.stride(), buf15.dtype, numa_node=0)
    _numa_p0_buf15_n1 = empty_strided_cpu_numa(buf15.size(), buf15.stride(), buf15.dtype, numa_node=1)
    _numa_p0_buf16_n0 = empty_strided_cpu_numa(buf16.size(), buf16.stride(), buf16.dtype, numa_node=0)
    _numa_p0_buf16_n1 = empty_strided_cpu_numa(buf16.size(), buf16.stride(), buf16.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_primals_10_n0, _numa_p0_buf14_n0, _numa_p0_primals_11_n0, _numa_p0_buf15_n0, _numa_p0_buf16_n0), (_numa_p0_primals_10_n1, _numa_p0_buf14_n1, _numa_p0_primals_11_n1, _numa_p0_buf15_n1, _numa_p0_buf16_n1)]
    triton_tem_fused_add_addmm_gelu_8.run(primals_10, buf14, primals_11, buf15, buf16, grid=torch._inductor.kernel.mm_common.mm_grid(512, 6912, meta3), stream=streamNone, numa_args=_numa_p0_numa_args)
    del primals_10
    buf18 = empty_strided_cpu((512, 1152), (1152, 1), torch.float32)
    # Topologically Sorted Source Nodes: [u_1, mlp_out, x], Original ATen: [aten.gelu, aten.addmm, aten.add]
    _numa_p0_primals_12_n0 = torch.ops.aten.clone.default(primals_12, numa_node=0)
    _numa_p0_primals_12_n1 = torch.ops.aten.clone.default(primals_12, numa_node=1)
    _numa_p0_primals_13_n0 = torch.ops.aten.clone.default(primals_13, numa_node=0)
    _numa_p0_primals_13_n1 = torch.ops.aten.clone.default(primals_13, numa_node=1)
    _numa_p0_buf18_n0 = empty_strided_cpu_numa(buf18.size(), buf18.stride(), buf18.dtype, numa_node=0)
    _numa_p0_buf18_n1 = empty_strided_cpu_numa(buf18.size(), buf18.stride(), buf18.dtype, numa_node=1)
    _numa_p0_numa_args = [(_numa_p0_primals_12_n0, _numa_p0_buf16_n0, _numa_p0_primals_13_n0, _numa_p0_buf14_n0, _numa_p0_buf18_n0), (_numa_p0_primals_12_n1, _numa_p0_buf16_n1, _numa_p0_primals_13_n1, _numa_p0_buf14_n1, _numa_p0_buf18_n1)]
    triton_tem_fused_add_addmm_gelu_9.run(primals_12, buf16, primals_13, buf14, buf18, grid=torch._inductor.kernel.mm_common.mm_grid(512, 1152, meta2), stream=streamNone, numa_args=_numa_p0_numa_args)
    del primals_12
    _numa_p0_buf20_n0 = reinterpret_tensor(_numa_p0_buf11_n0, (4, 512, 256), (256, 1024, 1), 0)
    _numa_p0_buf20_n1 = reinterpret_tensor(_numa_p0_buf11_n1, (4, 512, 256), (256, 1024, 1), 0)
    buf20 = reinterpret_tensor(buf11, (4, 512, 256), (256, 1024, 1), 0); del buf11  # reuse
    # Topologically Sorted Source Nodes: [], Original ATen: [aten.transpose]
    _numa_p0_numa_args = [(_numa_p0_buf4_n0, _numa_p0_buf20_n0), (_numa_p0_buf4_n1, _numa_p0_buf20_n1)]
    triton_poi_fused_transpose_10.run(buf4, buf20, 2048, 256, grid=grid(2048, 256), stream=streamNone, numa_args=_numa_p0_numa_args)
    return (buf18, buf10, buf15, reinterpret_tensor(primals_13, (1152, 6912), (1, 1152), 0), reinterpret_tensor(buf16, (6912, 512), (1, 6912), 0), reinterpret_tensor(primals_11, (6912, 1152), (1, 6912), 0), reinterpret_tensor(buf14, (1152, 512), (1, 1152), 0), reinterpret_tensor(primals_9, (1152, 1024), (1, 1152), 0), reinterpret_tensor(buf12, (1024, 512), (1, 1024), 0), reinterpret_tensor(buf2, (4, 256, 512), (256, 1, 1024), 0), buf19, buf20, reinterpret_tensor(primals_3, (1152, 512), (1, 1152), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1152, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((512, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((1152, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((1152, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((1024, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((6912, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((1152, 6912), (6912, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((6912, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
