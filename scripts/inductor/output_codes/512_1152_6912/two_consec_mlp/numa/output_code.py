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


# kernel path: /tmp/torchinductor_shavin/r3/cr3ugov5n7rnqjpds73vlpybhlifj5sqnjoawifrcjz6h3f6lue7.py
# Topologically Sorted Source Nodes: [y1, y1_1], Original ATen: [aten.addmm, aten.gelu]
# Source node to ATen node mapping:
#   y1 => addmm
#   y1_1 => add, erf, mul, mul_1, mul_2
# NUMA shard tags:
#   l1_shard_out(shard=0/4,node=0) => addmm
#   y_shard(shard=0/4,node=0) => mul_2
# Graph fragment:
#   %addmm : [num_users=2] = call_function[target=torch.ops.aten.addmm.default](args = (%slice_tensor_2, %arg1_1, %slice_tensor), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %add), kwargs = {})
triton_tem_fused_addmm_gelu_0 = async_compile.triton('triton_tem_fused_addmm_gelu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=4,
    num_warps=1,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr1': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(3,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_gelu_0', 'backend_hash': '0F64F689F5FAE4D76568754EC720A8E1D7E00922253621D2889A87FD0F1068EC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_addmm_gelu_0(in_ptr0, arg_A, arg_B, out_ptr1):
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

    M = 2
    N = 2
    K = 4
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 4
    stride_ak = 1
    stride_bk = 8
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
    xindex = idx_n + (2*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
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
meta0 = {'GROUP_M': 8, 'EVEN_K': False, 'ALLOW_TF32': False, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}


# kernel path: /tmp/torchinductor_shavin/cg/ccguo3ubtieqbryygufozuafo3npqt2sxkrb5jf7ghd5olehvlcg.py
# Topologically Sorted Source Nodes: [y1_1, y1_2], Original ATen: [aten.gelu, aten.addmm]
# Source node to ATen node mapping:
#   y1_1 => add, erf, mul, mul_1, mul_2
#   y1_2 => addmm_1
# NUMA shard tags:
#   x_replica(shard=0/4,node=0) => addmm_1
#   y_shard(shard=0/4,node=0) => mul_2
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, 0.5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %add), kwargs = {})
#   %addmm_1 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg4_1, %mul_2, %slice_tensor_1), kwargs = {})
triton_tem_fused_addmm_gelu_1 = async_compile.triton('triton_tem_fused_addmm_gelu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=1,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_gelu_1', 'backend_hash': '0F64F689F5FAE4D76568754EC720A8E1D7E00922253621D2889A87FD0F1068EC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_addmm_gelu_1(in_ptr0, arg_A, arg_B, out_ptr0):
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

    M = 2
    N = 4
    K = 2
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2
    stride_ak = 1
    stride_bk = 4
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
    xindex = idx_n + (4*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cpu')


# kernel path: /tmp/torchinductor_shavin/eu/ceuinku7lz6tihuvqmjzckiapdcmoq3izc27u3wqokjs3xavkyry.py
# Topologically Sorted Source Nodes: [y2, y2_1], Original ATen: [aten.addmm, aten.gelu]
# Source node to ATen node mapping:
#   y2 => addmm_2
#   y2_1 => add_1, erf_1, mul_3, mul_4, mul_5
# NUMA shard tags:
#   l1_shard_out(shard=0/4,node=0) => addmm_2
#   y_shard(shard=0/4,node=0) => mul_5
# Graph fragment:
#   %addmm_2 : [num_users=2] = call_function[target=torch.ops.aten.addmm.default](args = (%slice_tensor_14, %addmm_1, %slice_tensor_12), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_2, 0.5), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_2, 0.7071067811865476), kwargs = {})
#   %erf_1 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_4,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_1, 1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %add_1), kwargs = {})
triton_tem_fused_addmm_gelu_2 = async_compile.triton('triton_tem_fused_addmm_gelu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=1,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr1': '*fp32'}, 'device': DeviceProperties(type='cpu', index=None, cc='', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None, warp_size=None), 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_gelu_2', 'backend_hash': '0F64F689F5FAE4D76568754EC720A8E1D7E00922253621D2889A87FD0F1068EC', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_tem_fused_addmm_gelu_2(in_ptr0, arg_A, arg_B, out_ptr1):
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

    M = 2
    N = 2
    K = 4
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 4
    stride_ak = 1
    stride_bk = 8
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
    xindex = idx_n + (2*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
    tmp1 = acc + tmp0
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 8), (8, 1))
    assert_size_stride(arg1_1, (2, 4), (4, 1))
    assert_size_stride(arg2_1, (8, ), (1, ))
    assert_size_stride(arg3_1, (8, 4), (4, 1))
    assert_size_stride(arg4_1, (4, ), (1, ))
    assert_size_stride(arg5_1, (4, 8), (8, 1))
    assert_size_stride(arg6_1, (8, ), (1, ))
    assert_size_stride(arg7_1, (8, 4), (4, 1))
    assert_size_stride(arg8_1, (4, ), (1, ))
    buf1 = empty_strided_cpu((2, 2), (2, 1), torch.float32)
    # Topologically Sorted Source Nodes: [y1, y1_1], Original ATen: [aten.addmm, aten.gelu]
    streamNone = get_raw_stream(None)
    _numa_p1_arg2_1_n0 = torch.ops.aten.clone.default(arg2_1, numa_node=0)
    _numa_p1_arg2_1_n1 = torch.ops.aten.clone.default(arg2_1, numa_node=1)
    _numa_p1_arg2_1_n2 = torch.ops.aten.clone.default(arg2_1, numa_node=2)
    _numa_p1_arg2_1_n3 = torch.ops.aten.clone.default(arg2_1, numa_node=3)
    _numa_p1_arg1_1_n0 = torch.ops.aten.clone.default(arg1_1, numa_node=0)
    _numa_p1_arg1_1_n1 = torch.ops.aten.clone.default(arg1_1, numa_node=1)
    _numa_p1_arg1_1_n2 = torch.ops.aten.clone.default(arg1_1, numa_node=2)
    _numa_p1_arg1_1_n3 = torch.ops.aten.clone.default(arg1_1, numa_node=3)
    _numa_p1_arg0_1_n0 = torch.ops.aten.clone.default(arg0_1, numa_node=0)
    _numa_p1_arg0_1_n1 = torch.ops.aten.clone.default(arg0_1, numa_node=1)
    _numa_p1_arg0_1_n2 = torch.ops.aten.clone.default(arg0_1, numa_node=2)
    _numa_p1_arg0_1_n3 = torch.ops.aten.clone.default(arg0_1, numa_node=3)
    _numa_p1_buf1_n0 = torch.ops.aten.clone.default(buf1, numa_node=0)
    _numa_p1_buf1_n1 = torch.ops.aten.clone.default(buf1, numa_node=1)
    _numa_p1_buf1_n2 = torch.ops.aten.clone.default(buf1, numa_node=2)
    _numa_p1_buf1_n3 = torch.ops.aten.clone.default(buf1, numa_node=3)
    _numa_p1_numa_args = [(_numa_p1_arg2_1_n0, _numa_p1_arg1_1_n0, _numa_p1_arg0_1_n0, _numa_p1_buf1_n0), (_numa_p1_arg2_1_n1, _numa_p1_arg1_1_n1, _numa_p1_arg0_1_n1, _numa_p1_buf1_n1), (_numa_p1_arg2_1_n2, _numa_p1_arg1_1_n2, _numa_p1_arg0_1_n2, _numa_p1_buf1_n2), (_numa_p1_arg2_1_n3, _numa_p1_arg1_1_n3, _numa_p1_arg0_1_n3, _numa_p1_buf1_n3)]
    triton_tem_fused_addmm_gelu_0.run(_numa_p1_arg2_1_n0, _numa_p1_arg1_1_n0, _numa_p1_arg0_1_n0, _numa_p1_buf1_n0, grid=torch._inductor.kernel.mm_common.mm_grid(2, 2, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_0.run(_numa_p1_arg2_1_n1, _numa_p1_arg1_1_n1, _numa_p1_arg0_1_n1, _numa_p1_buf1_n1, grid=torch._inductor.kernel.mm_common.mm_grid(2, 2, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_0.run(_numa_p1_arg2_1_n2, _numa_p1_arg1_1_n2, _numa_p1_arg0_1_n2, _numa_p1_buf1_n2, grid=torch._inductor.kernel.mm_common.mm_grid(2, 2, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_0.run(_numa_p1_arg2_1_n3, _numa_p1_arg1_1_n3, _numa_p1_arg0_1_n3, _numa_p1_buf1_n3, grid=torch._inductor.kernel.mm_common.mm_grid(2, 2, meta0), stream=streamNone)
    del arg0_1
    del arg1_1
    del arg2_1
    buf2 = empty_strided_cpu((2, 4), (4, 1), torch.float32)
    # Topologically Sorted Source Nodes: [y1_1, y1_2], Original ATen: [aten.gelu, aten.addmm]
    _numa_p1_arg4_1_n0 = torch.ops.aten.clone.default(arg4_1, numa_node=0)
    _numa_p1_arg4_1_n1 = torch.ops.aten.clone.default(arg4_1, numa_node=1)
    _numa_p1_arg4_1_n2 = torch.ops.aten.clone.default(arg4_1, numa_node=2)
    _numa_p1_arg4_1_n3 = torch.ops.aten.clone.default(arg4_1, numa_node=3)
    _numa_p1_arg3_1_n0 = torch.ops.aten.clone.default(arg3_1, numa_node=0)
    _numa_p1_arg3_1_n1 = torch.ops.aten.clone.default(arg3_1, numa_node=1)
    _numa_p1_arg3_1_n2 = torch.ops.aten.clone.default(arg3_1, numa_node=2)
    _numa_p1_arg3_1_n3 = torch.ops.aten.clone.default(arg3_1, numa_node=3)
    _numa_p1_buf2_n0 = torch.ops.aten.clone.default(buf2, numa_node=0)
    _numa_p1_buf2_n1 = torch.ops.aten.clone.default(buf2, numa_node=1)
    _numa_p1_buf2_n2 = torch.ops.aten.clone.default(buf2, numa_node=2)
    _numa_p1_buf2_n3 = torch.ops.aten.clone.default(buf2, numa_node=3)
    _numa_p1_numa_args = [(_numa_p1_arg4_1_n0, _numa_p1_buf1_n0, _numa_p1_arg3_1_n0, _numa_p1_buf2_n0), (_numa_p1_arg4_1_n1, _numa_p1_buf1_n1, _numa_p1_arg3_1_n1, _numa_p1_buf2_n1), (_numa_p1_arg4_1_n2, _numa_p1_buf1_n2, _numa_p1_arg3_1_n2, _numa_p1_buf2_n2), (_numa_p1_arg4_1_n3, _numa_p1_buf1_n3, _numa_p1_arg3_1_n3, _numa_p1_buf2_n3)]
    triton_tem_fused_addmm_gelu_1.run(_numa_p1_arg4_1_n0, _numa_p1_buf1_n0, _numa_p1_arg3_1_n0, _numa_p1_buf2_n0, grid=torch._inductor.kernel.mm_common.mm_grid(2, 4, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_1.run(_numa_p1_arg4_1_n1, _numa_p1_buf1_n1, _numa_p1_arg3_1_n1, _numa_p1_buf2_n1, grid=torch._inductor.kernel.mm_common.mm_grid(2, 4, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_1.run(_numa_p1_arg4_1_n2, _numa_p1_buf1_n2, _numa_p1_arg3_1_n2, _numa_p1_buf2_n2, grid=torch._inductor.kernel.mm_common.mm_grid(2, 4, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_1.run(_numa_p1_arg4_1_n3, _numa_p1_buf1_n3, _numa_p1_arg3_1_n3, _numa_p1_buf2_n3, grid=torch._inductor.kernel.mm_common.mm_grid(2, 4, meta0), stream=streamNone)
    del arg3_1
    del arg4_1
    buf4 = buf1; del buf1  # reuse
    # Topologically Sorted Source Nodes: [y2, y2_1], Original ATen: [aten.addmm, aten.gelu]
    _numa_p1_arg6_1_full_n0 = torch.ops.aten.clone.default(arg6_1, numa_node=0)
    _numa_p1_arg6_1_n0 = torch.ops.aten.slice.Tensor(_numa_p1_arg6_1_full_n0, 0, 0, 2, 1)
    _numa_p1_arg6_1_full_n1 = torch.ops.aten.clone.default(arg6_1, numa_node=1)
    _numa_p1_arg6_1_n1 = torch.ops.aten.slice.Tensor(_numa_p1_arg6_1_full_n1, 0, 2, 4, 1)
    _numa_p1_arg6_1_full_n2 = torch.ops.aten.clone.default(arg6_1, numa_node=2)
    _numa_p1_arg6_1_n2 = torch.ops.aten.slice.Tensor(_numa_p1_arg6_1_full_n2, 0, 4, 6, 1)
    _numa_p1_arg6_1_full_n3 = torch.ops.aten.clone.default(arg6_1, numa_node=3)
    _numa_p1_arg6_1_n3 = torch.ops.aten.slice.Tensor(_numa_p1_arg6_1_full_n3, 0, 6, 8, 1)
    _numa_p1_arg5_1_full_n0 = torch.ops.aten.clone.default(arg5_1, numa_node=0)
    _numa_p1_arg5_1_n0 = torch.ops.aten.slice.Tensor(_numa_p1_arg5_1_full_n0, 1, 0, 2, 1)
    _numa_p1_arg5_1_full_n1 = torch.ops.aten.clone.default(arg5_1, numa_node=1)
    _numa_p1_arg5_1_n1 = torch.ops.aten.slice.Tensor(_numa_p1_arg5_1_full_n1, 1, 2, 4, 1)
    _numa_p1_arg5_1_full_n2 = torch.ops.aten.clone.default(arg5_1, numa_node=2)
    _numa_p1_arg5_1_n2 = torch.ops.aten.slice.Tensor(_numa_p1_arg5_1_full_n2, 1, 4, 6, 1)
    _numa_p1_arg5_1_full_n3 = torch.ops.aten.clone.default(arg5_1, numa_node=3)
    _numa_p1_arg5_1_n3 = torch.ops.aten.slice.Tensor(_numa_p1_arg5_1_full_n3, 1, 6, 8, 1)
    _numa_p1_buf4_n0 = torch.ops.aten.clone.default(buf4, numa_node=0)
    _numa_p1_buf4_n1 = torch.ops.aten.clone.default(buf4, numa_node=1)
    _numa_p1_buf4_n2 = torch.ops.aten.clone.default(buf4, numa_node=2)
    _numa_p1_buf4_n3 = torch.ops.aten.clone.default(buf4, numa_node=3)
    _numa_p1_numa_args = [(_numa_p1_arg6_1_n0, _numa_p1_buf2_n0, _numa_p1_arg5_1_n0, _numa_p1_buf4_n0), (_numa_p1_arg6_1_n1, _numa_p1_buf2_n1, _numa_p1_arg5_1_n1, _numa_p1_buf4_n1), (_numa_p1_arg6_1_n2, _numa_p1_buf2_n2, _numa_p1_arg5_1_n2, _numa_p1_buf4_n2), (_numa_p1_arg6_1_n3, _numa_p1_buf2_n3, _numa_p1_arg5_1_n3, _numa_p1_buf4_n3)]
    triton_tem_fused_addmm_gelu_2.run(_numa_p1_arg6_1_n0, _numa_p1_buf2_n0, _numa_p1_arg5_1_n0, _numa_p1_buf4_n0, grid=torch._inductor.kernel.mm_common.mm_grid(2, 2, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_2.run(_numa_p1_arg6_1_n1, _numa_p1_buf2_n1, _numa_p1_arg5_1_n1, _numa_p1_buf4_n1, grid=torch._inductor.kernel.mm_common.mm_grid(2, 2, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_2.run(_numa_p1_arg6_1_n2, _numa_p1_buf2_n2, _numa_p1_arg5_1_n2, _numa_p1_buf4_n2, grid=torch._inductor.kernel.mm_common.mm_grid(2, 2, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_2.run(_numa_p1_arg6_1_n3, _numa_p1_buf2_n3, _numa_p1_arg5_1_n3, _numa_p1_buf4_n3, grid=torch._inductor.kernel.mm_common.mm_grid(2, 2, meta0), stream=streamNone)
    del arg5_1
    del arg6_1
    buf5 = buf2; del buf2  # reuse
    # Topologically Sorted Source Nodes: [y2_1, y2_2], Original ATen: [aten.gelu, aten.addmm]
    _numa_p1_arg8_1_n0 = torch.ops.aten.clone.default(arg8_1, numa_node=0)
    _numa_p1_arg8_1_zero_n1 = torch.ops.aten.zeros_like.default(arg8_1)
    _numa_p1_arg8_1_n1 = torch.ops.aten.clone.default(_numa_p1_arg8_1_zero_n1, numa_node=1)
    _numa_p1_arg8_1_zero_n2 = torch.ops.aten.zeros_like.default(arg8_1)
    _numa_p1_arg8_1_n2 = torch.ops.aten.clone.default(_numa_p1_arg8_1_zero_n2, numa_node=2)
    _numa_p1_arg8_1_zero_n3 = torch.ops.aten.zeros_like.default(arg8_1)
    _numa_p1_arg8_1_n3 = torch.ops.aten.clone.default(_numa_p1_arg8_1_zero_n3, numa_node=3)
    _numa_p1_arg7_1_full_n0 = torch.ops.aten.clone.default(arg7_1, numa_node=0)
    _numa_p1_arg7_1_n0 = torch.ops.aten.slice.Tensor(_numa_p1_arg7_1_full_n0, 0, 0, 2, 1)
    _numa_p1_arg7_1_full_n1 = torch.ops.aten.clone.default(arg7_1, numa_node=1)
    _numa_p1_arg7_1_n1 = torch.ops.aten.slice.Tensor(_numa_p1_arg7_1_full_n1, 0, 2, 4, 1)
    _numa_p1_arg7_1_full_n2 = torch.ops.aten.clone.default(arg7_1, numa_node=2)
    _numa_p1_arg7_1_n2 = torch.ops.aten.slice.Tensor(_numa_p1_arg7_1_full_n2, 0, 4, 6, 1)
    _numa_p1_arg7_1_full_n3 = torch.ops.aten.clone.default(arg7_1, numa_node=3)
    _numa_p1_arg7_1_n3 = torch.ops.aten.slice.Tensor(_numa_p1_arg7_1_full_n3, 0, 6, 8, 1)
    _numa_p1_buf5_n0 = torch.ops.aten.clone.default(buf5, numa_node=0)
    _numa_p1_buf5_n1 = torch.ops.aten.clone.default(buf5, numa_node=1)
    _numa_p1_buf5_n2 = torch.ops.aten.clone.default(buf5, numa_node=2)
    _numa_p1_buf5_n3 = torch.ops.aten.clone.default(buf5, numa_node=3)
    _numa_p1_numa_args = [(_numa_p1_arg8_1_n0, _numa_p1_buf4_n0, _numa_p1_arg7_1_n0, _numa_p1_buf5_n0), (_numa_p1_arg8_1_n1, _numa_p1_buf4_n1, _numa_p1_arg7_1_n1, _numa_p1_buf5_n1), (_numa_p1_arg8_1_n2, _numa_p1_buf4_n2, _numa_p1_arg7_1_n2, _numa_p1_buf5_n2), (_numa_p1_arg8_1_n3, _numa_p1_buf4_n3, _numa_p1_arg7_1_n3, _numa_p1_buf5_n3)]
    triton_tem_fused_addmm_gelu_1.run(_numa_p1_arg8_1_n0, _numa_p1_buf4_n0, _numa_p1_arg7_1_n0, _numa_p1_buf5_n0, grid=torch._inductor.kernel.mm_common.mm_grid(2, 4, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_1.run(_numa_p1_arg8_1_n1, _numa_p1_buf4_n1, _numa_p1_arg7_1_n1, _numa_p1_buf5_n1, grid=torch._inductor.kernel.mm_common.mm_grid(2, 4, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_1.run(_numa_p1_arg8_1_n2, _numa_p1_buf4_n2, _numa_p1_arg7_1_n2, _numa_p1_buf5_n2, grid=torch._inductor.kernel.mm_common.mm_grid(2, 4, meta0), stream=streamNone)
    triton_tem_fused_addmm_gelu_1.run(_numa_p1_arg8_1_n3, _numa_p1_buf4_n3, _numa_p1_arg7_1_n3, _numa_p1_buf5_n3, grid=torch._inductor.kernel.mm_common.mm_grid(2, 4, meta0), stream=streamNone)
    _numa_reduce_buf5_1 = torch.ops.aten.add.Tensor(_numa_p1_buf5_n0, _numa_p1_buf5_n1)
    _numa_reduce_buf5_2 = torch.ops.aten.add.Tensor(_numa_reduce_buf5_1, _numa_p1_buf5_n2)
    _numa_reduce_buf5_3 = torch.ops.aten.add.Tensor(_numa_reduce_buf5_2, _numa_p1_buf5_n3)
    _numa_p1_buf5_n0 = torch.ops.aten.clone.default(_numa_reduce_buf5_3, numa_node=0)
    _numa_p1_buf5_n1 = torch.ops.aten.clone.default(_numa_reduce_buf5_3, numa_node=1)
    _numa_p1_buf5_n2 = torch.ops.aten.clone.default(_numa_reduce_buf5_3, numa_node=2)
    _numa_p1_buf5_n3 = torch.ops.aten.clone.default(_numa_reduce_buf5_3, numa_node=3)
    buf5 = _numa_reduce_buf5_3
    del arg7_1
    del arg8_1
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2, 4), (4, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((8, 4), (4, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((4, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((8, 4), (4, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
