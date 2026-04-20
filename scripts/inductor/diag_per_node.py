#!/usr/bin/env python3
"""Per-node buffer diagnostic for the NUMA attention output_code.py."""
import torch, runpy, sys, inspect, textwrap, ast, re
from torch._dynamo.testing import rand_strided

torch._C._cpu._set_numa_enabled(True)
sys.argv = ['test']
mod = runpy.run_path(
    'scripts/inductor/output_codes/attention/B_1_S_6_H_in_12_H_4_E_8_H_qkv_32_N_2/numa/output_code.py',
    run_name='__debug__'
)
call_fn = mod['call']
src = textwrap.dedent(inspect.getsource(call_fn))

expected_raw = [
    ((12, 32), (32, 1)),
    ((6, 12), (12, 1)),
    ((32,), (1,)),
    ((12, 32), (32, 1)),
    ((32,), (1,)),
    ((12, 32), (32, 1)),
    ((32,), (1,)),
    ((32, 12), (12, 1)),
    ((12,), (1,)),
]

def __check(label, t):
    if not isinstance(t, torch.Tensor):
        print(f"  {label}: not tensor", flush=True)
        return
    f = bool(torch.isfinite(t).all().item())
    tf = t.detach().float()
    print(f"  {label}: finite={f} min={float(tf.min()):.4g} max={float(tf.max()):.4g} shape={tuple(t.shape)}", flush=True)

lines = src.splitlines(keepends=True)
out_lines = []
for line in lines:
    if 'triton_tem_fused_addmm_10.run(' in line:
        out_lines += [
            "    __check('buf12_n0_pre', _numa_p0_buf12_n0)\n",
            "    __check('buf12_n1_pre', _numa_p0_buf12_n1)\n",
            "    __check('Wo_n0', _numa_p0_arg7_1_n0)\n",
            "    __check('Wo_n1', _numa_p0_arg7_1_n1)\n",
            "    __check('bias_n0', _numa_p0_arg8_1_n0)\n",
            "    __check('bias_n1', _numa_p0_arg8_1_n1)\n",
            line,
            "    __check('buf13_n0_post', _numa_p0_buf13_n0)\n",
            "    __check('buf13_n1_post', _numa_p0_buf13_n1)\n",
        ]
    elif 'triton_poi_fused_clone_9.run(' in line:
        out_lines += [
            "    __check('buf11_n0_in', _numa_p0_buf11_n0)\n",
            "    __check('buf11_n1_in', _numa_p0_buf11_n1)\n",
            line,
            "    __check('buf12_n0_post_clone9', _numa_p0_buf12_n0)\n",
            "    __check('buf12_n1_post_clone9', _numa_p0_buf12_n1)\n",
        ]
    elif 'triton_poi_fused__safe_softmax_7.run(' in line:
        out_lines += [
            "    __check('buf10_n0_pre_sm7', _numa_p0_buf10_n0)\n",
            "    __check('buf10_n1_pre_sm7', _numa_p0_buf10_n1)\n",
        ]
        out_lines.append(line)
        out_lines += [
            "    __check('buf10_n0_post_sm7', _numa_p0_buf10_n0)\n",
            "    __check('buf10_n1_post_sm7', _numa_p0_buf10_n1)\n",
        ]
    elif 'triton_poi_fused__safe_softmax_6.run(' in line:
        out_lines += [
            "    __check('buf7_n0_pre_sm6', _numa_p0_buf7_n0)\n",
            "    __check('buf7_n1_pre_sm6', _numa_p0_buf7_n1)\n",
            "    __check('buf8_n0_pre_sm6', _numa_p0_buf8_n0)\n",
            "    __check('buf8_n1_pre_sm6', _numa_p0_buf8_n1)\n",
        ]
        out_lines.append(line)
        out_lines += [
            "    __check('buf8_n0_post_sm6', _numa_p0_buf8_n0)\n",
            "    __check('buf8_n1_post_sm6', _numa_p0_buf8_n1)\n",
        ]
    elif 'triton_poi_fused__safe_softmax_5.run(' in line:
        out_lines.append(line)
        out_lines += [
            "    __check('buf7_n0_post_sm5', _numa_p0_buf7_n0)\n",
            "    __check('buf7_n1_post_sm5', _numa_p0_buf7_n1)\n",
        ]
    elif 'triton_poi_fused__safe_softmax_4.run(' in line:
        out_lines += [
            "    __check('buf4_n0_pre_sm4', _numa_p0_buf4_n0)\n",
            "    __check('buf4_n1_pre_sm4', _numa_p0_buf4_n1)\n",
        ]
        out_lines.append(line)
        out_lines += [
            "    __check('buf5_n0_post_sm4', _numa_p0_buf5_n0)\n",
            "    __check('buf6_n0_post_sm4', _numa_p0_buf6_n0)\n",
            "    __check('buf6_n1_post_sm4', _numa_p0_buf6_n1)\n",
        ]
    elif 'triton_tem_fused_bmm_3.run(' in line:
        out_lines.append(line)
        out_lines += [
            "    __check('buf4_n0_post_bmm3', _numa_p0_buf4_n0)\n",
            "    __check('buf4_n1_post_bmm3', _numa_p0_buf4_n1)\n",
        ]
    elif 'triton_tem_fused_bmm_8.run(' in line:
        out_lines += [
            "    __check('buf10_n0_pre_bmm8', _numa_p0_buf10_n0)\n",
            "    __check('buf10_n1_pre_bmm8', _numa_p0_buf10_n1)\n",
            "    __check('buf9_n0_pre_bmm8', _numa_p0_buf9_n0)\n",
            "    __check('buf9_n1_pre_bmm8', _numa_p0_buf9_n1)\n",
            line,
            "    __check('buf11_n0_post_bmm8', _numa_p0_buf11_n0)\n",
            "    __check('buf11_n1_post_bmm8', _numa_p0_buf11_n1)\n",
        ]
    else:
        out_lines.append(line)

patched = "".join(out_lines).replace('def call(', 'def call_d(', 1)
g = dict(mod)
g['__check'] = __check
exec(patched, g)

torch.manual_seed(0)
call_args = [rand_strided(s, st, device='cpu', dtype=torch.float32) for s, st in expected_raw]
print("=== Running diagnostic ===", flush=True)
result = g['call_d'](list(call_args))
print(f"Final: finite={bool(torch.isfinite(result[0]).all())} shape={result[0].shape}", flush=True)
