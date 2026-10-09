"""Discriminate what the Gemma GeGLU NUMA path is ACTUALLY computing.

Runs the 1-layer model in EAGER mode with candidate (possibly wrong) GeGLU
implementations and correlates each against the saved NUMA output. Eager only:
no compile, small memory, safe to run alongside the sweep.

H0 is the mathematically correct sharded form; if it matches STD exactly the
harness is sound, and whichever H matches NUMA tells us the real defect.
"""
import os, sys, torch, torch.nn.functional as F
REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
from simple_gemma_decoder_block_torch_compile_demo import (
    SimpleGemmaStack, _build_prefill_inputs, GemmaMLP)
import torch.nn as nn

H_IN, NH, NKV, HD, MLP, W, S = 1152, 4, 1, 256, 6912, 4, 128
SD = MLP // W
sl = lambda i: slice(i * SD, (i + 1) * SD)

torch.manual_seed(0)
model = SimpleGemmaStack(num_layers=1, hidden_size=H_IN, num_heads=NH,
                         num_kv_heads=NKV, head_dim=HD, intermediate_size=MLP,
                         attn_logit_softcapping=None, use_kv_cache=False).eval()
for p in model.parameters():
    nn.init.normal_(p, std=0.02)
torch.manual_seed(1)
x, freqs, kvw, kvc, mask = _build_prefill_inputs(
    1, S, H_IN, NKV, HD, 1, torch.device("cpu"), use_kv_cache=False)

std = torch.load("/tmp/gc_std_w4_L1_s128.pt", weights_only=True).float().flatten()
numa = torch.load("/tmp/gc_mlp_w4_L1_s128.pt", weights_only=True).float().flatten()

def corr(a, b):
    ac, bc = a - a.mean(), b - b.mean()
    return float((ac * bc).sum() / (ac.norm() * bc.norm()))

G = lambda t: F.gelu(t, approximate="tanh")

HYP = {}
# correct sharded GeGLU
HYP["H0_correct"] = lambda s, x: sum(
    (G(F.linear(x, s.gate_proj[sl(i)])) * F.linear(x, s.up_proj[sl(i)]))
    @ s.down_proj[:, sl(i)].T for i in range(W))
# only shard 0 contributes (reduction dropped)
HYP["H1_shard0_only"] = lambda s, x: (
    G(F.linear(x, s.gate_proj[sl(0)])) * F.linear(x, s.up_proj[sl(0)])
) @ s.down_proj[:, sl(0)].T
# gate/up shard mispairing: gate_i paired with up_0
HYP["H2_gate_i_up_0"] = lambda s, x: sum(
    (G(F.linear(x, s.gate_proj[sl(i)])) * F.linear(x, s.up_proj[sl(0)]))
    @ s.down_proj[:, sl(i)].T for i in range(W))
# gate_0 paired with up_i
HYP["H3_gate_0_up_i"] = lambda s, x: sum(
    (G(F.linear(x, s.gate_proj[sl(0)])) * F.linear(x, s.up_proj[sl(i)]))
    @ s.down_proj[:, sl(i)].T for i in range(W))
# down_w sliced on the wrong axis (rows instead of cols)
HYP["H4_down_wrong_dim"] = lambda s, x: sum(
    (G(F.linear(x, s.gate_proj[sl(i)])) * F.linear(x, s.up_proj[sl(i)]))
    @ s.down_proj[:, sl(0)].T for i in range(W))
# gelu applied to the FUSED product instead of to gate only
HYP["H5_gelu_on_fuse"] = lambda s, x: sum(
    G(F.linear(x, s.gate_proj[sl(i)]) * F.linear(x, s.up_proj[sl(i)]))
    @ s.down_proj[:, sl(i)].T for i in range(W))
# gate and up weights swapped
HYP["H6_gate_up_swapped"] = lambda s, x: sum(
    (G(F.linear(x, s.up_proj[sl(i)])) * F.linear(x, s.gate_proj[sl(i)]))
    @ s.down_proj[:, sl(i)].T for i in range(W))
# every socket computes the FULL (unsharded) MLP, then summed -> W x overcount
HYP["H7_full_times_W"] = lambda s, x: W * (
    (G(F.linear(x, s.gate_proj)) * F.linear(x, s.up_proj)) @ s.down_proj.T)

orig = GemmaMLP.forward
print(f"{'hypothesis':<22} {'corr vs NUMA':>13} {'corr vs STD':>12}")
print("-" * 50)
for name, fn in HYP.items():
    GemmaMLP.forward = lambda self, xx, _f=fn: _f(self, xx)
    with torch.no_grad():
        out = model(x, freqs, kvw, kvc, mask)
    out = (out[0] if isinstance(out, (tuple, list)) else out).detach().float().flatten()
    print(f"{name:<22} {corr(out, numa):>+13.6f} {corr(out, std):>+12.6f}")
GemmaMLP.forward = orig
