"""Probe defect 9: does forcing the rotary input contiguous let B=4 S=512 compile?

view_as_complex rejects a layout whose stride is odd on any non-last axis. With
world_size == num_heads the head axis shards down to SIZE 1, where the stride is
semantically meaningless but Inductor may still pick an odd one (observed:
stride (131072, 1, 256, 2, 1)). A contiguous tensor of that shape has strides
(131072, 131072, 256, 2, 1) -- all even except the last -- so it should satisfy
the check.
"""
import os, sys
MODE = os.environ.get("CONTIG", "0")
B = int(os.environ.get("B", 4)); S = int(os.environ.get("S", 512))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/d9_{MODE}_b{B}_s{S}"

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch, torch.nn as nn
from torch._inductor import config as ic
import simple_gemma_decoder_block_torch_compile_demo as D
from simple_gemma_decoder_block_torch_compile_demo import (
    SimpleGemmaStack, _build_prefill_inputs)

if MODE == "1":
    def rope_contig(x, freqs_cis):
        x_ = torch.view_as_complex(
            torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1)
            .contiguous())
        o = torch.view_as_real(x_ * freqs_cis).type_as(x)
        o = torch.cat(torch.chunk(o, 2, dim=-1), dim=-2)
        return o.reshape(o.shape[0], o.shape[1], o.shape[2], -1).transpose(1, 2)
    D.apply_rotary_emb = rope_contig

ic.max_autotune_gemm_backends = "TRITON"
torch._C._cpu._set_numa_enabled(True)
H_IN, NH, NKV, HD, MLP, NL = 1152, 4, 1, 256, 6912, 18
torch.manual_seed(0)
m = SimpleGemmaStack(num_layers=NL, hidden_size=H_IN, num_heads=NH, num_kv_heads=NKV,
                     head_dim=HD, intermediate_size=MLP, attn_logit_softcapping=None,
                     use_kv_cache=False).eval()
for p in m.parameters():
    nn.init.normal_(p, std=0.02)
torch.manual_seed(1)
args = _build_prefill_inputs(B, S, H_IN, NKV, HD, NL, torch.device("cpu"), use_kv_cache=False)

flags = {"_numa_attention_tp": True, "_numa_attention_tp_world_size": 4,
         "_numa_attention_tp_compute_shard": 0,
         "_numa_mlp_tp": True, "_numa_mlp_tp_world_size": 4,
         "_numa_mlp_tp_compute_shard": 0, "fx_graph_cache": False}
try:
    with ic.patch(flags), torch.no_grad():
        torch._dynamo.reset()
        c = torch.compile(m)
        out = c(*args)
    print(f"RESULT contig={MODE} B={B} S={S}: COMPILED OK, out={tuple(out[0].shape) if isinstance(out,(tuple,list)) else tuple(out.shape)}")
except Exception as e:
    print(f"RESULT contig={MODE} B={B} S={S}: FAILED -> {str(e)[:150]}")
