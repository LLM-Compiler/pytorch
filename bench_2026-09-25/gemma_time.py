"""Time Gemma per-pass to find where the speedup is lost.

Isolates MLP TP vs attention TP vs both, with rotary on and off, so we can see
whether the cost is (a) attention sharding being degenerate at 4 heads / W=4,
(b) the per-socket rotary replication the alias-chain fix introduced, or (c)
the MLP path.
"""
import os, sys, time, json, statistics
MODE = sys.argv[1]
WS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SEQ = int(os.environ.get("SEQ", 2048))
NL = int(os.environ.get("NLAYERS", 18))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/gt_%s_w%d_s%d_%s" % (
    MODE, WS, SEQ, os.environ.get("NOROPE", "rope"))

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch, torch.nn as nn
from torch._inductor import config as ic
import simple_gemma_decoder_block_torch_compile_demo as _D
from simple_gemma_decoder_block_torch_compile_demo import (
    SimpleGemmaStack, _build_prefill_inputs)

if os.environ.get("NOROPE"):
    _D.apply_rotary_emb = lambda x, f: x

ic.max_autotune_gemm_backends = "TRITON"
torch._C._cpu._set_numa_enabled(True)
H_IN, NH, NKV, HD, MLP = 1152, 4, 1, 256, 6912

torch.manual_seed(0)
m = SimpleGemmaStack(num_layers=NL, hidden_size=H_IN, num_heads=NH,
                     num_kv_heads=NKV, head_dim=HD, intermediate_size=MLP,
                     attn_logit_softcapping=None, use_kv_cache=False).eval()
for p in m.parameters():
    nn.init.normal_(p, std=0.02)
torch.manual_seed(1)
args = _build_prefill_inputs(1, SEQ, H_IN, NKV, HD, NL, torch.device("cpu"),
                             use_kv_cache=False)

flags = {"_numa_attention_tp": False, "_numa_mlp_tp": False,
         "fx_graph_cache": False}
if MODE in ("mlp", "both"):
    flags.update({"_numa_mlp_tp": True, "_numa_mlp_tp_world_size": WS,
                  "_numa_mlp_tp_compute_shard": 0})
if MODE in ("attn", "both"):
    flags.update({"_numa_attention_tp": True, "_numa_attention_tp_world_size": WS,
                  "_numa_attention_tp_compute_shard": 0})

with ic.patch(flags), torch.no_grad():
    torch._dynamo.reset()
    c = torch.compile(m)
    for _ in range(3):
        c(*args)
    lat = []
    for _ in range(10):
        t0 = time.perf_counter(); c(*args); lat.append((time.perf_counter() - t0) * 1e3)

print(json.dumps({"mode": MODE, "ws": WS, "seq": SEQ,
                  "rope": not bool(os.environ.get("NOROPE")),
                  "mean_ms": statistics.fmean(lat),
                  "min_ms": min(lat)}), flush=True)
