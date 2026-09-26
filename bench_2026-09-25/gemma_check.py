import os, sys, glob
MODE = sys.argv[1]                       # std | numa | attn | mlp
WS   = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NL   = int(os.environ.get("NLAYERS", 2))
SEQ  = int(os.environ.get("SEQ", 128))
TAG  = "%s_w%d_L%d_s%d" % (MODE, WS, NL, SEQ)
CACHE = "/tmp/gc_cache_" + TAG
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch, torch.nn as nn
from torch._inductor import config as ic
from simple_gemma_decoder_block_torch_compile_demo import (
    SimpleGemmaStack, _build_prefill_inputs,
)

ic.max_autotune_gemm_backends = "TRITON"
cpu_mod = torch._C._cpu
cpu_mod._set_numa_enabled(True)
print("NUMA nodes:", cpu_mod._num_numa_nodes(), flush=True)

H_IN, NH, NKV, HD, MLP = 1152, 4, 1, 256, 6912

torch.manual_seed(0)
model = SimpleGemmaStack(num_layers=NL, hidden_size=H_IN, num_heads=NH,
                         num_kv_heads=NKV, head_dim=HD, intermediate_size=MLP,
                         attn_logit_softcapping=None, use_kv_cache=False).eval()
for p in model.parameters():
    nn.init.normal_(p, std=0.02)
torch.manual_seed(1)
x, freqs_cis, kv_write_indices, kv_caches, mask = _build_prefill_inputs(
    1, SEQ, H_IN, NKV, HD, NL, torch.device("cpu"), use_kv_cache=False)

flags = {"_numa_attention_tp": False, "_numa_mlp_tp": False,
         "fx_graph_cache": False, "force_disable_caches": True}
if MODE in ("numa", "attn"):
    flags.update({"_numa_attention_tp": True, "_numa_attention_tp_world_size": WS,
                  "_numa_attention_tp_compute_shard": 0})
if MODE in ("numa", "mlp"):
    flags.update({"_numa_mlp_tp": True, "_numa_mlp_tp_world_size": WS,
                  "_numa_mlp_tp_compute_shard": 0})

with ic.patch(flags), torch.no_grad():
    torch._dynamo.reset()
    compiled = torch.compile(model)
    out = compiled(x, freqs_cis, kv_write_indices, kv_caches, mask)

out = (out[0] if isinstance(out, (tuple, list)) else out).detach().float().clone()
torch.save(out, "/tmp/gc_%s.pt" % TAG)
sites = 0
for f in glob.glob(CACHE + "/**/*.py", recursive=True):
    sites += sum(1 for l in open(f, errors="ignore") if "numa_args=" in l)
print("%-22s mean=%+.6f std=%.6f numa_args_sites=%d" % (TAG, out.mean(), out.std(), sites), flush=True)
