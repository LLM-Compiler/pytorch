import os, sys
PASSES = os.environ.get("PASSES", "both")     # std | mlp | attn | both
WS     = int(os.environ.get("WS", 4))
NL     = int(os.environ.get("NLAYERS", 1))
SEQ    = int(os.environ.get("SEQ", 128))
TAG    = "%s_w%d_L%d_s%d" % (PASSES, WS, NL, SEQ)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/nc_cache_" + TAG

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch
from torch._inductor import config as ic
import gpt2_multi_size_speedup_cli as G

G._enable_numa()
torch.manual_seed(0)
p = G.MODEL_PRESETS["117m"]
m = G.GPT2PreLN(vocab_size=G.DEF_VOCAB, context_size=1024, d_model=p.d_model,
                num_layers=NL, num_heads=p.num_heads, head_dim=p.head_dim,
                d_ffn=p.d_ffn).eval()
x = torch.randint(0, G.DEF_VOCAB, (1, SEQ), dtype=torch.long)

flags = {"_numa_mlp_tp": False, "_numa_attention_tp": False,
         "_numa_layer_norm_tp": False, "_numa_residual_tp": False,
         "_numa_mlp_tp_world_size": 0, "_numa_attention_tp_world_size": 0,
         "_numa_layer_norm_tp_world_size": 0, "force_disable_caches": True}
if PASSES in ("mlp", "both"):
    flags["_numa_mlp_tp"] = True;       flags["_numa_mlp_tp_world_size"] = WS
if PASSES in ("attn", "both"):
    flags["_numa_attention_tp"] = True; flags["_numa_attention_tp_world_size"] = WS

with ic.patch(flags):
    c = torch.compile(m, mode="default", fullgraph=False)
    with torch.no_grad():
        out = c(x).detach().float().clone()

torch.save(out, "/tmp/nc_%s.pt" % TAG)
print("%-22s mean=%+.6f std=%.6f" % (TAG, out.mean(), out.std()), flush=True)
