import os, sys, glob
MODE = sys.argv[1]            # std | numa
WS   = int(sys.argv[2]) if len(sys.argv) > 2 else 4
ZERO = os.environ.get("ZERO", "none")
CACHE = "/tmp/mg_cache_%s_w%d_%s" % (MODE, WS, ZERO)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch
from torch._inductor import config as ic
import gpt2_multi_size_speedup_cli as G

G._enable_numa()
torch.manual_seed(0)
p = G.MODEL_PRESETS["117m"]
DFFN = int(os.environ.get("DFFN", p.d_ffn))
NL   = int(os.environ.get("NLAYERS", 1))
m = G.GPT2PreLN(vocab_size=G.DEF_VOCAB, context_size=1024, d_model=p.d_model,
                num_layers=NL, num_heads=p.num_heads, head_dim=p.head_dim,
                d_ffn=DFFN).eval()
x = torch.randint(0, G.DEF_VOCAB, (1, 128), dtype=torch.long)
with torch.no_grad():
    if "fc2" in ZERO:
        for b in m.blocks: b.mlp_fc2_w.zero_()
    if "wo" in ZERO:
        for b in m.blocks: b.Wo.zero_()

flags = {"_numa_mlp_tp": False, "_numa_attention_tp": False,
         "_numa_layer_norm_tp": False, "_numa_residual_tp": False,
         "_numa_mlp_tp_world_size": 0, "_numa_attention_tp_world_size": 0,
         "_numa_layer_norm_tp_world_size": 0, "force_disable_caches": True}
if MODE == "numa":
    flags["_numa_mlp_tp"] = True
    flags["_numa_mlp_tp_world_size"] = WS

with ic.patch(flags):
    c = torch.compile(m, mode="default", fullgraph=False)
    with torch.no_grad():
        out = c(x).detach().float().clone()

torch.save(out, "/tmp/mg_%s_w%d_%s_d%d_L%d.pt" % (MODE, WS, ZERO, DFFN, NL))
print("ZERO=%s DFFN=%d L=%d MODE=%s WS=%d mean=%.6f std=%.6f"
      % (ZERO, DFFN, NL, MODE, WS, out.mean(), out.std()))
