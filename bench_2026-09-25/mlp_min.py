import os, sys, glob
MODE = sys.argv[1]            # std | numa
WS   = int(sys.argv[2]) if len(sys.argv) > 2 else 4
CACHE = "/tmp/mlpmin_cache_%s_w%d" % (MODE, WS)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch, torch.nn as nn
from torch._inductor import config as ic
import gpt2_multi_size_speedup_cli as G

G._enable_numa()
torch.manual_seed(0)
H, F, B, S = 768, 3072, 1, 128

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(H, F)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(F, H)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

m = MLP().eval()
x = torch.randn(B, S, H)

flags = {"_numa_mlp_tp": False, "_numa_attention_tp": False,
         "_numa_layer_norm_tp": False, "_numa_residual_tp": False,
         "_numa_mlp_tp_world_size": 0, "_numa_attention_tp_world_size": 0,
         "_numa_layer_norm_tp_world_size": 0, "force_disable_caches": True, "_numa_mlp_tp_debug": True}
if MODE == "numa":
    flags["_numa_mlp_tp"] = True
    flags["_numa_mlp_tp_world_size"] = WS

with ic.patch(flags):
    c = torch.compile(m, mode="default", fullgraph=False)
    with torch.no_grad():
        out = c(x).detach().float().clone()

torch.save(out, "/tmp/mlpmin_%s_w%d.pt" % (MODE, WS))
codes = sorted(glob.glob(os.path.join(CACHE, "**", "output_code.py"), recursive=True))
print("MODE=%s WS=%d  mean=%.6f std=%.6f" % (MODE, WS, out.mean(), out.std()))
for cpath in codes:
    n = sum(1 for l in open(cpath) if "numa_args" in l)
    print("  code:", cpath, "numa_args_sites=", n)
