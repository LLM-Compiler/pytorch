"""Run ONLY the NUMA-TP arm for one config and report peak RSS.

The benchmark CLI runs standard then NUMA-TP in a single process and holds
std_out plus the standard arm's compiled artifacts alive across the NUMA run.
If the NUMA arm alone fits comfortably, the blocked cells are recoverable by
splitting the two arms into separate processes -- no change to the NUMA code.
"""
import os, sys, resource
MODEL = os.environ.get("MODEL", "762m")
B = int(os.environ.get("B", 1))
S = int(os.environ.get("S", 2048))
W = int(os.environ.get("W", 4))
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/numaonly_{MODEL}_b{B}_s{S}"

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch
from torch._inductor import config as ic
import gpt2_multi_size_speedup_cli as G

G._enable_numa()
torch.manual_seed(0)
p = G.MODEL_PRESETS[MODEL]
m = G.GPT2PreLN(vocab_size=G.DEF_VOCAB, context_size=max(1024, S),
                d_model=p.d_model, num_layers=p.num_layers,
                num_heads=p.num_heads, head_dim=p.head_dim, d_ffn=p.d_ffn).eval()
x = torch.randint(0, G.DEF_VOCAB, (B, S), dtype=torch.long)

flags = {"_numa_mlp_tp": True, "_numa_mlp_tp_world_size": W,
         "_numa_attention_tp": True, "_numa_attention_tp_world_size": W,
         "_numa_layer_norm_tp": False, "_numa_residual_tp": False,
         "_numa_layer_norm_tp_world_size": 0}

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0

print(f"config: {MODEL} B={B} S={S} W={W}", flush=True)
print(f"  rss after model build : {rss_gb():.1f} GB", flush=True)
with ic.patch(flags), torch.no_grad():
    torch._dynamo.reset()
    c = torch.compile(m, mode="default", fullgraph=False)
    out = c(x)
    print(f"  rss after 1st forward : {rss_gb():.1f} GB", flush=True)
    for _ in range(2):
        out = c(x)
print(f"  PEAK RSS (numa arm only): {rss_gb():.1f} GB", flush=True)
print("  RESULT: completed without error", flush=True)
