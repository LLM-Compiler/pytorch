"""Measure the actual memory held by per-socket weight clones in a generated
NUMA-TP wrapper, split into:
  - *_full_n<k>  : column-parallel weights cloned at FULL size then sliced (the waste)
  - compact      : row-parallel weights sliced THEN cloned (already efficient)
Reads the generated output_code.py and resolves each clone's tensor size from
the model's actual parameter shapes.
"""
import os, sys, re, glob
CACHE = "/tmp/waste_762m"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))
import torch
from torch._inductor import config as ic
import gpt2_multi_size_speedup_cli as G

G._enable_numa()
torch.manual_seed(0)
p = G.MODEL_PRESETS["762m"]
m = G.GPT2PreLN(vocab_size=G.DEF_VOCAB, context_size=1024, d_model=p.d_model,
                num_layers=p.num_layers, num_heads=p.num_heads,
                head_dim=p.head_dim, d_ffn=p.d_ffn).eval()
x = torch.randint(0, G.DEF_VOCAB, (1, 128), dtype=torch.long)

flags = {"_numa_mlp_tp": True, "_numa_mlp_tp_world_size": 4,
         "_numa_attention_tp": True, "_numa_attention_tp_world_size": 4,
         "_numa_layer_norm_tp": False, "_numa_residual_tp": False,
         "_numa_layer_norm_tp_world_size": 0,
         "force_disable_caches": True}
with ic.patch(flags), torch.no_grad():
    torch._dynamo.reset()
    c = torch.compile(m, mode="default", fullgraph=False)
    c(x)

# map placeholder arg name -> numel, by replaying the flat arg order
code = max(glob.glob(CACHE + "/**/*.py", recursive=True), key=os.path.getsize)
src = open(code, errors="ignore").read()
print("generated code:", code)

# arg sizes come from the benchmark harness block at the bottom
sizes = {}
for mt in re.finditer(r"(arg\d+_1) = rand_strided\(\(([^)]*)\)", src):
    name, dims = mt.group(1), mt.group(2)
    nums = [int(v.strip()) for v in dims.split(",") if v.strip().isdigit()]
    n = 1
    for v in nums:
        n *= v
    sizes[name] = n

full_bytes = 0; full_args = set()
compact_bytes = 0; compact_n = 0
for line in src.splitlines():
    mt = re.search(r"_numa_weight_cache\['(_numa_p\d+_(arg\d+_1)(_full)?_n(\d))'\]\s*=\s*torch\.ops\.aten\.clone", line)
    if not mt:
        continue
    arg, is_full = mt.group(2), bool(mt.group(3))
    n = sizes.get(arg, 0)
    if is_full:
        full_bytes += n * 4
        full_args.add(arg)
    else:
        # slice-then-clone: compact, carries only its shard
        compact_n += 1

print()
print("FULL-SIZE column-parallel clones (the waste):")
print(f"  distinct weights full-cloned : {len(full_args)}")
print(f"  total bytes held             : {full_bytes/1e9:.2f} GB")
print(f"  if sharded 1/4 instead       : {full_bytes/4/1e9:.2f} GB")
print(f"  RECOVERABLE                  : {full_bytes*3/4/1e9:.2f} GB")
print()
print(f"slice-then-clone (already compact) clone sites: {compact_n}")
total_params = sum(q.numel() for q in m.parameters())
print(f"\nmodel params: {total_params/1e6:.0f}M  ({total_params*4/1e9:.2f} GB single copy)")
