"""Minimal reproduction of the >2 GB SDPA-MATH segfault, standard arm only.

One transformer layer is enough: the overflow depends on B*H*S^2 for a single
attention intermediate, not on depth. Running with OMP_NUM_THREADS=1 and
faulthandler armed makes the faulting kernel name appear in the traceback
instead of 10 identical worker crashes.
"""
import argparse, faulthandler, os, sys

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="117m")
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--seq-length", type=int, default=4096)
ap.add_argument("--num-layers", type=int, default=1)
a = ap.parse_args()

faulthandler.enable(all_threads=True)
import gpt2_multi_size_speedup_cli as M
import torch

preset = M.MODEL_PRESETS[a.model]
M._enable_numa()
torch.manual_seed(0)
ctx = max(a.seq_length, M.DEF_CONTEXT)
model = M.GPT2PreLN(vocab_size=M.DEF_VOCAB, context_size=ctx,
                    d_model=preset.d_model, num_layers=a.num_layers,
                    num_heads=preset.num_heads, head_dim=preset.head_dim,
                    d_ffn=preset.d_ffn).eval()
ids = torch.randint(0, M.DEF_VOCAB, (a.batch_size, a.seq_length), dtype=torch.long)

H = preset.num_heads
bhs2 = a.batch_size * H * a.seq_length * a.seq_length
print(f"model={a.model} H={H} B={a.batch_size} S={a.seq_length} layers={a.num_layers}")
print(f"B*H*S^2 = {bhs2:,}  attn intermediate = {bhs2*4/2**30:.2f} GiB  "
      f"max_elem_byte_offset = {bhs2*4:,}  fits_i32={bhs2*4 < 2**31}")
sys.stdout.flush()

timing, out, info = M._run_path(model, ids, numa=False, world_size=4,
                                warmup=0, iters=1)
print("RAN OK  shape", tuple(out.shape), "finite", bool(torch.isfinite(out).all()))
print("mean_ms", timing.mean_ms)
