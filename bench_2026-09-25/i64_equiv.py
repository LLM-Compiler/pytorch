"""Does the int64 offset path compute the same numbers as the int32 path?

Runs one configuration small enough for both, once with offsets forced to
int64 and once left at int32, and compares the outputs elementwise. Both runs
build the model from the same seed, so any difference is the offset width.
"""
import os, subprocess, sys

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
HERE = f"{REPO}/bench_2026-09-25"
RUN = [sys.executable, f"{HERE}/split_arm.py", "--model", "gpt2-117m",
       "--batch-size", "1", "--seq-length", "2048", "--world-size", "4",
       "--arm", "standard", "--warmup", "0", "--iters", "1"]

for tag, force in (("i32", "0"), ("i64", "1")):
    env = dict(os.environ)
    env["INDUCTOR_FORCE_I64_OFFSETS"] = force
    env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/i64eq_{tag}"
    print(f"--- running {tag} (INDUCTOR_FORCE_I64_OFFSETS={force})", flush=True)
    r = subprocess.run(RUN + ["--out-prefix", f"/tmp/i64eq_{tag}_out"],
                       env=env, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{tag} FAILED rc={r.returncode}")
        print(r.stdout[-3000:]); print(r.stderr[-3000:]); sys.exit(1)
    print(f"{tag} ok")

import torch
a = torch.load("/tmp/i64eq_i32_out.out.pt")
b = torch.load("/tmp/i64eq_i64_out.out.pt")
d = (a - b).abs()
print(f"shape={tuple(a.shape)} max_abs={d.max().item():.6g} mean_abs={d.mean().item():.6g}")
print(f"bit_identical={bool(torch.equal(a, b))} both_finite="
      f"{bool(torch.isfinite(a).all() and torch.isfinite(b).all())}")
