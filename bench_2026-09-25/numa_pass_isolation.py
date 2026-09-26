#!/usr/bin/env python
"""Isolate which NUMA-TP FX pass breaks numerical correctness.

Runs ONE pass combination per invocation so that a segfault kills only that
combination. A driver loop invokes it repeatedly.

  python numa_pass_isolation.py --mode ref   --out DIR
  python numa_pass_isolation.py --mode combo --out DIR --combo mlp,attention

Reports for each combination:
  sites      number of emitted ``numa_args=`` call sites (0 => TP never fired,
             so the comparison is vacuous rather than a pass)
  codehash   md5 over the generated Inductor python, to detect combinations
             that produce byte-identical code (i.e. flags with no effect)
  max_abs / max_rel / finite / verdict against the all-passes-off reference
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

REPO = "/home/shavin/llm-cpu-compiler/pytorch"
sys.path.insert(0, os.path.join(REPO, "scripts", "inductor"))

import torch  # noqa: E402
from torch._inductor import config as inductor_config  # noqa: E402

import gpt2_multi_size_speedup_cli as G  # noqa: E402

PASSES = ("layer_norm", "attention", "mlp", "residual")


def numa_arg_sites(cache_dir):
    try:
        out = subprocess.run(["grep", "-rho", "numa_args=", cache_dir],
                             capture_output=True, text=True, timeout=180)
        return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:
        return -1


def code_hash(cache_dir):
    """md5 over every generated .py, so identical codegen is detectable."""
    h = hashlib.md5()
    paths = []
    for root, _dirs, files in os.walk(cache_dir):
        for fn in files:
            if fn.endswith(".py"):
                paths.append(os.path.join(root, fn))
    for p in sorted(paths):
        try:
            with open(p, "rb") as fh:
                h.update(fh.read())
        except OSError:
            pass
    return h.hexdigest()[:12], len(paths)


def flags_for(combo, world_size):
    f = {
        "_numa_layer_norm_tp": False,
        "_numa_attention_tp": False,
        "_numa_mlp_tp": False,
        "_numa_residual_tp": False,
        "_numa_lm_head_tp": False,
        "_numa_layer_norm_tp_world_size": 0,
        "_numa_attention_tp_world_size": 0,
        "_numa_mlp_tp_world_size": 0,
        "_numa_lm_head_tp_world_size": 0,
    }
    for name in combo:
        f["_numa_%s_tp" % name] = True
        key = "_numa_%s_tp_world_size" % name
        if key in f:
            f[key] = world_size
    return f


def build(args):
    """Deterministic: identical weights and inputs in every invocation."""
    torch.manual_seed(0)
    preset = G.MODEL_PRESETS["117m"]
    n_layers = args.num_layers if args.num_layers else preset.num_layers
    model = G.GPT2PreLN(
        vocab_size=G.DEF_VOCAB,
        context_size=max(args.seq_length, G.DEF_CONTEXT),
        d_model=preset.d_model,
        num_layers=n_layers,
        num_heads=preset.num_heads,
        head_dim=preset.head_dim,
        d_ffn=preset.d_ffn,
    ).eval()
    input_ids = torch.randint(
        0, G.DEF_VOCAB, (args.batch_size, args.seq_length), dtype=torch.long
    )
    return model, input_ids


def run(model, input_ids, combo, world_size):
    # Cache dirs are left in place on purpose: deleting one breaks Inductor's
    # compile worker pool for every later compile in the same process.
    cache_dir = tempfile.mkdtemp(prefix="numa_iso_cache_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    torch._dynamo.reset()
    with inductor_config.patch(flags_for(combo, world_size)):
        compiled = torch.compile(model, mode="default", fullgraph=False)
        with torch.no_grad():
            out = compiled(input_ids).detach().clone()
    chash, nfiles = code_hash(cache_dir)
    return out, numa_arg_sites(cache_dir), chash, nfiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ref", "combo"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--combo", default="")
    ap.add_argument("--world-size", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-length", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=None)
    args = ap.parse_args()

    G._enable_numa()
    os.makedirs(args.out, exist_ok=True)
    ref_path = os.path.join(args.out, "ref.pt")

    model, input_ids = build(args)

    if args.mode == "ref":
        out, sites, chash, nfiles = run(model, input_ids, (), args.world_size)
        torch.save(out, ref_path)
        print(json.dumps({"mode": "ref", "sites": sites, "codehash": chash,
                          "files": nfiles, "shape": list(out.shape)}))
        return 0

    combo = tuple(x for x in args.combo.split(",") if x)
    for name in combo:
        if name not in PASSES:
            raise SystemExit("unknown pass %r (known: %s)" % (name, PASSES))

    ref = torch.load(ref_path)
    out, sites, chash, nfiles = run(model, input_ids, combo, args.world_size)

    d = (out.float() - ref.float()).abs()
    max_abs = float(d.max())
    max_rel = float((d / ref.float().abs().clamp_min(1e-12)).max())
    finite = bool(torch.isfinite(out).all())
    ok = bool(torch.allclose(ref, out, atol=1e-3, rtol=1e-3))
    verdict = "NOFIRE" if (sites == 0 and combo) else ("PASS" if ok else "FAIL")

    print(json.dumps({
        "mode": "combo", "combo": "+".join(combo) or "(none)",
        "sites": sites, "codehash": chash, "files": nfiles,
        "max_abs": max_abs, "max_rel": max_rel,
        "finite": finite, "verdict": verdict,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
