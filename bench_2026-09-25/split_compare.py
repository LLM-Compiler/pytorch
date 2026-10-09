"""Pair two split-arm runs into one result row: speedup + correctness.

Correctness is checked across processes by comparing the saved output tensors.
Both arms built the model from the same seed, so any difference beyond float
reduction-order noise is a real defect.
"""
import argparse, json, sys, os

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std-prefix", required=True)
    ap.add_argument("--numa-prefix", required=True)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    a = ap.parse_args()

    import torch
    s = json.load(open(a.std_prefix + ".timing.json"))
    n = json.load(open(a.numa_prefix + ".timing.json"))
    so = torch.load(a.std_prefix + ".out.pt", weights_only=True).float()
    no = torch.load(a.numa_prefix + ".out.pt", weights_only=True).float()

    if so.shape != no.shape:
        print(f"SHAPE MISMATCH {tuple(so.shape)} vs {tuple(no.shape)}", file=sys.stderr)
        return 2

    diff = (so - no).abs()
    max_abs = float(diff.max())
    denom = so.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max())
    finite = bool(torch.isfinite(no).all())
    ok = finite and torch.allclose(so, no, atol=a.atol, rtol=a.rtol)
    sc, nc = so.flatten() - so.mean(), no.flatten() - no.mean()
    corr = float((sc * nc).sum() / (sc.norm() * nc.norm()))

    speedup = s["mean_ms"] / n["mean_ms"] if n["mean_ms"] else float("nan")
    rec = {"model": s["model"], "batch": s["batch"], "seq": s["seq"],
           "world_size": s["world_size"], "speedup": speedup,
           "std_mean_ms": s["mean_ms"], "numa_mean_ms": n["mean_ms"],
           "correctness": "PASS" if ok else "FAIL",
           "max_abs": max_abs, "max_rel": max_rel, "corr": corr,
           "std_peak_rss_gb": s["peak_rss_gb"], "numa_peak_rss_gb": n["peak_rss_gb"]}
    print(json.dumps(rec), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
