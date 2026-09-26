#!/usr/bin/env python3
"""Build the Stage 3 cell list: apply both guards, project cost, emit a plan file.

Two independent limits gate a cell, and the drivers enforce neither:

  segfault guard  B*H*S^2 <= 5.37e8   32-bit byte-offset overflow in the
                                      Triton-CPU BMM gather lowering when an
                                      SDPA-MATH intermediate reaches 2 GB
                                      (BUG.md). Hard crash, not an error.
  memory ceiling  B*H*S^2 <= 3.0e8    attention intermediates are replicated
                                      PER SOCKET, so peak RSS is ~4x what a
                                      single-socket run would use. Fitted on
                                      hpgen9-2 in stage2b:
                                        RSS_MB ~= 4700 + 1.49e-4 * B*H*S^2
                                      3.0e8 keeps a W=4 cell under ~50 GB of
                                      the machine's 62 GB.

The memory ceiling is the tighter of the two by ~1.8x.

Cost projection is ROUGH -- a two-term fit (GEMM linear in S, attention
quadratic in S) calibrated on measured W=4 cells, then scaled across models by
L*D^2 for the linear term and L*D for the quadratic one. Measured residuals on
the calibration points run 8-14% low. Treat totals as a lower bound.
"""
import argparse, sys

# name -> (cli, --model arg, layers, d_model, heads)
MODELS = {
    "gpt2-117m":  ("gpt2", "117m", 12,  768, 12),
    "gpt2-345m":  ("gpt2", "345m", 24, 1024, 16),
    "gpt2-762m":  ("gpt2", "762m", 36, 1280, 20),
    "bert-base":  ("bert", "base", 12,  768, 12),
    "bert-large": ("bert", "large",24, 1024, 16),
}

SEGFAULT_GUARD = 536_870_912      # B*H*S^2, the 2 GB SDPA-MATH limit
MEMORY_CEILING = 300_000_000      # B*H*S^2, ~50 GB of 62 GB at W=4
RSS_CONST_MB, RSS_SLOPE_MB = 4700.0, 1.49e-4

# Calibration: GPT-2 117M, B=1, W=4, standard arm, ms per forward.
A_LIN, A_QUAD = 0.4464, 6.827e-4
REF_LD2, REF_LD = 12 * 768 ** 2, 12 * 768


def fwd_ms(L, D, H, B, S):
    lin = A_LIN * S * (L * D * D) / REF_LD2
    quad = A_QUAD * S * S * (L * D) / REF_LD
    return B * (lin + quad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--seqs", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--tier-base", type=int, default=8192, help="keep S <= tier_base / B")
    p.add_argument("--mem-ceiling", type=int, default=MEMORY_CEILING)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", default=None, help="write the plan file here")
    a = p.parse_args()

    n_fwd = (a.warmup + a.iters) * 2          # both arms
    rows, total_s, skipped = [], 0.0, []

    for m in a.models:
        if m not in MODELS:
            sys.exit("unknown model %r; known: %s" % (m, ", ".join(MODELS)))
        cli, marg, L, D, H = MODELS[m]
        for B in sorted(a.batches):
            for S in sorted(a.seqs):
                if S * B > a.tier_base:
                    continue                   # tiering, not a guard
                load = B * H * S * S
                if load > SEGFAULT_GUARD:
                    skipped.append((m, B, S, load, "segfault-guard")); continue
                if load > a.mem_ceiling:
                    skipped.append((m, B, S, load, "memory-ceiling")); continue
                t = fwd_ms(L, D, H, B, S)
                compile_s = 200.0 * L / 12.0 * 2      # two cold compiles
                cell_s = compile_s + n_fwd * t / 1000.0
                rss_mb = RSS_CONST_MB + RSS_SLOPE_MB * load
                rows.append((m, cli, marg, B, S, load, rss_mb, cell_s))
                total_s += cell_s

    w = open(a.out, "w") if a.out else None
    print("%-11s %-3s %-5s %-12s %-9s %s" % ("model", "B", "S", "B*H*S^2", "proj RSS", "proj cell"))
    for m, cli, marg, B, S, load, rss, cs in rows:
        print("%-11s %-3d %-5d %-12d %6.1f GB  %5.1f min" % (m, B, S, load, rss / 1024.0, cs / 60.0))
        if w:
            w.write("%s %s %s %d %d %d\n" % (m, cli, marg, B, S, int(cs)))
    if w:
        w.close()

    print("\n%d cells, %d skipped" % (len(rows), len(skipped)))
    for m, B, S, load, why in skipped:
        print("   SKIP %-11s B=%-2d S=%-5d B*H*S^2=%-11d %s" % (m, B, S, load, why))
    print("\nprojected total: %.1f h  (rough, and a lower bound -- see docstring)" % (total_s / 3600.0))
    print("largest projected cell: %.1f min" % (max(r[7] for r in rows) / 60.0 if rows else 0))
    print("peak projected RSS: %.1f GB" % (max(r[6] for r in rows) / 1024.0 if rows else 0))
    if a.out:
        print("plan written to %s" % a.out)


if __name__ == "__main__":
    main()
