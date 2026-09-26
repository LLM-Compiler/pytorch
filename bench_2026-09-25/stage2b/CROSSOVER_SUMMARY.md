# Stage 2b -- sequence-length crossover, GPT-2 117M, W=4

Host hpgen9-2, 4 NUMA nodes. --warmup 3 --iters 10, --check-correctness on
every cell. Baseline is the harness baseline (TRITON GEMM backend), not stock
torch.compile. Run 2026-09-26 19:28-20:04 IST, 36 min. B=1 throughout.

| S    | speedup | std ms   | numa ms | std ms/tok | numa ms/tok | peak RSS | corr |
|------|---------|----------|---------|------------|-------------|----------|------|
|  512 | 0.605x  |   584.7  |   966.6 |   1.14     |   1.89      |  3.3 GB  | PASS |
| 1024 | 0.700x  |  1173.3  |  1676.3 |   1.15     |   1.64      |  4.6 GB  | PASS |
| 1280 | 0.935x  |  1853.2  |  1981.0 |   1.45     |   1.55      |  6.7 GB  | PASS |
| 1536 | 1.050x  |  2614.6  |  2490.0 |   1.70     |   1.62      |  8.2 GB  | PASS |
| 2048 | 1.232x  |  4159.1  |  3377.1 |   2.03     |   1.65      | 11.8 GB  | PASS |
| 3072 | 1.347x  |  7542.8  |  5599.6 |   2.46     |   1.82      | 21.5 GB  | PASS |
| 4096 | 1.679x  | 13281.8  |  7912.8 |   3.24     |   1.93      | 34.6 GB  | PASS |

NUMA arm 338 numa_args sites in every cell; standard arm 0. Monotone in S,
no inversions.

## Crossover

Between S=1280 (0.935x) and S=1536 (1.050x). Linear interpolation puts it at
S ~= 1400. Every configuration at or above S=1536 wins; everything at or
below S=1280 loses.

## Mechanism: it is a scaling difference, not a constant offset

The two right-hand columns are the point. The BASELINE's per-token cost grows
2.8x across the sweep (1.14 -> 3.24 ms/token) as the quadratic attention term
takes over. The NUMA arm's per-token cost is nearly FLAT (1.89 -> 1.93, with a
dip to 1.55 mid-range). NUMA-TP is not winning by being faster per unit work;
it is winning because it scales far better in S.

That reframes the "no speedup" result. The reconciliation overhead (W-1
full-tensor add_ passes, W re-clones, W accumulator clones per reduction
point, 96 reduction points per forward at 12 layers x W=4) is close to
constant per token. At short S it dominates. At long S the baseline's
quadratic term dominates instead and the overhead is amortised away.

## Reproducibility

S=1024: 0.6999 here vs 0.6990 in Stage 2 -> 0.13% apart.
S=2048: 1.2316 here vs 1.3081 in Stage 2 -> 5.9% apart.

The short cells are very stable; the S=2048 cell is not. Stage 3's
--warmup 10 --iters 50 should tighten this. Do not quote a Stage 2/2b
speedup to 3 significant figures.

## NEW CONSTRAINT FOR STAGE 3: a memory ceiling, tighter than the 2GB guard

Peak RSS is dominated by SDPA-MATH attention intermediates REPLICATED PER
SOCKET. Fitting the three largest cells:

    RSS_MB  ~=  4700  +  1.49e-4 * (B * H * S^2)

The slope works out to ~37 floats per attention element, i.e. roughly 9
(B,H,S,S)-sized tensors live at once x 4 sockets. Extrapolating on a 62 GB
machine and leaving headroom (<= 50 GB):

    B * H * S^2  <~  3.0e8      (memory ceiling, W=4)
    B * H * S^2  <=  5.37e8     (the existing 2GB SDPA-MATH segfault guard)

The memory ceiling BITES FIRST -- it is ~1.8x tighter. Notably B=2, S=4096 on
GPT-2 117M (B*H*S^2 = 4.03e8) PASSES the 2GB guard but projects to ~64 GB and
would OOM. The sweep drivers guard neither. Stage 3 must pre-filter on both.

Caveat: fitted from 3 points, one model, W=4 only. Confirm before relying on
it; the constant term will differ per model and the slope scales with W.

## Where this leaves the paper comparison

At W=4, S=4096 the speedup is 1.68x. The paper reports 2.63x (Haswell) and
1.87x (Cascade Lake), both DUAL-socket. So the effect is real and, at the
long-sequence end, within striking distance of the paper's Cascade Lake
number -- but it is a 4-socket result, and W=2 on this machine still shows
no gain at S=2048 (0.97x, Stage 2). Whether W=2 crosses 1.0 at S=3072/4096
is UNTESTED and is the most direct paper-reproduction question outstanding.
