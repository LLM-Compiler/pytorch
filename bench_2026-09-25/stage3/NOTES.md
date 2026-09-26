# Stage 3 sweep -- running notes

Launched 2026-09-26 23:32 IST. Expected to finish around 2026-09-29.

## What this is

Full W=4 prefill sweep, standard `torch.compile` (TRITON GEMM backend) vs
NUMA-TP, on hpgen9-2 (4 sockets / 4 NUMA nodes, 80 logical CPUs, 62 GB).

104 cells. Models gpt2 117m/345m/762m and bert base/large, B in {1,2,4,8},
S in {128..4096}, tiered so S <= 8192/B. `--warmup 10 --iters 50`, and
`--check-correctness` on every cell.

Gemma is excluded: it is still numerically wrong (corr 0.50-0.77, two
defects distinct from the six fixed on this branch). See
`bench_2026-09-25/gemma_check.py` and STATE.txt.

## Deliberate choices

- **Memory ceiling disabled at user request.** Only the 2 GB SDPA-MATH
  segfault guard remains (`B*H*S^2 <= 5.37e8`), because that is a hard crash
  in the Triton-CPU BMM gather lowering, not a resource limit. One cell is
  skipped by it: gpt2-762m B=2 S=4096.
- Consequently a handful of cells project above the machine's RAM --
  gpt2-345m B=2 S=4096 projects ~83 GB, gpt2-117m/bert-base B=2 S=4096
  ~65 GB. These are expected to die. Each is recorded with status OOM or
  TIMEOUT-OR-KILLED and the sweep continues to the next cell.
- Every cell runs in its own subprocess, so a crash costs one cell.
- Committed and pushed after every cell, so nothing already finished is lost
  to an interruption.

## Reading the results

`results.csv` columns worth watching:

- `correctness` must be PASS. A FAIL sets `status=CORRECTNESS-FAIL` and the
  timing for that cell is meaningless.
- `numa_args_numa` must be > 0 and `numa_args_std` must be 0. This is the
  integrity check that catches a vacuous pass -- both arms silently running
  the same standard kernels. The repo's own `--profile` output reports
  "numa_args_calls=0 ... TP did NOT fire" on *every* run; that warning is
  false and must be ignored (it scans the wrong directory).
- `speedup` is `std_mean_ms / numa_mean_ms`. Above the world size (4) would
  mean the baseline is being penalised rather than NUMA-TP winning; flag it
  rather than celebrate it.

Expect cells below S ~= 1134 to lose. That is the measured crossover on this
machine (`bench_2026-09-25/stage2c/`), not a defect: reconciliation overhead
is roughly constant per token, so it dominates until the baseline's quadratic
attention term takes over.

## If it stops

Relaunch exactly the same command. Cells already in `results.csv` are
skipped.

    cd /home/shavin/llm-cpu-compiler/pytorch
    setsid nohup env PLAN=/tmp/stage3_plan.txt bash bench_2026-09-25/stage3.sh \
      > /tmp/stage3_outer.log 2>&1 < /dev/null &

If `/tmp/stage3_plan.txt` is gone (a reboot clears /tmp), regenerate it:

    python bench_2026-09-25/stage3_plan.py --mem-ceiling 99999999999 \
      --out /tmp/stage3_plan.txt

**A reboot also kills the ssh-agent**, which holds the deploy-key passphrase.
Without it the per-cell push fails (the commits still land locally, and a
later successful push carries them). Restart it with:

    eval $(ssh-agent -a /home/shavin/.ssh/agent.sock)
    ssh-add ~/.ssh/pytorch_private_key
