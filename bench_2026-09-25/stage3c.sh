#!/bin/bash
# Stage 3c -- the 10 cells still unrecovered after stage3b, retried with
# NUMA_COLPAR_VIEW=1 (column-parallel operands become views of the original
# weight instead of per-node full clones).
#
# Measured on gpt2-762m B=1 S=2048 W=4: 48.9 -> 43.5 GB (-5.4 GB) for a 0.6%
# slowdown, which is inside run-to-run noise. Correctness bit-identical.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25
OUT=$R/stage3c
mkdir -p $OUT
say () { echo "$(date -Is) $*" | tee -a $OUT/progress.log; }

export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/home/shavin/.ssh/agent.sock}
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
export NUMA_COLPAR_VIEW=1
BRANCH=fix/numa-tp-row-parallel-allreduce
GIT_ID=(-c user.name=ThisaraWeerakoon -c user.email=thisara.weerakoon2001@gmail.com)
publish () {
  git add -A $OUT >/dev/null 2>&1
  git diff --cached --quiet && return 0
  git "${GIT_ID[@]}" commit -q -m "$1" >/dev/null 2>&1 || { say "    (commit failed)"; return 0; }
  git push -q origin $BRANCH >/dev/null 2>&1 || say "    (push failed, commit is local)"
}

CELLS="
gpt2-762m 2 2048
gpt2-762m 8 1024
gpt2-345m 4 2048
bert-large 4 2048
gpt2-345m 1 4096
bert-large 1 4096
gpt2-762m 1 4096
gpt2-762m 4 2048
gpt2-345m 2 4096
bert-large 2 4096
"
TOTAL=$(echo "$CELLS" | grep -c '[a-z]')
say "stage3c start: $TOTAL cells, split-process + NUMA_COLPAR_VIEW=1"
publish "bench(stage3c): colpar-view retry started, $TOTAL cells"

n=0
echo "$CELLS" | while read -r MODEL B S; do
  [ -z "$MODEL" ] && continue
  n=$((n+1))
  if grep -q "^$MODEL,$B,$S,4," $OUT/results.csv 2>/dev/null; then
    say "[$n/$TOTAL] $MODEL B=$B S=$S already recorded, skipping"; continue
  fi
  say "[$n/$TOTAL] $MODEL B=$B S=$S  free=$(free -g | awk '/^Mem:/{print $7}')GB"
  MODEL=$MODEL B=$B S=$S W=4 WARMUP=10 ITERS=50 OUT=$OUT \
    timeout -k 60 21600 bash $R/split_cell.sh >> $OUT/cell_driver.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && say "    cell driver rc=$rc -- continuing"
  publish "bench(stage3c): [$n/$TOTAL] $MODEL B=$B S=$S"
done

say "=== stage3c summary ==="
python - "$OUT/results.csv" <<'PY' | tee -a $OUT/progress.log
import csv,sys,os
if not os.path.exists(sys.argv[1]):
    print("no results"); raise SystemExit
rows=list(csv.DictReader(open(sys.argv[1])))
ok=[r for r in rows if r.get("status")=="ok"]
print(f"recovered: {len(ok)} of {len(rows)} attempted")
for r in rows:
    sp=r.get("speedup") or ""
    try: sp=f"{float(sp):.3f}x"
    except Exception: sp="-"
    print(f"  {r['model']:<11} B={r['batch']:<2} S={r['seq']:<5} {sp:<9} "
          f"{r.get('correctness') or '-':<5} rss={r.get('numa_rss_gb') or '-'}")
PY
publish "bench(stage3c): colpar-view retry complete"
say "STAGE3CDONE"
