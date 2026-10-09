#!/bin/bash
# Stage 3b -- re-run the 15 cells that failed in stage3, with the two arms in
# SEPARATE processes. Confirmed to recover gpt2-762m B=1 S=2048 (previously a
# hard failure, now 1.591x PASS).
#
# Ordered cheapest-first so the recoverable cells land early; the four known
# oversized cells (>=59 GB standalone) are last and may still die -- that is
# expected and recorded, not a bug.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25
OUT=bench_2026-09-25/stage3b
mkdir -p $OUT
PROG=$OUT/progress.log
say () { echo "$(date -Is) $*" | tee -a $PROG; }

export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/home/shavin/.ssh/agent.sock}
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
BRANCH=fix/numa-tp-row-parallel-allreduce
GIT_ID=(-c user.name=ThisaraWeerakoon -c user.email=thisara.weerakoon2001@gmail.com)
publish () {
  git add -A $OUT >/dev/null 2>&1
  git diff --cached --quiet && return 0
  git "${GIT_ID[@]}" commit -q -m "$1" >/dev/null 2>&1 || { say "    (commit failed)"; return 0; }
  git push -q origin $BRANCH >/dev/null 2>&1 || say "    (push failed, commit is local)"
}

# model B S  -- previously-failed cells, smallest footprint first
CELLS="
gpt2-762m 4 1024
gpt2-345m 2 2048
bert-base 2 4096
gpt2-117m 2 4096
gpt2-762m 2 2048
gpt2-762m 8 1024
gpt2-345m 1 4096
gpt2-345m 4 2048
bert-large 1 4096
bert-large 4 2048
gpt2-762m 1 4096
gpt2-762m 4 2048
gpt2-345m 2 4096
bert-large 2 4096
"
TOTAL=$(echo "$CELLS" | grep -c '[a-z]')
say "stage3b start: $TOTAL cells, split-process, warmup 10 iters 50"
publish "bench(stage3b): split-process re-run started, $TOTAL cells"

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
  [ $rc -ne 0 ] && say "    cell driver rc=$rc (arm failed or timed out) -- continuing"
  publish "bench(stage3b): [$n/$TOTAL] $MODEL B=$B S=$S"
done

say "=== stage3b summary ==="
python - "$OUT/results.csv" <<'PY' | tee -a $PROG
import csv,sys,os
if not os.path.exists(sys.argv[1]):
    print("no results"); raise SystemExit
rows=list(csv.DictReader(open(sys.argv[1])))
ok=[r for r in rows if r.get("status")=="ok"]
print(f"recovered: {len(ok)} of {len(rows)} attempted")
for r in rows:
    sp=r.get("speedup","") or "-"
    try: sp=f"{float(sp):.3f}x"
    except Exception: sp="-"
    print(f"  {r['model']:<11} B={r['batch']:<2} S={r['seq']:<5} {sp:<9} {r.get('correctness','-'):<5} {r.get('status','-')}")
PY
publish "bench(stage3b): split-process re-run complete"
say "STAGE3BDONE"
