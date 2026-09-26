#!/bin/bash
# Stage 3 -- full W=4 sweep, self-contained and detached.
#
# Reads a plan file from stage3_plan.py. One subprocess per cell, so a crash,
# OOM or hang costs that cell and nothing else. After EVERY cell the result is
# committed and pushed, so progress is durable on GitHub in real time rather
# than only at the end.
#
#   setsid nohup env PLAN=/tmp/stage3_plan.txt bash bench_2026-09-25/stage3.sh \
#     > /tmp/stage3_outer.log 2>&1 < /dev/null &
#
# Resumable: a cell already in results.csv is skipped, so relaunching the same
# command after any interruption continues where it stopped.
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch

# The agent holds the deploy-key passphrase; without this the per-cell push
# cannot authenticate. BatchMode makes a missing agent fail fast instead of
# blocking the sweep on a prompt.
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/home/shavin/.ssh/agent.sock}
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
export GIT_TERMINAL_PROMPT=0
BRANCH=${BRANCH:-fix/numa-tp-row-parallel-allreduce}
GIT_ID=(-c user.name=ThisaraWeerakoon -c user.email=thisara.weerakoon2001@gmail.com)

PLAN=${PLAN:-/tmp/stage3_plan.txt}
OUT=${OUT:-bench_2026-09-25/stage3}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-50}
WORLD=${WORLD:-4}
MIN_FREE_GB=${MIN_FREE_GB:-0}      # 0 = attempt every cell, no deferral
mkdir -p $OUT/logs
PROG=$OUT/progress.log
CSV=$OUT/results.csv

say () { echo "$(date -Is) $*" | tee -a $PROG; }

publish () {   # $1 = commit subject
  git add -A $OUT >/dev/null 2>&1
  if git diff --cached --quiet; then return 0; fi
  if ! git "${GIT_ID[@]}" commit -q -m "$1" >/dev/null 2>&1; then
    say "    (commit failed, continuing)"; return 0
  fi
  if ! git push -q origin $BRANCH >/dev/null 2>&1; then
    say "    (push failed, commit is local; the next cell will carry it)"
  fi
}

[ -f "$PLAN" ] || { say "FATAL: no plan file at $PLAN"; exit 1; }
[ -f $CSV ] || echo "model,batch,seq,world_size,speedup,std_mean_ms,numa_mean_ms,correctness,numa_args_numa,numa_args_std,peak_rss_kb,wall_s,status" > $CSV

TOTAL=$(wc -l < "$PLAN")
say "stage3 start: $TOTAL planned cells, W=$WORLD, warmup $WARMUP iters $ITERS"
say "  memory ceiling DISABLED at user request -- every planned cell is attempted."
say "  A cell larger than RAM is expected to die; it is recorded and the sweep continues."
say "  Committing and pushing after every cell to origin/$BRANCH."
publish "bench(stage3): sweep started, $TOTAL cells"

n=0
while read -r MODEL CLI MARG B S EST; do
  n=$((n+1))
  [ -z "$MODEL" ] && continue

  if grep -q "^$MODEL,$B,$S,$WORLD," $CSV 2>/dev/null; then
    say "[$n/$TOTAL] $MODEL B=$B S=$S already recorded, skipping"; continue
  fi

  FREE=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$MIN_FREE_GB" -gt 0 ] && [ "$FREE" -lt "$MIN_FREE_GB" ]; then
    say "[$n/$TOTAL] $MODEL B=$B S=$S DEFERRED: only ${FREE}GB available"
    echo "$MODEL,$B,$S,$WORLD,,,,,,,,,deferred-low-memory" >> $CSV
    publish "bench(stage3): $MODEL B=$B S=$S deferred"; continue
  fi

  case $CLI in
    gpt2) SCRIPT=scripts/inductor/gpt2_multi_size_speedup_cli.py ;;
    bert) SCRIPT=scripts/inductor/bert_multi_size_speedup_cli.py ;;
    *)    say "[$n/$TOTAL] unknown cli $CLI"; continue ;;
  esac

  # 3x the projection, floor 30 min, cap 6 h. A cell that hits this is hung.
  TMO=$(( EST * 3 )); [ $TMO -lt 1800 ] && TMO=1800; [ $TMO -gt 21600 ] && TMO=21600

  TAG="${MODEL}_b${B}_s${S}_w${WORLD}"
  LOG=$OUT/logs/$TAG.log
  before=$(mktemp); ls -d /tmp/*_speedup_inductor_cache_* 2>/dev/null | sort > $before

  say "[$n/$TOTAL] $MODEL B=$B S=$S  free=${FREE}GB  est=$((EST/60))min  timeout=$((TMO/60))min"
  T0=$(date +%s)
  timeout -k 60 $TMO /usr/bin/time -v -o $LOG.time \
    python $SCRIPT --model $MARG --batch-size $B --seq-length $S \
      --world-size $WORLD --warmup $WARMUP --iters $ITERS --check-correctness \
      > $LOG 2> $LOG.err
  rc=$?
  T1=$(date +%s); WALL=$((T1-T0))

  after=$(mktemp); ls -d /tmp/*_speedup_inductor_cache_* 2>/dev/null | sort > $after
  newdirs=$(comm -13 $before $after)
  counts=""
  for d in $newdirs; do counts="$counts $(grep -rha 'numa_args=' $d 2>/dev/null | wc -l)"; done
  na=$(echo $counts | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1)
  ns=$(echo $counts | tr ' ' '\n' | grep -v '^$' | sort -n | head -1)
  [ -z "$na" ] && na=0; [ -z "$ns" ] && ns=0

  sp=$(tail -1 $LOG | awk '{print $NF}')
  sm=$(grep -o 'standard mean_ms=[0-9.]*' $LOG.err | head -1 | cut -d= -f2)
  nm=$(grep -o 'numa_tp mean_ms=[0-9.]*' $LOG.err | head -1 | cut -d= -f2)
  co=$(grep -o 'PASS\|FAIL' $LOG.err $LOG 2>/dev/null | grep -o 'PASS\|FAIL' | head -1); [ -z "$co" ] && co="n/a"
  rss=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' $LOG.time 2>/dev/null | grep -o '[0-9]*$'); [ -z "$rss" ] && rss=0

  case $rc in
    0)       st="ok" ;;
    124|137) st="TIMEOUT-OR-KILLED" ;;
    139)     st="SEGFAULT" ;;
    *)       st="rc=$rc" ;;
  esac
  if [ $rc -ne 0 ] && grep -qi "out of memory\|MemoryError\|Killed" $LOG.err 2>/dev/null; then st="OOM"; fi
  [ "$co" = "FAIL" ] && st="CORRECTNESS-FAIL"

  echo "$MODEL,$B,$S,$WORLD,$sp,$sm,$nm,$co,$na,$ns,$rss,$WALL,$st" >> $CSV
  say "    -> speedup=$sp std=${sm}ms numa=${nm}ms corr=$co numa_args=$na rss=$((rss/1024))MB wall=$((WALL/60))min $st"
  [ "$st" != "ok" ] && say "    !! $TAG ended $st -- see $LOG.err"

  publish "bench(stage3): [$n/$TOTAL] $MODEL B=$B S=$S -> ${sp:-n/a} ($st)"

  for d in $newdirs; do rm -rf $d; done
  rm -f $before $after
done < "$PLAN"

say "=== summary ==="
python - "$CSV" <<'PY' | tee -a $PROG $OUT/SUMMARY.txt
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
ok = [r for r in rows if r["status"] == "ok"]
bad = [r for r in rows if r["status"] != "ok"]
print("cells recorded: %d   ok: %d   not ok: %d" % (len(rows), len(ok), len(bad)))
for r in bad:
    print("   %-11s B=%-2s S=%-5s %s" % (r["model"], r["batch"], r["seq"], r["status"]))
bad_int = [r for r in ok if r["numa_args_numa"] in ("0", "") or r["numa_args_std"] not in ("0",)]
if bad_int:
    print("\nINTEGRITY PROBLEMS (numa arm must be >0, standard arm must be 0):")
    for r in bad_int:
        print("   %-11s B=%-2s S=%-5s numa=%s std=%s"
              % (r["model"], r["batch"], r["seq"], r["numa_args_numa"], r["numa_args_std"]))
else:
    print("\nintegrity: all ok cells have numa_args>0 in the NUMA arm and 0 in the standard arm")
by = collections.defaultdict(list)
for r in ok:
    try: by[r["model"]].append((int(r["batch"]), int(r["seq"]), float(r["speedup"])))
    except Exception: pass
print("\nspeedup by model (min / median / max over its ok cells):")
for m in sorted(by):
    v = sorted(x[2] for x in by[m])
    print("   %-11s n=%-3d  %.3f / %.3f / %.3f   above 1.0x: %d of %d"
          % (m, len(v), v[0], v[len(v)//2], v[-1], sum(1 for x in v if x > 1.0), len(v)))
print("\nbest cell per model:")
for m in sorted(by):
    B, S, sp = max(by[m], key=lambda x: x[2])
    print("   %-11s B=%-2d S=%-5d %.3fx" % (m, B, S, sp))
PY

publish "bench(stage3): sweep complete, summary"
say "STAGE3DONE"
