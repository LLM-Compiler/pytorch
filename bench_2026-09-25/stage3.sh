#!/bin/bash
# Stage 3 -- full W=4 sweep. Reads a plan file from stage3_plan.py (which
# applies the segfault guard and the memory ceiling) and runs one cell per
# subprocess so a crash, OOM or hang takes down only that cell.
#
#   PLAN=/tmp/stage3_plan.txt  OUT=bench_2026-09-25/stage3  bash stage3.sh
#
# Resumable: a cell already in results.csv is skipped, so a killed run can be
# relaunched with the same command and picks up where it stopped.
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch

PLAN=${PLAN:-/tmp/stage3_plan.txt}
OUT=${OUT:-bench_2026-09-25/stage3}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-50}
WORLD=${WORLD:-4}
MIN_FREE_GB=${MIN_FREE_GB:-6}      # refuse to start a cell below this
mkdir -p $OUT/logs
PROG=$OUT/progress.log
CSV=$OUT/results.csv

say () { echo "$(date -Is) $*" | tee -a $PROG; }

[ -f "$PLAN" ] || { say "FATAL: no plan file at $PLAN"; exit 1; }
[ -f $CSV ] || echo "model,batch,seq,world_size,speedup,std_mean_ms,numa_mean_ms,correctness,numa_args_numa,numa_args_std,peak_rss_kb,wall_s,status" > $CSV

TOTAL=$(wc -l < "$PLAN")
say "stage3 start: $TOTAL planned cells, W=$WORLD, warmup $WARMUP iters $ITERS"
say "  resumable -- relaunch the same command to continue after any interruption"

n=0
while read -r MODEL CLI MARG B S EST; do
  n=$((n+1))
  [ -z "$MODEL" ] && continue

  if grep -q "^$MODEL,$B,$S,$WORLD," $CSV 2>/dev/null; then
    say "[$n/$TOTAL] $MODEL B=$B S=$S already recorded, skipping"; continue
  fi

  FREE=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$FREE" -lt "$MIN_FREE_GB" ]; then
    say "[$n/$TOTAL] $MODEL B=$B S=$S DEFERRED: only ${FREE}GB available"
    echo "$MODEL,$B,$S,$WORLD,,,,,,,,,deferred-low-memory" >> $CSV; continue
  fi

  case $CLI in
    gpt2) SCRIPT=scripts/inductor/gpt2_multi_size_speedup_cli.py ;;
    bert) SCRIPT=scripts/inductor/bert_multi_size_speedup_cli.py ;;
    *)    say "[$n/$TOTAL] unknown cli $CLI"; continue ;;
  esac

  # Generous ceiling: 3x the projection, floor 30 min, cap 6 h. A cell that
  # hits this is hung, not slow.
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
    0)   st="ok" ;;
    124|137) st="TIMEOUT" ;;
    139) st="SEGFAULT" ;;
    *)   st="rc=$rc" ;;
  esac
  # A correctness FAIL invalidates the timing; record it as such.
  [ "$co" = "FAIL" ] && st="CORRECTNESS-FAIL"

  echo "$MODEL,$B,$S,$WORLD,$sp,$sm,$nm,$co,$na,$ns,$rss,$WALL,$st" >> $CSV
  say "    -> speedup=$sp std=${sm}ms numa=${nm}ms corr=$co numa_args=$na rss=$((rss/1024))MB wall=$((WALL/60))min $st"
  [ "$st" != "ok" ] && say "    !! $TAG ended $st -- see $LOG.err"

  for d in $newdirs; do rm -rf $d; done
  rm -f $before $after
done < "$PLAN"

say "=== summary ==="
python - "$CSV" <<'PY' | tee -a $PROG
import csv, sys, collections
rows = [r for r in csv.DictReader(open(sys.argv[1]))]
ok = [r for r in rows if r["status"] == "ok"]
bad = [r for r in rows if r["status"] not in ("ok",)]
print("cells recorded: %d   ok: %d   not ok: %d" % (len(rows), len(ok), len(bad)))
for r in bad:
    print("   %-11s B=%-2s S=%-5s %s" % (r["model"], r["batch"], r["seq"], r["status"]))
bad_int = [r for r in ok if r["numa_args_numa"] in ("0", "") or r["numa_args_std"] not in ("0",)]
if bad_int:
    print("\nINTEGRITY PROBLEMS (numa arm must be >0, standard arm must be 0):")
    for r in bad_int:
        print("   %-11s B=%-2s S=%-5s numa=%s std=%s"
              % (r["model"], r["batch"], r["seq"], r["numa_args_numa"], r["numa_args_std"]))
by = collections.defaultdict(list)
for r in ok:
    try: by[r["model"]].append((int(r["seq"]), int(r["batch"]), float(r["speedup"])))
    except Exception: pass
print("\nspeedup by model (min / median / max over its ok cells):")
for m in sorted(by):
    v = sorted(x[2] for x in by[m])
    print("   %-11s n=%-3d  %.3f / %.3f / %.3f" % (m, len(v), v[0], v[len(v)//2], v[-1]))
    wins = [x for x in by[m] if x[2] > 1.0]
    print("       above 1.0x: %d of %d" % (len(wins), len(v)))
PY

say "STAGE3DONE"
