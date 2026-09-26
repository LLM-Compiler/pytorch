#!/bin/bash
# Step 1 -- alias-prefix fix: verify correctness, then measure its cost.
# Fully detached; survives loss of the controlling session.
#
# Phase A  correctness matrix, W=4 only, fix ON. Must stay 12/12-equivalent
#          (here 6 NUMA cells + 2 standard references across 1 and 2 layers).
# Phase B  interleaved timing. Variants alternate WITHIN each repetition so
#          machine drift hits both equally -- S=2048 drifts ~6% run to run,
#          so a single before/after pair proves nothing.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch

OUT=bench_2026-09-25/stage1fix
mkdir -p $OUT
PROG=$OUT/progress.log
R=bench_2026-09-25
GPT2=scripts/inductor/gpt2_multi_size_speedup_cli.py

say () { echo "$(date -Is) $*" | tee -a $PROG; }

say "PHASE A: correctness matrix, W=4, NUMA_ALIAS_PREFIX_FIX=1"
rm -f /tmp/s1f_*.pt
for NL in 1 2; do
  PASSES=std NLAYERS=$NL WS=4 python $R/numa_check.py >> $OUT/phaseA.log 2>&1
  for P in mlp attn both; do
    NUMA_ALIAS_PREFIX_FIX=1 PASSES=$P NLAYERS=$NL WS=4 \
      python $R/numa_check.py >> $OUT/phaseA.log 2>&1 || say "  PHASE A CELL FAILED: $P L$NL"
  done
  say "  phase A layers=$NL done"
done

say "PHASE A: comparing"
: > $OUT/phaseA_compare.txt
for NL in 1 2; do
  for P in mlp attn both; do
    python $R/cmp.py /tmp/nc_std_w4_L${NL}_s128.pt /tmp/nc_${P}_w4_L${NL}_s128.pt \
      >> $OUT/phaseA_compare.txt 2>&1
  done
done
cat $OUT/phaseA_compare.txt | tee -a $PROG
NPASS=$(grep -c "corr=+1.00000" $OUT/phaseA_compare.txt)
say "PHASE A RESULT: $NPASS of 6 cells at corr=1.000000"
if [ "$NPASS" -ne 6 ]; then
  say "PHASE A FAILED -- stopping before timing. The fix changes numerics."
  say "STEP1DONE-FAILED"
  exit 1
fi

say "PHASE B: interleaved timing, GPT-2 117M B=1 W=4, warmup 3 iters 10"
CSV=$OUT/timing.csv
echo "rep,variant,seq,speedup,std_mean_ms,numa_mean_ms,correctness,status" > $CSV

time_cell () {
  local rep=$1 variant=$2 S=$3
  local tag="r${rep}_${variant}_s${S}"
  local log=$OUT/$tag.log
  local before=$(mktemp); ls -d /tmp/gpt2_speedup_inductor_cache_* 2>/dev/null | sort > $before

  local envvar=""
  [ "$variant" = "on" ] && envvar="1"

  say "  cell $tag"
  NUMA_ALIAS_PREFIX_FIX=$envvar \
    python $GPT2 --model 117m --batch-size 1 --seq-length $S --world-size 4 \
      --warmup 3 --iters 10 --check-correctness > $log 2> $log.err
  local rc=$?

  local after=$(mktemp); ls -d /tmp/gpt2_speedup_inductor_cache_* 2>/dev/null | sort > $after
  local newdirs=$(comm -13 $before $after)

  # Only scan the generated code once per (variant, S) -- it is identical
  # across repetitions and the scan is not free.
  if [ "$rep" = "1" ]; then
    python $R/reclone_scan.py $newdirs > $OUT/reclone_${variant}_s${S}.txt 2>&1
    say "    $(grep numa_ar_names $OUT/reclone_${variant}_s${S}.txt)"
  fi

  local sp=$(tail -1 $log | awk '{print $NF}')
  local sm=$(grep -o 'standard mean_ms=[0-9.]*' $log.err | head -1 | cut -d= -f2)
  local nm=$(grep -o 'numa_tp mean_ms=[0-9.]*' $log.err | head -1 | cut -d= -f2)
  local co=$(grep -o 'PASS\|FAIL' $log.err $log 2>/dev/null | grep -o 'PASS\|FAIL' | head -1)
  [ -z "$co" ] && co="n/a"
  local st="ok"; [ $rc -ne 0 ] && st="rc=$rc"
  echo "$rep,$variant,$S,$sp,$sm,$nm,$co,$st" >> $CSV
  say "    -> speedup=$sp std=${sm}ms numa=${nm}ms corr=$co $st"

  for d in $newdirs; do rm -rf $d; done
  rm -f $before $after
}

for rep in 1 2 3; do
  for S in 2048 4096; do
    time_cell $rep off $S
    time_cell $rep on  $S
  done
  say "  repetition $rep complete"
done

say "PHASE B complete"
python - <<'PY' | tee -a $PROG
import csv, statistics, collections
rows = list(csv.DictReader(open("bench_2026-09-25/stage1fix/timing.csv")))
by = collections.defaultdict(list)
for r in rows:
    if r["status"] != "ok":
        continue
    by[(r["variant"], r["seq"])].append(
        (float(r["speedup"]), float(r["std_mean_ms"]), float(r["numa_mean_ms"])))
print("\n=== Step 1 measurement: alias-prefix fix ===")
print("%-4s %-8s %-3s %-22s %-22s %s" % ("S", "variant", "n", "numa mean_ms", "std mean_ms", "speedup"))
for S in ("2048", "4096"):
    for v in ("off", "on"):
        vals = by.get((v, S), [])
        if not vals:
            continue
        sp = [x[0] for x in vals]; sm = [x[1] for x in vals]; nm = [x[2] for x in vals]
        f = lambda a: "%9.1f +/- %-7.1f" % (statistics.fmean(a),
                                            statistics.stdev(a) if len(a) > 1 else 0.0)
        print("%-4s %-8s %-3d %-22s %-22s %.4f" % (S, v, len(sp), f(nm), f(sm), statistics.fmean(sp)))
    a = by.get(("off", S)); b = by.get(("on", S))
    if a and b:
        na = statistics.fmean([x[2] for x in a]); nb = statistics.fmean([x[2] for x in b])
        print("     -> NUMA arm %+.2f%% (%.1f -> %.1f ms); speedup %.4f -> %.4f\n"
              % ((nb - na) / na * 100.0, na, nb,
                 statistics.fmean([x[0] for x in a]), statistics.fmean([x[0] for x in b])))
PY

say "STEP1DONE-OK"
