#!/bin/bash
# Stage 2c -- crossover curve RE-MEASURED with the alias-prefix fix (0baf9dba34).
# Supersedes stage2b, which was taken before that fix and is pessimistic.
# GPT-2 117M, B=1, W=4. Detached-safe: progress markers + DONE sentinel.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch

OUT=bench_2026-09-25/stage2c
mkdir -p $OUT
PROG=$OUT/progress.log
CSV=$OUT/crossover_fixed.csv
GPT2=scripts/inductor/gpt2_multi_size_speedup_cli.py
H=12; B=1; W=4

say () { echo "$(date -Is) $*" | tee -a $PROG; }

[ -f $CSV ] || echo "seq,speedup,std_mean_ms,numa_mean_ms,correctness,numa_args,peak_rss_kb,status" > $CSV

say "stage2c start: GPT-2 117M B=1 W=4, alias-prefix fix IN, warmup 3 iters 10"

for S in 512 1024 1152 1280 1536 2048 3072 4096; do
  # Resumable: skip anything already recorded.
  if grep -q "^$S," $CSV 2>/dev/null; then say "  S=$S already recorded, skipping"; continue; fi

  GUARD=$(( B * H * S * S ))
  if [ $GUARD -gt 536870912 ]; then
    say "  SKIP S=$S: B*H*S^2=$GUARD over the 2GB SDPA-MATH limit"
    echo "$S,,,,skipped,,,guard-2GB" >> $CSV; continue
  fi

  log=$OUT/s${S}.log
  before=$(mktemp); ls -d /tmp/gpt2_speedup_inductor_cache_* 2>/dev/null | sort > $before
  say "  S=$S  free=$(free -g | awk '/^Mem:/{print $7}')GB"

  /usr/bin/time -v -o $log.time \
    python $GPT2 --model 117m --batch-size $B --seq-length $S --world-size $W \
      --warmup 3 --iters 10 --check-correctness > $log 2> $log.err
  rc=$?

  after=$(mktemp); ls -d /tmp/gpt2_speedup_inductor_cache_* 2>/dev/null | sort > $after
  newdirs=$(comm -13 $before $after)
  counts=""
  for d in $newdirs; do counts="$counts $(grep -rha 'numa_args=' $d 2>/dev/null | wc -l)"; done
  na=$(echo $counts | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1); [ -z "$na" ] && na=0

  sp=$(tail -1 $log | awk '{print $NF}')
  sm=$(grep -o 'standard mean_ms=[0-9.]*' $log.err | head -1 | cut -d= -f2)
  nm=$(grep -o 'numa_tp mean_ms=[0-9.]*' $log.err | head -1 | cut -d= -f2)
  co=$(grep -o 'PASS\|FAIL' $log.err $log 2>/dev/null | grep -o 'PASS\|FAIL' | head -1); [ -z "$co" ] && co="n/a"
  rss=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' $log.time 2>/dev/null | grep -o '[0-9]*$'); [ -z "$rss" ] && rss=0
  st="ok"; [ $rc -ne 0 ] && st="rc=$rc"

  echo "$S,$sp,$sm,$nm,$co,$na,$rss,$st" >> $CSV
  say "    -> speedup=$sp std=${sm}ms numa=${nm}ms corr=$co numa_args=$na rss=$((rss/1024))MB $st"
  for d in $newdirs; do rm -rf $d; done
  rm -f $before $after
done

say "=== stage2c vs stage2b (pre-fix) ==="
python - <<'PY' | tee -a $PROG
import csv, os
def load(p, k, v):
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            try: out[int(r[k])] = float(r[v])
            except Exception: pass
    return out
new = load("bench_2026-09-25/stage2c/crossover_fixed.csv", "seq", "speedup")
old = load("bench_2026-09-25/stage2b/crossover_results.csv", "seq", "speedup")
print("%-6s %-10s %-10s %s" % ("S", "pre-fix", "fixed", "delta"))
for S in sorted(new):
    o = old.get(S)
    print("%-6d %-10s %-10.3f %s" % (
        S, ("%.3f" % o) if o else "-", new[S],
        ("%+.3f" % (new[S] - o)) if o else "(new point)"))
xs = sorted(new)
cross = None
for a, b in zip(xs, xs[1:]):
    if new[a] < 1.0 <= new[b]:
        cross = a + (b - a) * (1.0 - new[a]) / (new[b] - new[a])
        break
print("\ncrossover ~ S=%s" % (("%.0f" % cross) if cross else "not bracketed"))
PY

say "STAGE2CDONE"
