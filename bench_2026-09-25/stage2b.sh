#!/bin/bash
# Stage 2b -- sequence-length crossover sweep, GPT-2 117M, W=4.
# One cell per subprocess so a segfault or OOM in one cell cannot take the
# sweep down. Speedup on stdout, absolute means on stderr.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch

OUT=bench_2026-09-25/stage2b
mkdir -p $OUT
CSV=$OUT/crossover_results.csv
echo "cell,model,batch,seq,world_size,speedup,std_mean_ms,numa_mean_ms,correctness,numa_args_numa,numa_args_std,peak_rss_kb,status" > $CSV

GPT2=scripts/inductor/gpt2_multi_size_speedup_cli.py
WARMUP=3
ITERS=10
W=4

# 2 GB SDPA-MATH guard: skip any config with B*H*S^2 > 536870912.
# GPT-2 117M has H=12, so at B=1 the limit is S <= 6689. All cells below pass.
H=12
B=1

for S in 512 1024 1280 1536 2048 3072 4096; do
  LIMIT=$(( B * H * S * S ))
  if [ $LIMIT -gt 536870912 ]; then
    echo "SKIP S=$S (B*H*S^2=$LIMIT exceeds the 2GB SDPA-MATH limit)" | tee -a $OUT/progress.log
    echo "c_s${S}_w$W,gpt2-117m,$B,$S,$W,,,,skipped-2GB-guard,,,,skip" >> $CSV
    continue
  fi

  cell="c_s${S}_w$W"
  log="$OUT/${cell}.log"
  before=$(mktemp); ls -d /tmp/gpt2_speedup_inductor_cache_* 2>/dev/null | sort > "$before"

  echo "===== $cell  S=$S W=$W  $(date -Is)  free=$(free -g | awk '/^Mem:/{print $7}')GB" | tee -a $OUT/progress.log

  /usr/bin/time -v -o "$log.time" \
    python $GPT2 --model 117m --batch-size $B --seq-length $S --world-size $W \
      --warmup $WARMUP --iters $ITERS --check-correctness > "$log" 2> "$log.err"
  rc=$?

  after=$(mktemp); ls -d /tmp/gpt2_speedup_inductor_cache_* 2>/dev/null | sort > "$after"
  newdirs=$(comm -13 "$before" "$after")

  counts=""
  for d in $newdirs; do
    counts="$counts $(grep -rha 'numa_args=' "$d" 2>/dev/null | wc -l)"
  done
  na_numa=$(echo $counts | tr ' ' '\n' | grep -v '^$' | sort -n | tail -1)
  na_std=$(echo $counts | tr ' ' '\n' | grep -v '^$' | sort -n | head -1)
  [ -z "$na_numa" ] && na_numa=0
  [ -z "$na_std" ] && na_std=0

  speedup=$(tail -1 "$log" | awk '{print $NF}')
  stdm=$(grep -o 'standard mean_ms=[0-9.]*' "$log.err" | head -1 | cut -d= -f2)
  numam=$(grep -o 'numa_tp mean_ms=[0-9.]*' "$log.err" | head -1 | cut -d= -f2)
  corr=$(grep -o 'PASS\|FAIL' "$log.err" "$log" 2>/dev/null | grep -o 'PASS\|FAIL' | head -1)
  [ -z "$corr" ] && corr="n/a"
  rss=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' "$log.time" 2>/dev/null | grep -o '[0-9]*$')
  [ -z "$rss" ] && rss=0
  status="ok"; [ $rc -ne 0 ] && status="rc=$rc"

  echo "$cell,gpt2-117m,$B,$S,$W,$speedup,$stdm,$numam,$corr,$na_numa,$na_std,$rss,$status" >> $CSV
  echo "  -> S=$S speedup=$speedup std=${stdm}ms numa=${numam}ms corr=$corr numa_args=$na_numa rss=$((rss/1024))MB $status" | tee -a $OUT/progress.log

  for d in $newdirs; do rm -rf "$d"; done
  rm -f "$before" "$after"
done

echo "STAGE2BDONE $(date -Is)" | tee -a $OUT/progress.log
