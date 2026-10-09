#!/bin/bash
# Gemma 270M speedup sweep, W=4, now that NUMA-TP is numerically correct.
# The stage2 Gemma numbers (0.762x / 0.820x) were measured on WRONG output and
# are void; these supersede them.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch
OUT=bench_2026-09-25/gemma_speed
mkdir -p $OUT
CSV=$OUT/results.csv
PROG=$OUT/progress.log
say () { echo "$(date -Is) $*" | tee -a $PROG; }
[ -f $CSV ] || echo "model,batch,seq,world_size,std_mean_ms,numa_mean_ms,speedup,status" > $CSV

GEMMA=scripts/inductor/simple_gemma_decoder_block_speedup_cli.py
B=1; W=4
say "gemma 270M speedup sweep: L=18 h=1152 heads=4 kv=1 hd=256 mlp=6912, W=$W"

for S in 512 1024 2048 4096; do
  if grep -q "^gemma-270m,$B,$S,$W," $CSV 2>/dev/null; then say "S=$S done, skip"; continue; fi
  say "  S=$S  free=$(free -g | awk '/^Mem:/{print $7}')GB"
  timeout -k 60 7200 python $GEMMA --batch-size $B --seq-length $S --num-layers 18 \
      --h-in 1152 --num-heads 4 --num-kv-heads 1 --head-dim 256 \
      --mlp-hidden 6912 --num-tp-shards $W --warmup 3 --iters 10 --detailed \
      > $OUT/s${S}.log 2> $OUT/s${S}.err
  rc=$?
  line=$(tail -1 $OUT/s${S}.log)
  nf=$(echo "$line" | awk '{print NF}')
  if [ $rc -eq 0 ] && [ "$nf" = "3" ]; then
    sm=$(echo "$line" | awk '{print $1}'); nm=$(echo "$line" | awk '{print $2}'); sp=$(echo "$line" | awk '{print $3}')
    echo "gemma-270m,$B,$S,$W,$sm,$nm,$sp,ok" >> $CSV
    say "    -> std=${sm}ms numa=${nm}ms speedup=${sp}x"
  else
    echo "gemma-270m,$B,$S,$W,,,,rc=$rc" >> $CSV
    say "    -> FAILED rc=$rc"
  fi
done
say "=== gemma speedup summary ==="
cat $CSV | tee -a $PROG
say "GEMMASPEEDDONE"
