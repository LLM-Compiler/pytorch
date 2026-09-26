#!/bin/bash
# Stage 2 -- speedup spot-check. One cell per subprocess so a segfault in one
# cell cannot take the sweep down. Absolute means come from stderr, the
# speedup ratio from stdout.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch

OUT=bench_2026-09-25/stage2
mkdir -p $OUT
CSV=$OUT/stage2_results.csv
echo "cell,model,batch,seq,world_size,binding,speedup,std_mean_ms,numa_mean_ms,correctness,numa_args_numa,numa_args_std,status" > $CSV

WARMUP=3
ITERS=10

run_cell () {
  local cell="$1"; local model="$2"; local B="$3"; local S="$4"; local W="$5"; shift 5
  local log="$OUT/${cell}.log"
  local before=$(mktemp)
  ls -d /tmp/*_speedup_inductor_cache_* 2>/dev/null | sort > "$before"

  local binding="none"
  local pre=""
  if [ "$W" = "2" ]; then
    binding="cpunodebind=0,1"
    pre="numactl --cpunodebind=0,1 --membind=0,1"
  fi

  echo "===== $cell  model=$model B=$B S=$S W=$W binding=$binding  $(date -Is)" | tee -a $OUT/progress.log
  $pre "$@" > "$log" 2> "$log.err"
  local rc=$?

  local after=$(mktemp)
  ls -d /tmp/*_speedup_inductor_cache_* 2>/dev/null | sort > "$after"
  local newdirs=$(comm -13 "$before" "$after")

  # Integrity: numa_args call sites per arm. Each run makes one cache dir per
  # arm; the NUMA arm is whichever has more sites, the standard arm must be 0.
  local counts=""
  for d in $newdirs; do
    counts="$counts $(grep -rha "numa_args=" "$d" 2>/dev/null | wc -l)"
  done
  local na_numa=$(echo $counts | tr ' ' '\n' | sort -n | tail -1)
  local na_std=$(echo $counts | tr ' ' '\n' | sort -n | head -1)
  [ -z "$na_numa" ] && na_numa=0
  [ -z "$na_std" ] && na_std=0

  # GPT-2/BERT print just the speedup; gemma --detailed prints
  # "std_mean numa_mean speedup". The last field is the speedup either way.
  local lastline=$(tail -1 "$log")
  local nf=$(echo "$lastline" | awk '{print NF}')
  local speedup=$(echo "$lastline" | awk '{print $NF}')
  local stdm=$(grep -o 'standard mean_ms=[0-9.]*' "$log.err" | head -1 | cut -d= -f2)
  local numam=$(grep -o 'numa_tp mean_ms=[0-9.]*' "$log.err" | head -1 | cut -d= -f2)
  if [ "$nf" = "3" ]; then
    stdm=$(echo "$lastline" | awk '{print $1}')
    numam=$(echo "$lastline" | awk '{print $2}')
  fi
  local corr=$(grep -o 'correctness: .*' "$log.err" "$log" 2>/dev/null | grep -o 'PASS\|FAIL' | head -1)
  [ -z "$corr" ] && corr="n/a"
  local status="ok"
  [ $rc -ne 0 ] && status="rc=$rc"

  echo "$cell,$model,$B,$S,$W,$binding,$speedup,$stdm,$numam,$corr,$na_numa,$na_std,$status" >> $CSV
  echo "  -> speedup=$speedup std=${stdm}ms numa=${numam}ms corr=$corr numa_args=$na_numa $status" | tee -a $OUT/progress.log

  for d in $newdirs; do rm -rf "$d"; done
  rm -f "$before" "$after"
}

GPT2=scripts/inductor/gpt2_multi_size_speedup_cli.py
BERT=scripts/inductor/bert_multi_size_speedup_cli.py
GEMMA=scripts/inductor/simple_gemma_decoder_block_speedup_cli.py

for W in 4 2; do
  run_cell "c1_gpt2_117m_b1_s1024_w$W" gpt2-117m 1 1024 $W \
    python $GPT2 --model 117m --batch-size 1 --seq-length 1024 --world-size $W \
      --warmup $WARMUP --iters $ITERS --check-correctness

  run_cell "c2_gpt2_117m_b1_s2048_w$W" gpt2-117m 1 2048 $W \
    python $GPT2 --model 117m --batch-size 1 --seq-length 2048 --world-size $W \
      --warmup $WARMUP --iters $ITERS --check-correctness

  run_cell "c3_gpt2_345m_b1_s2048_w$W" gpt2-345m 1 2048 $W \
    python $GPT2 --model 345m --batch-size 1 --seq-length 2048 --world-size $W \
      --warmup $WARMUP --iters $ITERS --check-correctness

  run_cell "c4_bert_large_b1_s2048_w$W" bert-large 1 2048 $W \
    python $BERT --model large --batch-size 1 --seq-length 2048 --world-size $W \
      --warmup $WARMUP --iters $ITERS --check-correctness

  run_cell "c5_gemma_270m_b1_s2048_w$W" gemma-270m 1 2048 $W \
    python $GEMMA --batch-size 1 --seq-length 2048 --num-layers 18 \
      --h-in 1152 --num-heads 4 --num-kv-heads 1 --head-dim 256 \
      --mlp-hidden 6912 --num-tp-shards $W --warmup $WARMUP --iters $ITERS --detailed
done

echo "STAGE2DONE $(date -Is)" | tee -a $OUT/progress.log
