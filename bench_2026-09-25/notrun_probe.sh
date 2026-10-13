#!/bin/bash
# Attempt every NOT-RUN cell. Gemma's remaining one first (user priority), then
# the rest ordered by B*H*S^2, which is what both limits key off. Cells over the
# 2 GB SDPA-MATH guard are expected to crash; we run them anyway to record what
# actually happens rather than relying on the projection.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25
OUT=$R/notrun_probe
mkdir -p $OUT/logs $OUT/tmp
CSV=$OUT/results.csv
say(){ echo "$(date -Is) $*" | tee -a $OUT/progress.log; }
export SSH_AUTH_SOCK=/home/shavin/.ssh/agent.sock
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
GIT_ID=(-c user.name=ThisaraWeerakoon -c user.email=thisara.weerakoon2001@gmail.com)
publish(){ git add -A $OUT >/dev/null 2>&1; git diff --cached --quiet && return 0
  git "${GIT_ID[@]}" commit -q -m "$1" >/dev/null 2>&1 || return 0
  git push -q origin fix/numa-tp-row-parallel-allreduce >/dev/null 2>&1 || say "   (push failed, local)"; }

[ -f $CSV ] || echo "model,batch,seq,world_size,speedup,std_mean_ms,numa_mean_ms,correctness,numa_rss_gb,outcome" > $CSV

# model cli preset B S   -- gemma first, then ascending B*H*S^2
CELLS="
gemma-270m gemma - 8 4096
bert-base bert base 8 2048
gpt2-117m gpt2 117m 8 2048
bert-large bert large 8 2048
gpt2-345m gpt2 345m 8 2048
gpt2-762m gpt2 762m 8 2048
bert-base bert base 4 4096
gpt2-117m gpt2 117m 4 4096
bert-large bert large 4 4096
gpt2-345m gpt2 345m 4 4096
gpt2-762m gpt2 762m 4 4096
bert-base bert base 8 4096
gpt2-117m gpt2 117m 8 4096
bert-large bert large 8 4096
gpt2-345m gpt2 345m 8 4096
gpt2-762m gpt2 762m 8 4096
"
TOTAL=$(echo "$CELLS" | grep -c '[a-z]')
say "notrun probe: $TOTAL cells, W=4, warmup 3 iters 10 (timing is indicative; the question is whether they run at all)"
publish "bench(notrun): probe started, $TOTAL cells"

n=0
echo "$CELLS" | while read -r MODEL CLI PRESET B S; do
  [ -z "$MODEL" ] && continue
  n=$((n+1))
  grep -q "^$MODEL,$B,$S,4," $CSV 2>/dev/null && { say "[$n/$TOTAL] $MODEL B=$B S=$S recorded"; continue; }
  TAG="${MODEL}_b${B}_s${S}"; PFX="$OUT/tmp/$TAG"
  say "[$n/$TOTAL] $MODEL B=$B S=$S  free=$(free -g | awk '/^Mem:/{print $7}')GB"
  ok=1
  for ARM in standard numa; do
    if [ "$CLI" = "gemma" ]; then
      CMD="python $R/gemma_arm.py --batch-size $B --seq-length $S --world-size 4 --arm $ARM --warmup 3 --iters 10 --out-prefix ${PFX}.${ARM}"
    else
      CMD="python $R/split_arm.py --model $MODEL --batch-size $B --seq-length $S --world-size 4 --arm $ARM --warmup 3 --iters 10 --out-prefix ${PFX}.${ARM}"
    fi
    TORCHINDUCTOR_CACHE_DIR=/tmp/np_${TAG}_${ARM} timeout -k 60 10800 \
      /usr/bin/time -v -o $OUT/logs/${TAG}.${ARM}.time $CMD \
      > $OUT/logs/${TAG}.${ARM}.log 2> $OUT/logs/${TAG}.${ARM}.err
    rc=$?
    rss=$(grep -a "Maximum resident" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null | grep -oE "[0-9]+$")
    sig=$(grep -a "terminated by signal" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null | sed 's/.*signal //')
    rm -rf /tmp/np_${TAG}_${ARM}
    if [ $rc -ne 0 ]; then
      why="rc=$rc"
      [ "$sig" = "9" ] && why="OOM (signal 9)"
      [ $rc -eq 124 ] && why="TIMEOUT"
      grep -q "stride divisible by 2" $OUT/logs/${TAG}.${ARM}.err 2>/dev/null && why="view_as_complex stride"
      grep -qi "invalid for input of size" $OUT/logs/${TAG}.${ARM}.err 2>/dev/null && why="shape mismatch"
      [ $rc -eq 139 ] && why="SEGFAULT"
      say "   arm=$ARM FAILED: $why  rss=$((${rss:-0}/1048576))GB"
      echo "$MODEL,$B,$S,4,,,,,$((${rss:-0}/1048576)),${ARM}-arm: $why" >> $CSV
      ok=0; break
    fi
    say "   arm=$ARM ok rss=$((${rss:-0}/1048576))GB"
  done
  if [ $ok -eq 1 ]; then
    python $R/split_compare.py --std-prefix ${PFX}.standard --numa-prefix ${PFX}.numa \
      > $OUT/logs/${TAG}.cmp.json 2> $OUT/logs/${TAG}.cmp.err
    python - "$OUT/logs/${TAG}.cmp.json" "$CSV" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
open(sys.argv[2],"a").write("{model},{batch},{seq},{world_size},{speedup},{std_mean_ms},"
  "{numa_mean_ms},{correctness},{numa_peak_rss_gb},RUNS\n".format(**r))
PY
    say "   -> $(tail -1 $CSV)"
  fi
  rm -f ${PFX}.standard.out.pt ${PFX}.numa.out.pt
  publish "bench(notrun): [$n/$TOTAL] $MODEL B=$B S=$S"
done

say "=== probe summary ==="
cat $CSV | tee -a $OUT/progress.log
publish "bench(notrun): probe complete"
say "NOTRUNPROBEDONE"
echo DONE > /tmp/np_done
