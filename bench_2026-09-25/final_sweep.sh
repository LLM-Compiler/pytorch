#!/bin/bash
# Final sweep: every cell of the 144 grid that still carries no number.
#
# Differences from the earlier probe, all deliberate:
#   * No pre-flight B*H*S^2 guard. The 2 GB SDPA-MATH overflow that guard
#     existed for is fixed (int64 element offsets in the bmm template), so the
#     cells it used to exclude are now run like any other.
#   * Both arms are always attempted. The probe stopped a cell as soon as one
#     arm failed, which hid whether the other arm could have run. Here each arm
#     is recorded on its own, so a cell's limit is attributable to an arm.
#   * One retry for failures that are neither OOM nor the time cap, because
#     autotuning makes layout selection timing-dependent and a cell can fail to
#     compile once and pass immediately after.
#   * Per-arm cap raised 3 h -> 8 h. This is the only limit left, it is there
#     purely so one hung cell cannot stall the queue, and hitting it is recorded
#     explicitly as a cap rather than as a failure.
#
# The only outcome treated as final-and-acceptable is memory exhaustion: those
# cells are abandoned and reported as memory overflow, with the measured peak.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25          # results root; readonly so no temp can clobber it
readonly R
OUT=$R/final_sweep
mkdir -p $OUT/logs $OUT/tmp
CSV=$OUT/results.csv
CAP=28800
say(){ echo "$(date -Is) $*" | tee -a $OUT/progress.log; }
export SSH_AUTH_SOCK=/home/shavin/.ssh/agent.sock
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
GIT_ID=(-c user.name=ThisaraWeerakoon -c user.email=thisara.weerakoon2001@gmail.com)
publish(){ git add -A $OUT >/dev/null 2>&1; git diff --cached --quiet && return 0
  git "${GIT_ID[@]}" commit -q -m "$1" >/dev/null 2>&1 || return 0
  git push -q origin fix/numa-tp-row-parallel-allreduce >/dev/null 2>&1 || say "   (push failed, local)"; }

[ -f $CSV ] || echo "model,batch,seq,world_size,speedup,std_mean_ms,numa_mean_ms,correctness,numa_rss_gb,outcome,std_rss_gb,std_outcome,numa_outcome,std_elapsed_s,numa_elapsed_s" > $CSV

# model cli preset B S  -- ascending B*H*S^2, which is what every limit keys off
CELLS="
gpt2-762m gpt2 762m 2 2048
gpt2-345m gpt2 345m 1 4096
gpt2-345m gpt2 345m 4 2048
bert-large bert large 1 4096
bert-large bert large 4 2048
gpt2-762m gpt2 762m 1 4096
gpt2-762m gpt2 762m 4 2048
gpt2-117m gpt2 117m 8 2048
gpt2-345m gpt2 345m 2 4096
gpt2-345m gpt2 345m 8 2048
bert-large bert large 2 4096
bert-large bert large 8 2048
gpt2-762m gpt2 762m 2 4096
gpt2-762m gpt2 762m 8 2048
gpt2-117m gpt2 117m 4 4096
bert-base bert base 4 4096
gpt2-345m gpt2 345m 4 4096
bert-large bert large 4 4096
gpt2-762m gpt2 762m 4 4096
gpt2-117m gpt2 117m 8 4096
bert-base bert base 8 4096
gpt2-345m gpt2 345m 8 4096
bert-large bert large 8 4096
gpt2-762m gpt2 762m 8 4096
"
TOTAL=$(echo "$CELLS" | grep -c '[a-z]')
say "final sweep: $TOTAL cells, W=4, warmup 3 iters 10, per-arm cap ${CAP}s, no size guard"
publish "bench(final): sweep started, $TOTAL cells, no guards"

# run_arm MODEL CLI PRESET B S ARM TAG
# Writes "<outcome>|<peak RSS GB>" to $OUT/tmp/<TAG>.<ARM>.outcome. Nothing is
# returned on stdout, because the caller needs two values and a subshell would
# swallow any variable set in here.
run_arm(){
  local MODEL=$1 CLI=$2 PRESET=$3 B=$4 S=$5 ARM=$6 TAG=$7
  local PFX=$OUT/tmp/$TAG CMD rc rss sig why try gb
  if [ "$CLI" = "gemma" ]; then
    CMD="python $R/gemma_arm.py --batch-size $B --seq-length $S --world-size 4 --arm $ARM --warmup 3 --iters 10 --out-prefix ${PFX}.${ARM}"
  else
    CMD="python $R/split_arm.py --model $MODEL --batch-size $B --seq-length $S --world-size 4 --arm $ARM --warmup 3 --iters 10 --out-prefix ${PFX}.${ARM}"
  fi
  why=ok
  for try in 1 2; do
    rm -rf /tmp/fs_${TAG}_${ARM}
    TORCHINDUCTOR_CACHE_DIR=/tmp/fs_${TAG}_${ARM} timeout -k 60 $CAP \
      /usr/bin/time -v -o $OUT/logs/${TAG}.${ARM}.time $CMD \
      > $OUT/logs/${TAG}.${ARM}.log 2> $OUT/logs/${TAG}.${ARM}.err
    rc=$?
    rss=$(grep -a "Maximum resident" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null | grep -oE "[0-9]+$")
    sig=$(grep -a "terminated by signal" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null | sed 's/.*signal //')
    # Elapsed wall clock, in seconds, from /usr/bin/time's "h:mm:ss or m:ss".
    el=$(grep -a "Elapsed (wall clock)" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null \
         | awk '{print $NF}' \
         | awk -F: '{s=0; for(i=1;i<=NF;i++) s=s*60+$i; printf "%d", s}')
    # /usr/bin/time writes its Elapsed line at process exit; under extreme
    # starvation the child can be too wedged for it to flush before the
    # outer `timeout -k` SIGKILL arrives, leaving the .time file empty. A
    # bare `rc -eq 124` from the OUTER timeout is authoritative proof the
    # full cap elapsed even when the inner stats are missing -- fall back
    # to $CAP rather than a misleading 0, which would read as "instant".
    if [ -z "$el" ] && [ $rc -eq 124 ]; then el=$CAP; else el=${el:-0}; fi
    rm -rf /tmp/fs_${TAG}_${ARM}
    gb=$(( ${rss:-0} / 1048576 ))
    if [ $rc -eq 0 ]; then why=ok; break; fi
    # Attributing the kill correctly matters: `timeout -k` escalates to SIGKILL,
    # which is indistinguishable from an OOM kill by signal number alone. The
    # earlier probe called both "OOM (signal 9)" and so recorded two cells as
    # memory failures at 15-16 GB on a 62 GB machine, which cannot be memory.
    # Elapsed time against the cap settles it.
    why="rc=$rc"
    [ $rc -eq 139 ] && why="SEGFAULT"
    if [ "$sig" = "9" ] || [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
      if [ $rc -eq 124 ] || [ $el -ge $((CAP - 120)) ]; then
        why="HIT ${CAP}s CAP (elapsed ${el}s)"
      else
        why="MEMORY OVERFLOW (killed by signal 9 at ${gb}GB after ${el}s)"
      fi
    fi
    grep -q "stride divisible by 2" $OUT/logs/${TAG}.${ARM}.err 2>/dev/null && why="compile: view_as_complex stride"
    grep -qi "invalid for input of size" $OUT/logs/${TAG}.${ARM}.err 2>/dev/null && why="compile: shape mismatch"
    # memory and the cap are final; anything else gets exactly one retry
    case "$why" in *"MEMORY OVERFLOW"*|*CAP*) break;; esac
    [ $try -eq 2 ] && break
    say "   arm=$ARM attempt 1 failed ($why), retrying once"
  done
  echo "${why}|${gb}|${el}" > $OUT/tmp/${TAG}.${ARM}.outcome
}

n=0
echo "$CELLS" | while read -r MODEL CLI PRESET B S; do
  [ -z "$MODEL" ] && continue
  n=$((n+1))
  grep -q "^$MODEL,$B,$S,4," $CSV 2>/dev/null && { say "[$n/$TOTAL] $MODEL B=$B S=$S already recorded"; continue; }
  TAG="${MODEL}_b${B}_s${S}"; PFX="$OUT/tmp/$TAG"
  say "[$n/$TOTAL] $MODEL B=$B S=$S  free=$(free -g | awk '/^Mem:/{print $7}')GB"

  run_arm $MODEL $CLI $PRESET $B $S standard $TAG
  L=$(cat $OUT/tmp/${TAG}.standard.outcome)
  SO=${L%%|*}; _rest=${L#*|}; SR=${_rest%%|*}; SE=${_rest#*|}
  say "   standard: $SO  rss=${SR}GB  elapsed=${SE}s"
  run_arm $MODEL $CLI $PRESET $B $S numa $TAG
  L=$(cat $OUT/tmp/${TAG}.numa.outcome)
  NO=${L%%|*}; _rest=${L#*|}; NR=${_rest%%|*}; NE=${_rest#*|}
  say "   numa:     $NO  rss=${NR}GB  elapsed=${NE}s"

  if [ "$SO" = "ok" ] && [ "$NO" = "ok" ]; then
    python $R/split_compare.py --std-prefix ${PFX}.standard --numa-prefix ${PFX}.numa \
      > $OUT/logs/${TAG}.cmp.json 2> $OUT/logs/${TAG}.cmp.err
    python - "$OUT/logs/${TAG}.cmp.json" "$CSV" "$SR" "$SE" "$NE" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); r["std_rss"]=sys.argv[3]
r["std_el"]=sys.argv[4]; r["numa_el"]=sys.argv[5]
open(sys.argv[2],"a").write("{model},{batch},{seq},{world_size},{speedup},{std_mean_ms},"
  "{numa_mean_ms},{correctness},{numa_peak_rss_gb},RUNS,{std_rss},ok,ok,"
  "{std_el},{numa_el}\n".format(**r))
PY
  else
    O="standard-arm: $SO; numa-arm: $NO"
    echo "$MODEL,$B,$S,4,,,,,$NR,$O,$SR,$SO,$NO,$SE,$NE" >> $CSV
  fi
  say "   -> $(tail -1 $CSV)"
  rm -f ${PFX}.standard.out.pt ${PFX}.numa.out.pt
  publish "bench(final): [$n/$TOTAL] $MODEL B=$B S=$S"
done

say "=== final sweep summary ==="
cat $CSV | tee -a $OUT/progress.log
publish "bench(final): sweep complete"
say "FINALSWEEPDONE"
echo DONE > /tmp/fs_done
