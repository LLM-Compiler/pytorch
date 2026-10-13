#!/bin/bash
# The two Gemma cells excluded only by the B*S <= 8192 token cap, not by any
# hard limit. Projected: B=4 S=4096 ~44 GB, B=8 S=2048 ~24 GB -- both inside
# the machine's 62 GB, and neither approaches the 2 GB SDPA-MATH guard
# (Gemma has 4 heads, so B*H*S^2 stays small).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25
OUT=$R/gemma_sweep
mkdir -p $OUT/logs $OUT/tmp
CSV=$OUT/results.csv
say(){ echo "$(date -Is) $*" | tee -a $OUT/progress.log; }

export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/home/shavin/.ssh/agent.sock}
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
BRANCH=fix/numa-tp-row-parallel-allreduce
GIT_ID=(-c user.name=ThisaraWeerakoon -c user.email=thisara.weerakoon2001@gmail.com)
publish(){ git add -A $OUT >/dev/null 2>&1; git diff --cached --quiet && return 0
  git "${GIT_ID[@]}" commit -q -m "$1" >/dev/null 2>&1 || return 0
  git push -q origin $BRANCH >/dev/null 2>&1 || say "   (push failed, commit is local)"; }

say "=== gemma extra cells (beyond the 8192 token cap) ==="
for cell in "8:2048" "4:4096"; do
  B=${cell%%:*}; S=${cell##*:}
  grep -q "^gemma-270m,$B,$S,4," $CSV 2>/dev/null && { say "B=$B S=$S already recorded"; continue; }
  TAG="gemma_b${B}_s${S}_w4"; PFX="$OUT/tmp/$TAG"
  say "B=$B S=$S  (tokens=$((B*S)))  free=$(free -g | awk '/^Mem:/{print $7}')GB"
  ok=1
  for ARM in standard numa; do
    TORCHINDUCTOR_CACHE_DIR=/tmp/gx_${TAG}_${ARM} timeout -k 60 21600 \
      /usr/bin/time -v -o $OUT/logs/${TAG}.${ARM}.time \
      python $R/gemma_arm.py --batch-size $B --seq-length $S --world-size 4 \
        --arm $ARM --warmup 10 --iters 50 --out-prefix ${PFX}.${ARM} \
        > $OUT/logs/${TAG}.${ARM}.log 2> $OUT/logs/${TAG}.${ARM}.err
    rc=$?
    rss=$(grep -a "Maximum resident" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null | grep -oE "[0-9]+$")
    if [ $rc -ne 0 ]; then
      sig=$(grep -a "terminated by signal" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null | sed 's/.*signal //')
      say "   arm=$ARM FAILED rc=$rc signal=${sig:-none} rss=$((${rss:-0}/1048576))GB"
      echo "gemma-270m,$B,$S,4,,,,,,,,,${ARM}-arm-failed" >> $CSV; ok=0; break
    fi
    say "   arm=$ARM ok rss=$((${rss:-0}/1048576))GB"
    rm -rf /tmp/gx_${TAG}_${ARM}
  done
  if [ $ok -eq 1 ]; then
    python $R/split_compare.py --std-prefix ${PFX}.standard --numa-prefix ${PFX}.numa \
      > $OUT/logs/${TAG}.cmp.json 2> $OUT/logs/${TAG}.cmp.err
    python - "$OUT/logs/${TAG}.cmp.json" "$CSV" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
open(sys.argv[2],"a").write(
 "{model},{batch},{seq},{world_size},{speedup},{std_mean_ms},{numa_mean_ms},"
 "{correctness},{max_abs},{corr},{std_peak_rss_gb},{numa_peak_rss_gb},ok\n".format(**r))
PY
    say "   -> $(tail -1 $CSV)"
  fi
  rm -f ${PFX}.standard.out.pt ${PFX}.numa.out.pt
  publish "bench(gemma): extra cell B=$B S=$S (beyond token cap)"
done
say "GEMMAEXTRADONE"
echo DONE > /tmp/gx_done
