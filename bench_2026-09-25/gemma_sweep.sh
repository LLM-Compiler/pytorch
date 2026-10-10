#!/bin/bash
# Gemma 270M grid sweep, W=4, split-process. Newly possible: Gemma was excluded
# from the entire stage3 grid because it was numerically wrong.
# Grid matches stage3: B {1,2,4,8} x S {128..4096}, tiered S <= 8192/B.
# Gemma has H=4 so B*H*S^2 never approaches the 2 GB SDPA-MATH limit.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25
OUT=$R/gemma_sweep
mkdir -p $OUT/logs $OUT/tmp
CSV=$OUT/results.csv
say () { echo "$(date -Is) $*" | tee -a $OUT/progress.log; }

export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-/home/shavin/.ssh/agent.sock}
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
BRANCH=fix/numa-tp-row-parallel-allreduce
GIT_ID=(-c user.name=ThisaraWeerakoon -c user.email=thisara.weerakoon2001@gmail.com)
publish () {
  git add -A $OUT >/dev/null 2>&1
  git diff --cached --quiet && return 0
  git "${GIT_ID[@]}" commit -q -m "$1" >/dev/null 2>&1 || { say "   (commit failed)"; return 0; }
  git push -q origin $BRANCH >/dev/null 2>&1 || say "   (push failed, commit is local)"
}

[ -f $CSV ] || echo "model,batch,seq,world_size,speedup,std_mean_ms,numa_mean_ms,correctness,max_abs,corr,std_rss_gb,numa_rss_gb,status" > $CSV

W=4; WARMUP=10; ITERS=50
CELLS=""
for B in 1 2 4 8; do for S in 128 256 512 1024 2048 4096; do
  [ $((B*S)) -le 8192 ] && CELLS="$CELLS $B:$S"
done; done
TOTAL=$(echo $CELLS | wc -w)
say "gemma sweep start: $TOTAL cells, W=$W, split-process, warmup $WARMUP iters $ITERS"
publish "bench(gemma-sweep): started, $TOTAL cells"

n=0
for cell in $CELLS; do
  B=${cell%%:*}; S=${cell##*:}
  n=$((n+1))
  if grep -q "^gemma-270m,$B,$S,$W," $CSV 2>/dev/null; then
    say "[$n/$TOTAL] B=$B S=$S already recorded"; continue
  fi
  TAG="gemma_b${B}_s${S}_w${W}"; PFX="$OUT/tmp/$TAG"
  say "[$n/$TOTAL] B=$B S=$S  free=$(free -g | awk '/^Mem:/{print $7}')GB"

  ok=1
  for ARM in standard numa; do
    TORCHINDUCTOR_CACHE_DIR=/tmp/gsw_${TAG}_${ARM} \
    timeout -k 60 14400 /usr/bin/time -v -o $OUT/logs/${TAG}.${ARM}.time \
      python $R/gemma_arm.py --batch-size $B --seq-length $S --world-size $W \
        --arm $ARM --warmup $WARMUP --iters $ITERS --out-prefix ${PFX}.${ARM} \
        > $OUT/logs/${TAG}.${ARM}.log 2> $OUT/logs/${TAG}.${ARM}.err
    rc=$?
    rss=$(grep -a "Maximum resident" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null | grep -oE "[0-9]+$")
    if [ $rc -ne 0 ]; then
      sig=$(grep -a "terminated by signal" $OUT/logs/${TAG}.${ARM}.time 2>/dev/null | sed 's/.*signal //')
      say "    arm=$ARM FAILED rc=$rc signal=${sig:-none} rss=$((${rss:-0}/1048576))GB"
      echo "gemma-270m,$B,$S,$W,,,,,,,,,${ARM}-arm-failed" >> $CSV
      ok=0; break
    fi
    say "    arm=$ARM ok rss=$((${rss:-0}/1048576))GB"
    rm -rf /tmp/gsw_${TAG}_${ARM}
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
print("  -> speedup=%.3fx corr=%s %s" % (r["speedup"], r["correctness"], r["model"]))
PY
    say "    $(tail -1 $CSV)"
  fi
  rm -f ${PFX}.standard.out.pt ${PFX}.numa.out.pt
  publish "bench(gemma-sweep): [$n/$TOTAL] B=$B S=$S"
done

say "=== gemma sweep summary ==="
python - "$CSV" <<'PY' | tee -a $OUT/progress.log
import csv,sys
rows=[r for r in csv.DictReader(open(sys.argv[1]))]
ok=[r for r in rows if r["status"]=="ok"]
print(f"cells: {len(rows)}  ok: {len(ok)}")
bad=[r for r in ok if r["correctness"]!="PASS"]
print("correctness failures:", len(bad))
for r in sorted(ok,key=lambda r:(int(r["batch"]),int(r["seq"]))):
    print(f"  B={r['batch']:<2} S={r['seq']:<5} {float(r['speedup']):.3f}x  {r['correctness']}")
wins=[r for r in ok if float(r["speedup"])>1.0]
print(f"above 1.0x: {len(wins)}/{len(ok)}")
PY
publish "bench(gemma-sweep): complete"
say "GEMMASWEEPDONE"
