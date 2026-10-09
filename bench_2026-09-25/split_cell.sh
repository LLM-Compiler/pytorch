#!/bin/bash
# Run ONE sweep cell with the two arms in SEPARATE processes.
#   MODEL=gpt2-762m B=1 S=2048 W=4 bash split_cell.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch

MODEL=${MODEL:?}; B=${B:?}; S=${S:?}; W=${W:-4}
WARMUP=${WARMUP:-10}; ITERS=${ITERS:-50}
OUT=${OUT:-bench_2026-09-25/stage3b}
mkdir -p $OUT/logs $OUT/tmp
TAG="${MODEL}_b${B}_s${S}_w${W}"
PFX="$OUT/tmp/$TAG"
R=bench_2026-09-25

say () { echo "$(date -Is) $*" | tee -a $OUT/progress.log; }

run_arm () {
  local arm=$1
  say "  [$TAG] arm=$arm starting"
  /usr/bin/time -v -o $OUT/logs/${TAG}.${arm}.time \
    python $R/split_arm.py --model $MODEL --batch-size $B --seq-length $S \
      --world-size $W --arm $arm --warmup $WARMUP --iters $ITERS \
      --out-prefix ${PFX}.${arm} \
      > $OUT/logs/${TAG}.${arm}.log 2> $OUT/logs/${TAG}.${arm}.err
  local rc=$?
  local sig=$(grep -a "terminated by signal" $OUT/logs/${TAG}.${arm}.time 2>/dev/null | sed 's/.*signal //')
  local rss=$(grep -a "Maximum resident" $OUT/logs/${TAG}.${arm}.time 2>/dev/null | grep -oE "[0-9]+$")
  if [ $rc -ne 0 ]; then
    say "  [$TAG] arm=$arm FAILED rc=$rc signal=${sig:-none} rss=$((${rss:-0}/1048576))GB"
    return 1
  fi
  say "  [$TAG] arm=$arm ok rss=$((${rss:-0}/1048576))GB"
  return 0
}

say "=== cell $TAG (split-process) ==="
run_arm standard || { echo "$MODEL,$B,$S,$W,,,,,,standard-arm-failed" >> $OUT/results.csv; exit 1; }
run_arm numa     || { echo "$MODEL,$B,$S,$W,,,,,,numa-arm-failed" >> $OUT/results.csv; exit 1; }

say "  [$TAG] comparing"
python $R/split_compare.py --std-prefix ${PFX}.standard --numa-prefix ${PFX}.numa \
  > $OUT/logs/${TAG}.cmp.json 2> $OUT/logs/${TAG}.cmp.err
CMPRC=$?
python - "$OUT/logs/${TAG}.cmp.json" "$OUT/results.csv" <<'PY'
import json,sys,os
rec=json.load(open(sys.argv[1]))
line="{model},{batch},{seq},{world_size},{speedup},{std_mean_ms},{numa_mean_ms},{correctness},{max_abs},{corr},{std_peak_rss_gb},{numa_peak_rss_gb},ok".format(**rec)
hdr="model,batch,seq,world_size,speedup,std_mean_ms,numa_mean_ms,correctness,max_abs,corr,std_rss_gb,numa_rss_gb,status"
p=sys.argv[2]
if not os.path.exists(p): open(p,"w").write(hdr+"\n")
open(p,"a").write(line+"\n")
print(line)
PY
say "  [$TAG] done (compare rc=$CMPRC)"
rm -f ${PFX}.standard.out.pt ${PFX}.numa.out.pt
