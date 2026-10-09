#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25
OUT=$R/colpar
mkdir -p $OUT
say () { echo "$(date -Is) $*" | tee -a $OUT/progress.log; }

# Phase A: correctness must be IDENTICAL -- same data, only placement differs.
say "PHASE A: correctness matrix, W=4, NUMA_COLPAR_VIEW=1"
rm -f /tmp/nc_*.pt
for NL in 1 2; do
  PASSES=std NLAYERS=$NL WS=4 python $R/numa_check.py >> $OUT/phaseA.log 2>&1
  for P in mlp attn both; do
    NUMA_COLPAR_VIEW=1 PASSES=$P NLAYERS=$NL WS=4 python $R/numa_check.py >> $OUT/phaseA.log 2>&1 \
      || say "  CELL FAILED: $P L$NL"
  done
done
: > $OUT/phaseA_compare.txt
for NL in 1 2; do
  for P in mlp attn both; do
    python $R/cmp.py /tmp/nc_std_w4_L${NL}_s128.pt /tmp/nc_${P}_w4_L${NL}_s128.pt >> $OUT/phaseA_compare.txt 2>&1
  done
done
cat $OUT/phaseA_compare.txt | tee -a $OUT/progress.log
NPASS=$(grep -c "corr=+1.00000" $OUT/phaseA_compare.txt)
say "PHASE A: $NPASS of 6 at corr=1.000000"
[ "$NPASS" -ne 6 ] && { say "PHASE A FAILED - stopping"; say "COLPARDONE-FAILED"; exit 1; }

# Phase B: A/B on a cell with a known baseline (1.591x, 48.65 GB numa arm)
say "PHASE B: A/B gpt2-762m B=1 S=2048 W=4"
for V in off on; do
  env_val=""; [ "$V" = "on" ] && env_val="1"
  for ARM in standard numa; do
    say "  variant=$V arm=$ARM"
    NUMA_COLPAR_VIEW=$env_val /usr/bin/time -v -o $OUT/${V}_${ARM}.time \
      python $R/split_arm.py --model gpt2-762m --batch-size 1 --seq-length 2048 \
        --world-size 4 --arm $ARM --warmup 3 --iters 10 \
        --out-prefix $OUT/${V}_${ARM} > $OUT/${V}_${ARM}.log 2> $OUT/${V}_${ARM}.err
    rc=$?
    rss=$(grep -a "Maximum resident" $OUT/${V}_${ARM}.time | grep -oE "[0-9]+$")
    say "    rc=$rc rss=$((${rss:-0}/1048576))GB"
  done
  python $R/split_compare.py --std-prefix $OUT/${V}_standard --numa-prefix $OUT/${V}_numa \
    > $OUT/cmp_${V}.json 2> $OUT/cmp_${V}.err
  say "    $(cat $OUT/cmp_${V}.json)"
done

say "=== A/B summary ==="
python - <<'PY' | tee -a $OUT/progress.log
import json
for v in ("off","on"):
    try:
        r=json.load(open(f"bench_2026-09-25/colpar/cmp_{v}.json"))
        print(f"  COLPAR_VIEW={v:<4} speedup={r['speedup']:.3f}x  numa={r['numa_mean_ms']:.0f}ms  "
              f"numa_rss={r['numa_peak_rss_gb']:.1f}GB  corr={r['correctness']}")
    except Exception as e:
        print(f"  COLPAR_VIEW={v}: {e}")
PY
say "COLPARDONE-OK"
