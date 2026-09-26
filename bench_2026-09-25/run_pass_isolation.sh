#!/usr/bin/env bash
# Driver: one subprocess per pass combination so a segfault kills only that cell.
set -uo pipefail

WS="${WS:-4}"
B="${B:-1}"
S="${S:-128}"
L="${L:-2}"
OUT="${OUT:-/home/shavin/llm-cpu-compiler/pytorch/bench_2026-09-25/iso_W${WS}_L${L}}"

source /home/shavin/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:${LD_LIBRARY_PATH:-}
cd /home/shavin/llm-cpu-compiler/pytorch

SCRIPT=bench_2026-09-25/numa_pass_isolation.py
mkdir -p "$OUT"
RESULTS="$OUT/results.jsonl"
: > "$RESULTS"

echo "### world_size=$WS  B=$B  S=$S  layers=$L  out=$OUT"

echo "### computing reference (all passes off)"
timeout 1200 python "$SCRIPT" --mode ref --out "$OUT" \
  --world-size "$WS" --batch-size "$B" --seq-length "$S" ${L:+--num-layers $L} \
  2>"$OUT/ref.stderr" | tee -a "$RESULTS"
if [ ! -f "$OUT/ref.pt" ]; then
  echo "### FATAL: reference failed; see $OUT/ref.stderr"; tail -20 "$OUT/ref.stderr"; exit 1
fi

COMBOS=(
  ""
  "mlp"
  "attention"
  "layer_norm"
  "residual"
  "mlp,residual"
  "attention,residual"
  "layer_norm,attention"
  "layer_norm,mlp"
  "attention,mlp"
  "layer_norm,attention,mlp,residual"
)

for c in "${COMBOS[@]}"; do
  name="${c:-none}"
  safe=$(echo "$name" | tr ',' '_')
  timeout 1200 python "$SCRIPT" --mode combo --combo "$c" --out "$OUT" \
    --world-size "$WS" --batch-size "$B" --seq-length "$S" ${L:+--num-layers $L} \
    2>"$OUT/$safe.stderr" | tee -a "$RESULTS"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "{\"mode\":\"combo\",\"combo\":\"$name\",\"verdict\":\"CRASH\",\"rc\":$rc}" \
      | tee -a "$RESULTS"
  fi
done

echo "### done -> $RESULTS"
