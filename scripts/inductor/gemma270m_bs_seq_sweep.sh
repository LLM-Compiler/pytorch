#!/usr/bin/env bash
# Gemma-3 270M end-to-end prefill sweep: NUMA-TP vs standard speedup across
# (batch_size, seq_length) ∈ { (1,4096), (2,2048), (4,512), (8,256) }.
#
# Architecture (Gemma-3 270M; matches gemma_multi_size_compile_debug_sweep.py preset):
#   hidden_size       = 1152
#   num_heads         = 4
#   num_kv_heads      = 1   (MQA)
#   head_dim          = 256
#   intermediate_size = 6912
#   num_layers        = 18  (full base model)
#
# Usage:
#   conda activate triton_cpu_integrated
#   bash scripts/inductor/gemma270m_bs_seq_sweep.sh
#
# Override defaults via env vars, e.g.
#   NUM_LAYERS=1 WARMUP=2 ITERS=5 bash scripts/inductor/gemma270m_bs_seq_sweep.sh
set -euo pipefail

NUM_LAYERS="${NUM_LAYERS:-18}"
H_IN="${H_IN:-1152}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_KV_HEADS="${NUM_KV_HEADS:-1}"
HEAD_DIM="${HEAD_DIM:-256}"
MLP_HIDDEN="${MLP_HIDDEN:-6912}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"

CONFIGS=(
  "1 2048"
  "1 4096"
  "1 8192"
  "2 2048"
  "2 4096"
  "2 8192"
  "4 512"
  "4 1024"
  "4 2048"
  "4 4096"
  "4 8192"
  "8 256"
  "8 512"
  "8 1024"
  "8 2048"
  "8 4096"
  "8 8192"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${SCRIPT_DIR}/simple_gemma_decoder_block_speedup_cli.py"

if [[ ! -f "${CLI}" ]]; then
  echo "ERROR: speedup CLI not found at ${CLI}" >&2
  exit 1
fi

CSV_OUT="gemma270m_sweep_results.csv"
if [[ ! -f "${CSV_OUT}" ]]; then
  echo "B,S,std_ms,numa_ms,speedup" > "${CSV_OUT}"
fi

printf 'Gemma-3 270M end-to-end prefill (L=%s)  NUMA-TP vs standard\n' "${NUM_LAYERS}"
printf '%-4s %-6s %-10s %-10s %-10s\n' "B" "S" "std(ms)" "numa(ms)" "speedup"
printf '%-4s %-6s %-10s %-10s %-10s\n' "----" "------" "----------" "----------" "----------"

for cfg in "${CONFIGS[@]}"; do
  B="${cfg% *}"
  S="${cfg#* }"
  if out="$(python "${CLI}" \
            --batch-size "${B}" \
            --seq-length "${S}" \
            --num-layers "${NUM_LAYERS}" \
            --h-in "${H_IN}" \
            --num-heads "${NUM_HEADS}" \
            --num-kv-heads "${NUM_KV_HEADS}" \
            --head-dim "${HEAD_DIM}" \
            --mlp-hidden "${MLP_HIDDEN}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}" \
            --detailed 2>/tmp/gemma270m_sweep_err.log)"; then
    read std_mean numa_mean speedup <<< "${out}"
    printf '%-4s %-6s %-10s %-10s %-10s\n' "${B}" "${S}" "${std_mean}" "${numa_mean}" "${speedup}"
    echo "${B},${S},${std_mean},${numa_mean},${speedup}" >> "${CSV_OUT}"
    git add "${CSV_OUT}"
    git commit -m "Auto-commit: Gemma 270M benchmark result B=${B} S=${S}" || true
    git push origin feature/niman_gemma_prefil_pass || true
  else
    printf '%-4s %-6s %-10s %-10s %-10s\n' "${B}" "${S}" "FAIL" "FAIL" "FAIL"
    echo "${B},${S},FAIL,FAIL,FAIL" >> "${CSV_OUT}"
    git add "${CSV_OUT}"
    git commit -m "Auto-commit: Gemma 270M benchmark result B=${B} S=${S} (FAIL)" || true
    git push origin feature/niman_gemma_prefil_pass || true
    echo "  (stderr → /tmp/gemma270m_sweep_err.log)" >&2
  fi
done
