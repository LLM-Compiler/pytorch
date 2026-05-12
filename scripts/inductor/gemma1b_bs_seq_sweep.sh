#!/usr/bin/env bash
# Gemma-3 1B end-to-end prefill sweep: NUMA-TP vs standard speedup across
# (batch_size, seq_length) ∈ { (1,4096), (2,2048), (4,512), (8,256) }.
#
# Architecture (Gemma-3 1B; matches gemma_multi_size_compile_debug_sweep.py preset):
#   hidden_size       = 1152
#   num_heads         = 4
#   num_kv_heads      = 1   (MQA)
#   head_dim          = 256
#   intermediate_size = 6912
#   num_layers        = 26  (full base model)
#
# Usage:
#   conda activate triton_cpu_integrated
#   bash scripts/inductor/gemma1b_bs_seq_sweep.sh
#
# Override defaults via env vars, e.g.
#   NUM_LAYERS=1 WARMUP=2 ITERS=5 bash scripts/inductor/gemma1b_bs_seq_sweep.sh
set -euo pipefail

NUM_LAYERS="${NUM_LAYERS:-26}"
H_IN="${H_IN:-1152}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_KV_HEADS="${NUM_KV_HEADS:-1}"
HEAD_DIM="${HEAD_DIM:-256}"
MLP_HIDDEN="${MLP_HIDDEN:-6912}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"

CONFIGS=(
  "1 4096"
  "2 2048"
  "4 512"
  "8 256"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${SCRIPT_DIR}/simple_gemma_decoder_block_speedup_cli.py"

if [[ ! -f "${CLI}" ]]; then
  echo "ERROR: speedup CLI not found at ${CLI}" >&2
  exit 1
fi

printf 'Gemma-3 1B end-to-end prefill (L=%s)  NUMA-TP vs standard\n' "${NUM_LAYERS}"
printf '%-4s %-6s %-12s\n' "B" "S" "speedup"
printf '%-4s %-6s %-12s\n' "----" "------" "------------"

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
            --iters "${ITERS}" 2>/tmp/gemma1b_sweep_err.log)"; then
    printf '%-4s %-6s %-12s\n' "${B}" "${S}" "${out}"
  else
    printf '%-4s %-6s %-12s\n' "${B}" "${S}" "FAIL"
    echo "  (stderr → /tmp/gemma1b_sweep_err.log)" >&2
  fi
done
