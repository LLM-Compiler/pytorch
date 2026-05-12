#!/usr/bin/env bash
# Sweep batch sizes 1, 2, 4, 8 at S=4096, L=1, num_kv_heads=1 against the
# Gemma decoder-block speedup CLI. Prints one speedup line per batch size.
#
# Usage:
#   conda activate triton_cpu_integrated
#   bash scripts/inductor/gemma_batch_sweep_S4096.sh
#
# Override defaults via env vars, e.g.:
#   SEQ_LENGTH=2048 NUM_LAYERS=18 BATCH_SIZES="1 4" \
#     bash scripts/inductor/gemma_batch_sweep_S4096.sh
set -euo pipefail

SEQ_LENGTH="${SEQ_LENGTH:-4096}"
NUM_LAYERS="${NUM_LAYERS:-1}"
NUM_KV_HEADS="${NUM_KV_HEADS:-1}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${SCRIPT_DIR}/simple_gemma_decoder_block_speedup_cli.py"

printf '%-6s %-12s\n' "B" "speedup"
printf '%-6s %-12s\n' "------" "------------"
for B in ${BATCH_SIZES}; do
  out="$(python "${CLI}" \
            --batch-size "${B}" \
            --seq-length "${SEQ_LENGTH}" \
            --num-layers "${NUM_LAYERS}" \
            --num-kv-heads "${NUM_KV_HEADS}" \
            --warmup "${WARMUP}" \
            --iters "${ITERS}" 2>/dev/null)" || out="FAIL"
  printf '%-6s %-12s\n' "${B}" "${out}"
done
