#!/bin/bash
# Gemma grid first (newly unblocked), then resume stage3c for the main grid.
cd /home/shavin/llm-cpu-compiler/pytorch
export SSH_AUTH_SOCK=/home/shavin/.ssh/agent.sock
echo "$(date -Is) CHAIN: gemma sweep" >> /tmp/chain.log
bash bench_2026-09-25/gemma_sweep.sh >> /tmp/chain.log 2>&1
echo "$(date -Is) CHAIN: stage3c resume" >> /tmp/chain.log
# plan file lives in /tmp and may have been cleared
[ -f /tmp/stage3_plan.txt ] || python bench_2026-09-25/stage3_plan.py \
    --mem-ceiling 99999999999 --out /tmp/stage3_plan.txt >> /tmp/chain.log 2>&1
bash bench_2026-09-25/stage3c.sh >> /tmp/chain.log 2>&1
echo "$(date -Is) CHAIN: ALLDONE" >> /tmp/chain.log
