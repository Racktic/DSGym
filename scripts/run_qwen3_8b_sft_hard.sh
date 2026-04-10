#!/bin/bash
# Evaluate Qwen3-8B SFT V6 checkpoint-30 on hard, mledojo, swap
# Step 1: Switch to hard containers
#   cd /data/fnie/qixin/DSGym/executors
#   sudo docker compose -f docker-dspredict-easy.yml down
#   sudo docker compose -f docker-dspredict-hard.yml up -d
# Step 2:
#   cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_qwen3_8b_sft_v6_hard_mledojo_swap.sh

set -e
export PYTHONUNBUFFERED=1

cd /data/fnie/qixin/DSGym
MODEL=/data/fnie/LLaMA-Factory/saves/qwen3-8b-distill_all_truncAF_swap_only_round1/full/sft/checkpoint-5

# --- hard ---
echo "=============================="
echo "Starting hard at $(date)"
echo "=============================="
dsgym eval \
    --model $MODEL \
    --dataset dspredict-hard \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --no-task-memory \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --temperature 0.7 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/qwen3_8b_sft_distill_all_truncAF_hard_swap_only_round1_neft5-wd01 \
    2>&1 | tee logs/qwen3_8b_sft_distill_all_truncAF_hard_swap_only_round1_neft5-wd01.out

echo "hard finished at $(date)"

