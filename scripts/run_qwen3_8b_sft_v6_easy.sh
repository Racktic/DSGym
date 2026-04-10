#!/bin/bash
# Evaluate Qwen3-8B SFT V6 checkpoint on dspredict-easy (with V6 memory)
# Usage: cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_qwen3_8b_sft_v6_easy.sh

set -e
export PYTHONUNBUFFERED=1

cd /data/fnie/qixin/DSGym

dsgym eval \
    --model /data/fnie/LLaMA-Factory/saves/qwen3-8b-aide-v6-w_submission/full/sft/checkpoint-21 \
    --dataset dspredict-easy \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --memory-path evaluation_results/qwen3_8b_sft_v6_easy_w_submission_epoch3/cross_task_memory.json \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/qwen3_8b_sft_v6_easy_w_submission_epoch3 \
    2>&1 | tee logs/qwen3_8b_sft_v6_easy_w_submission_epoch3.out
