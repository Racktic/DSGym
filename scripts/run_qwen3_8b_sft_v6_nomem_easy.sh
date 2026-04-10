#!/bin/bash
# Evaluate Qwen3-8B SFT V6 checkpoint on dspredict-easy (no memory, bare AIDE)
# Usage: cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_qwen3_8b_sft_v6_nomem_easy.sh

set -e
export PYTHONUNBUFFERED=1

cd /data/fnie/qixin/DSGym

dsgym eval \
    --model /data/fnie/LLaMA-Factory/saves/qwen3-8b-aide-v6-combined/full/sft/checkpoint-50 \
    --dataset dspredict-easy \
    --backend multi-vllm \
    --agent aide \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/qwen3_8b_sft_v6_nomem_easy \
    2>&1 | tee logs/qwen3_8b_sft_v6_nomem_easy.out
