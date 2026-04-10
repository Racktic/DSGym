#!/bin/bash
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
cd /data/fnie/qixin/DSGym

# ========== V5 best ==========
OUTPUT_DIR="/data/fnie/qixin/DSGym/evaluation_results/aide_v5_best_easy"
mkdir -p "$OUTPUT_DIR"

echo ">>> [1/2] Starting AIDE V5 (best-node) full easy split"
echo ">>> Time: $(date)"

stdbuf -oL .venv/bin/dsgym eval \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --best-node-strategy best \
    --memory-version v5 \
    --memory-path "$OUTPUT_DIR/cross_task_memory.json" \
    --output-dir "$OUTPUT_DIR"

echo ">>> [1/2] V5 best done at $(date)"

# Restart manager to release containers
echo ">>> Restarting manager..."
sudo docker restart executors-manager-1
sleep 30

# ========== V5 latest ==========
OUTPUT_DIR="/data/fnie/qixin/DSGym/evaluation_results/aide_v5_latest_easy"
mkdir -p "$OUTPUT_DIR"

echo ">>> [2/2] Starting AIDE V5 (latest-node) full easy split"
echo ">>> Time: $(date)"

stdbuf -oL .venv/bin/dsgym eval \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --best-node-strategy latest \
    --memory-version v5 \
    --memory-path "$OUTPUT_DIR/cross_task_memory.json" \
    --output-dir "$OUTPUT_DIR"

echo ">>> [2/2] V5 latest done at $(date)"
echo ">>> ALL DONE"
