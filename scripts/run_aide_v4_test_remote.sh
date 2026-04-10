#!/bin/bash
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"
cd /data/fnie/qixin/DSGym

OUTPUT_DIR="/data/fnie/qixin/DSGym/evaluation_results/aide_v4_best_test"
mkdir -p "$OUTPUT_DIR"

.venv/bin/dsgym eval \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 2 \
    --limit 2 \
    --best-node-strategy best \
    --memory-path "$OUTPUT_DIR/cross_task_memory.json" \
    --output-dir "$OUTPUT_DIR"
