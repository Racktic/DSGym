#!/bin/bash
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"
export PYTHONUNBUFFERED=1
cd /data/fnie/qixin/DSGym

BASE_DIR="/data/fnie/qixin/DSGym/evaluation_results/aide_v5_swap"
MEMORY_PATH="$BASE_DIR/cross_task_memory.json"

for ROUND in 2 3 4; do
    OUTPUT_DIR="$BASE_DIR/run${ROUND}"
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    echo ">>> [Round ${ROUND}/4] Starting AIDE V5 best on swap split"
    echo ">>> Memory: $MEMORY_PATH"
    echo ">>> Output: $OUTPUT_DIR"
    echo ">>> Time: $(date)"

    /data/fnie/qixin/DSGym/.venv/bin/dsgym eval \
        --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
        --dataset dspredict-swap \
        --backend litellm \
        --agent aide \
        --num-drafts 5 \
        --max-turns 20 \
        --max-workers 8 \
        --best-node-strategy best \
        --memory-version v5 \
        --memory-path "$MEMORY_PATH" \
        --output-dir "$OUTPUT_DIR"

    echo ">>> [Round ${ROUND}/4] Done at $(date)"

    if [ "$ROUND" -lt 4 ]; then
        echo ">>> Restarting manager for next round..."
        sudo docker restart executors-manager-1
        sleep 30
    fi
done

echo ">>> ALL 3 REMAINING ROUNDS DONE at $(date)"
