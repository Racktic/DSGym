#!/bin/bash
# V6 swap 4-round distillation on node 26
# Shared cross-task memory across rounds

set -e

export PYTHONUNBUFFERED=1
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
EVAL_CMD="$DSGYM/.venv/bin/dsgym eval"
OUTPUT_BASE="$DSGYM/evaluation_results/aide_v6_swap"
MEMORY_PATH="$OUTPUT_BASE/cross_task_memory.json"
LOG_DIR="$DSGYM/logs"

mkdir -p "$OUTPUT_BASE"

for round in 1 2 3 4; do
    echo "=============================="
    echo "Starting round $round at $(date)"
    echo "=============================="

    $EVAL_CMD \
        --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
        --dataset dspredict-swap \
        --backend litellm \
        --agent aide \
        --memory-version v6 \
        --best-node-strategy best \
        --memory-path "$MEMORY_PATH" \
        --num-drafts 5 \
        --max-turns 20 \
        --max-workers 8 \
        --output-dir "$OUTPUT_BASE/run${round}" \
        2>&1 | tee "$LOG_DIR/aide_v6_swap_run${round}.out"

    echo "Round $round finished at $(date)"

    # Restart manager between rounds to release all containers
    sudo docker restart executors-manager-1
    sleep 5
done

echo "All 4 rounds completed at $(date)"
