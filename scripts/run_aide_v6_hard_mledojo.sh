#!/bin/bash
# V6 distillation: hard + mledojo + hard_swap on node 03 (executor-mle)

set -e

export PYTHONUNBUFFERED=1
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
EVAL_CMD="$DSGYM/.venv/bin/dsgym eval"
LOG_DIR="$DSGYM/logs"

# --- hard ---
echo "=============================="
echo "Starting hard at $(date)"
echo "=============================="
$EVAL_CMD \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-hard \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --best-node-strategy best \
    --memory-path "$DSGYM/evaluation_results/aide_v6_hard/cross_task_memory.json" \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --output-dir "$DSGYM/evaluation_results/aide_v6_hard" \
    2>&1 | tee "$LOG_DIR/aide_v6_hard.out"

echo "hard finished at $(date)"
sudo docker restart executors_manager_1
sleep 5

# --- mledojo ---
echo "=============================="
echo "Starting mledojo at $(date)"
echo "=============================="
$EVAL_CMD \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-mledojo \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --best-node-strategy best \
    --memory-path "$DSGYM/evaluation_results/aide_v6_mledojo/cross_task_memory.json" \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --output-dir "$DSGYM/evaluation_results/aide_v6_mledojo" \
    2>&1 | tee "$LOG_DIR/aide_v6_mledojo.out"

echo "mledojo finished at $(date)"
sudo docker restart executors_manager_1
sleep 5

# --- hard_swap ---
echo "=============================="
echo "Starting hard_swap at $(date)"
echo "=============================="
$EVAL_CMD \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-hard-swap \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --best-node-strategy best \
    --memory-path "$DSGYM/evaluation_results/aide_v6_hard_swap/cross_task_memory.json" \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --output-dir "$DSGYM/evaluation_results/aide_v6_hard_swap" \
    2>&1 | tee "$LOG_DIR/aide_v6_hard_swap.out"

echo "All done at $(date)"
