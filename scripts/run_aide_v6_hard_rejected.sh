#!/bin/bash
# V6 hard rejected tasks re-run: verify container pollution fix
# Only re-runs the 24 tasks that were Kaggle-rejected due to container pollution

set -e

export PYTHONUNBUFFERED=1
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
EVAL_CMD="$DSGYM/.venv/bin/dsgym eval"
LOG_DIR="$DSGYM/logs"

# Restart manager to reset container states (use docker compose v2 naming)
sudo docker restart executors-manager-1 2>/dev/null || sudo docker restart executors_manager_1 2>/dev/null || true
sleep 5

echo "=============================="
echo "Starting hard_rejected re-run at $(date)"
echo "=============================="
$EVAL_CMD \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-hard-rejected \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --best-node-strategy best \
    --memory-path "$DSGYM/evaluation_results/aide_v6_hard/cross_task_memory.json" \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --output-dir "$DSGYM/evaluation_results/aide_v6_hard_rejected" 

echo "hard_rejected finished at $(date)"
