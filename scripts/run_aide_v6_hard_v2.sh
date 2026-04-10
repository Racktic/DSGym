#!/bin/bash
# V6 hard v2: full 54 tasks re-run with container cleanup fix
# Run on node with hard containers (e.g., 03)
# Usage: cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_aide_v6_hard_v2.sh

set -e

export PYTHONUNBUFFERED=1
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym

sudo docker restart executors_manager_1 2>/dev/null || sudo docker restart executors-manager-1 2>/dev/null || true
sleep 5

echo "=============================="
echo "Starting V6 hard v2 at $(date)"
echo "=============================="
dsgym eval \
    --model together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --dataset dspredict-hard \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --best-node-strategy best \
    --memory-path "$DSGYM/evaluation_results/aide_v6_hard_v2/cross_task_memory.json" \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --output-dir "$DSGYM/evaluation_results/aide_v6_hard_v2" \
    2>&1 | tee "$DSGYM/logs/aide_v6_hard_v2.out"

echo "V6 hard v2 finished at $(date)"
