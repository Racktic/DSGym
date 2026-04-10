#!/bin/bash
# Distill mledojo with Together AI Qwen3-Coder-480B
# Usage: cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_distill_coder480_mledojo.sh
# Requires: executor-mle containers up via docker-dspredict-mledojo.yml

set -e
export PYTHONUNBUFFERED=1
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym

# Reset container manager
sudo docker restart executors-manager-1 2>/dev/null || sudo docker restart executors_manager_1 2>/dev/null || true
sleep 5

dsgym eval \
    --model together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --dataset dspredict-mledojo \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --num-drafts 3 \
    --max-turns 12 \
    --max-tokens 2048 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir "$DSGYM/evaluation_results/distill_coder480_mledojo" \
    2>&1 | tee "$DSGYM/logs/distill_coder480_mledojo.out"
