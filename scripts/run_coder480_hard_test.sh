#!/bin/bash
# Qwen3-Coder-480B baseline on dspredict-hard-test (no memory)
# Inference via Together AI → no GPU needed, any node with docker works.
# Usage:
#   export TOGETHER_API_KEY=...
#   ssh research-common-19 "TOGETHER_API_KEY=$TOGETHER_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_coder480_hard_test.sh"

set -e
export PYTHONUNBUFFERED=1
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MODEL=together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
TAG=coder480_hard_test

cd $DSGYM
source .venv/bin/activate

echo "=============================="
echo "[${TAG}] bring up docker-dspredict-hard.yml at $(date)"
echo "=============================="
cd $DSGYM/executors
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml up -d
sudo docker restart executors-manager-1 2>/dev/null || sudo docker restart executors_manager_1 2>/dev/null || true
sleep 15
curl -s http://localhost:5000/status | head -c 500 || true
echo

cd $DSGYM
echo "Starting ${TAG} at $(date)"
dsgym eval \
    --model $MODEL \
    --dataset dspredict-hard-test \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --no-task-memory \
    --num-drafts 3 \
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir evaluation_results/${TAG} \
    2>&1 | tee logs/${TAG}.out
echo "${TAG} finished at $(date)"
