#!/bin/bash
# GPT-5.2 baseline (no memory) on dspredict-hard-test.
# Inference via LiteLLM proxy — no local GPU.
# Hyperparams match Claude/Gemini memtest v6/v7 for apples-to-apples comparison.
#
# Usage:
#   export LITELLM_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=$LITELLM_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_gpt52_baseline_hard_test.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MODEL=openai/gpt-5.2
TAG=gpt52_baseline_hard_test

cd $DSGYM
source .venv/bin/activate

echo "=============================="
echo "[${TAG}] bring up docker-dspredict-hard.yml at $(date)"
echo "=============================="
cd $DSGYM/executors
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml up -d
sleep 15
curl -s http://localhost:5000/status | head -c 300 || true
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
    --api-key $LITELLM_API_KEY \
    --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
    --num-drafts 3 \
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir evaluation_results/${TAG} \
    2>&1 | tee logs/${TAG}.out
echo "${TAG} finished at $(date)"
