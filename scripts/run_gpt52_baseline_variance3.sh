#!/bin/bash
# GPT-5.2 baseline rerun on the 2 tasks that failed during the earlier baseline run
# because of an accidental docker teardown mid-eval.
# Tasks: ieee-fraud-detection, ventilator-pressure-prediction.
#
# Trajectories are written to a separate dir (_variance3). After this finishes,
# merge the 2 new trajectories into evaluation_results/gpt52_baseline_hard_test/
# (overwriting the prior failed trajectories) so the main baseline dir has 10/10.
#
# Usage:
#   export LITELLM_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=$LITELLM_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_gpt52_baseline_variance3.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MODEL=openai/gpt-5.2
TAG=gpt52_baseline_hard_variance3

cd $DSGYM
source .venv/bin/activate

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
    --dataset dspredict-hard-test-variance3 \
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
