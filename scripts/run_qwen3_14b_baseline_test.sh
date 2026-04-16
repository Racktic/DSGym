#!/bin/bash
# Evaluate base Qwen3-14B (no SFT) on easy_test + hard_test as baseline
# Runs on node 19 directly via ssh.
# Usage: ssh research-common-19 "bash /data/fnie/qixin/DSGym/scripts/run_qwen3_14b_baseline_test.sh"

set -e
export PYTHONUNBUFFERED=1

DSGYM=/data/fnie/qixin/DSGym
MODEL=/data/fnie/qixin/models/Qwen3-14B
TAG=qwen3_14b_baseline

cd $DSGYM
source .venv/bin/activate

# --- easy_test ---
echo "=============================="
echo "[easy_test] switching to docker-dspredict-easy.yml at $(date)"
echo "=============================="
cd $DSGYM/executors
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-easy.yml up -d
sleep 10
curl -s http://localhost:5000/status | head -c 500 || true
echo

cd $DSGYM
echo "Starting easy_test at $(date)"
dsgym eval \
    --model $MODEL \
    --dataset dspredict-easy-test \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --no-task-memory \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --temperature 0.7 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/${TAG}_easy_test \
    2>&1 | tee logs/${TAG}_easy_test.out
echo "easy_test finished at $(date)"

# --- hard_test ---
echo "=============================="
echo "[hard_test] switching to docker-dspredict-hard.yml at $(date)"
echo "=============================="
cd $DSGYM/executors
sudo docker compose -f docker-dspredict-easy.yml down
sudo docker compose -f docker-dspredict-hard.yml up -d
sleep 10
curl -s http://localhost:5000/status | head -c 500 || true
echo

cd $DSGYM
echo "Starting hard_test at $(date)"
dsgym eval \
    --model $MODEL \
    --dataset dspredict-hard-test \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --no-task-memory \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --temperature 0.7 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/${TAG}_hard_test \
    2>&1 | tee logs/${TAG}_hard_test.out
echo "hard_test finished at $(date)"

echo "All done at $(date)"
