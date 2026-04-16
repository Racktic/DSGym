#!/bin/bash
# Baseline Qwen3-14B (no SFT) + cross-task memory (SmartRetriever).
# Pool: 15 entries per (task, action); each turn randomly samples 3.
# Compare against qwen3_14b_baseline_* (no memory) to see if memory helps.

set -e
export PYTHONUNBUFFERED=1

# Keys (SmartRetriever needs BOTH):
#   LITELLM_API_KEY: Claude metadata classification via LiteLLM proxy
#   OPENAI_API_KEY : text-embedding-3-small direct (LiteLLM proxy has no embedding model)
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MODEL=/data/fnie/qixin/models/Qwen3-14B
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_enriched.json
TAG=qwen3_14b_baseline_smartmem

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
    --memory-path $MEMORY \
    --no-task-memory \
    --no-cross-memory-write \
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
    --memory-path $MEMORY \
    --no-task-memory \
    --no-cross-memory-write \
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
