#!/bin/bash
# Claude Sonnet 4.6 + SmartMem on hard_test ONLY.
# Purpose: validate score_norm removal + collect cosine distribution
# (DSGYM_RETRIEVAL_LOG) to pick a similarity threshold.
#
# Cost estimate: ~$12 (10 tasks × ~250k input + 27k output × $3/$15 per 1M)
#
# Usage:
#   export LITELLM_API_KEY=...
#   export OPENAI_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=$LITELLM_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_claude_sonnet_smartmem_hard_only.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_enriched.json
MODEL=openai/claude-sonnet-4.6
TAG=claude_sonnet_smartmem_hard_test_v2

# Retriever will append one JSONL line per (task, action) retrieval here.
export DSGYM_RETRIEVAL_LOG=$DSGYM/logs/retrieval_debug_${TAG}.jsonl
# Truncate any prior content so each run starts clean.
: > "$DSGYM_RETRIEVAL_LOG"

cd $DSGYM
source .venv/bin/activate

cd $DSGYM/executors
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml up -d
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
    --no-task-memory \
    --memory-path $MEMORY \
    --no-cross-memory-write \
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
echo "Retrieval debug log: $DSGYM_RETRIEVAL_LOG ($(wc -l < $DSGYM_RETRIEVAL_LOG) lines)"
