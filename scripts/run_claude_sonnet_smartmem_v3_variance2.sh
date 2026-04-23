#!/bin/bash
# V3 variance re-run: ONLY spaceship-titanic + mens-march-mania-2022 on hard_test.
# Purpose: isolate whether V3's big drop on these 2 tasks vs V2 is LLM sampling
# variance (re-run lands back near V2 / V3 mean) or a persistent effect.
#
# Same memory, same retriever, same everything as run_claude_sonnet_smartmem_memtest_v3.sh,
# just running on the 2-task subset hard_test_variance2.json.
#
# Usage:
#   export LITELLM_API_KEY=...
#   export OPENAI_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=$LITELLM_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_claude_sonnet_smartmem_v3_variance2.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_caveat_enriched.json
MODEL=openai/claude-sonnet-4.6
TAG=claude_sonnet_smartmem_hard_test_v3_variance2

export DSGYM_RETRIEVAL_LOG=$DSGYM/logs/retrieval_debug_${TAG}.jsonl
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
    --dataset dspredict-hard-test-variance2 \
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
