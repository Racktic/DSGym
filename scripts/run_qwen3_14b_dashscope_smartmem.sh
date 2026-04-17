#!/bin/bash
# Qwen3-14B via Alibaba DashScope (OpenAI-compatible) — SmartMem variant.
# Runs easy_test + hard_test with the AIDE agent and SmartRetriever-backed
# cross-task memory (read-only; enriched memory + embeddings built offline).
#
# DashScope serves the model; OpenAI embeddings are still needed by
# SmartRetriever to embed task descriptions (independent of DashScope).
#
# Usage:
#   export DASHSCOPE_API_KEY=sk-...
#   export OPENAI_API_KEY=sk-...
#   export LITELLM_API_KEY=...           # required by SmartRetriever init
#   bash /data/fnie/qixin/DSGym/scripts/run_qwen3_14b_dashscope_smartmem.sh

set -e
export PYTHONUNBUFFERED=1

export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_enriched.json
MODEL=qwen3-14b
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

cd $DSGYM
source .venv/bin/activate

run_eval() {
    local tag="$1"
    local dataset="$2"
    local compose="$3"
    local split="$4"

    echo "=============================="
    echo "[${tag}] bring up ${compose} at $(date)"
    echo "=============================="
    cd $DSGYM/executors
    sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
    sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
    sudo docker compose -f $compose up -d
    sleep 15
    curl -s http://localhost:5000/status | head -c 500 || true
    echo

    # Retriever writes one JSONL line per (task, action) retrieval.
    export DSGYM_RETRIEVAL_LOG=$DSGYM/logs/retrieval_debug_qwen3_14b_smartmem_${split}.jsonl
    : > "$DSGYM_RETRIEVAL_LOG"

    cd $DSGYM
    echo "Starting ${tag} at $(date)"
    dsgym eval \
        --model $MODEL \
        --dataset $dataset \
        --backend litellm \
        --agent aide \
        --memory-version v6 \
        --no-task-memory \
        --memory-path $MEMORY \
        --no-cross-memory-write \
        --api-key $DASHSCOPE_API_KEY \
        --base-url $BASE_URL \
        --num-drafts 3 \
        --max-turns 12 \
        --max-tokens 4096 \
        --max-workers 4 \
        --no-think \
        --best-node-strategy best \
        --output-dir evaluation_results/${tag} \
        2>&1 | tee logs/${tag}.out
    echo "${tag} finished at $(date)"
    echo "Retrieval debug log: $DSGYM_RETRIEVAL_LOG ($(wc -l < $DSGYM_RETRIEVAL_LOG) lines)"
}

# run_eval "qwen3_14b_dashscope_smartmem_easy_test" "dspredict-easy-test" "docker-dspredict-easy.yml" "easy_test"
run_eval "qwen3_14b_dashscope_smartmem_hard_test" "dspredict-hard-test" "docker-dspredict-hard.yml" "hard_test"

echo "All done at $(date)"
