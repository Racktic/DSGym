#!/bin/bash
# Claude Sonnet 4.6 + SmartMem (collective memory v2) on easy_test + hard_test.
#
# v2 retriever change (commit aae9212): re-ranking formula simplified from
#   s = 0.8 * cosine + 0.2 * score_norm   -->   s = cosine
# Everything else matches scripts/run_claude_sonnet_memtest.sh so the only
# variable vs the v1 runs (evaluation_results/claude_sonnet_smartmem_{easy,hard}_test)
# is the retriever re-ranking.
#
# Each split dumps per-retrieval cosines via DSGYM_RETRIEVAL_LOG, so we can
# inspect the cosine distribution to pick a similarity threshold.
#
# Cost estimate: ~$12 per split (10 tasks x ~250k input + 27k output at $3/$15 per 1M).
# Total ~$20-24.
#
# Usage:
#   export LITELLM_API_KEY=...
#   export OPENAI_API_KEY=...
#   ssh research-common-21 "LITELLM_API_KEY=$LITELLM_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_claude_sonnet_smartmem_memtest_v2.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_enriched.json
MODEL=openai/claude-sonnet-4.6

cd $DSGYM
source .venv/bin/activate

run_eval() {
    local tag="$1"
    local dataset="$2"
    local compose="$3"

    # Retriever will append one JSONL line per (task, action) retrieval here.
    export DSGYM_RETRIEVAL_LOG=$DSGYM/logs/retrieval_debug_${tag}.jsonl
    : > "$DSGYM_RETRIEVAL_LOG"

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
        --api-key $LITELLM_API_KEY \
        --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
        --num-drafts 3 \
        --max-turns 12 \
        --max-tokens 4096 \
        --max-workers 8 \
        --best-node-strategy best \
        --output-dir evaluation_results/${tag} \
        2>&1 | tee logs/${tag}.out
    echo "${tag} finished at $(date)"
    echo "Retrieval debug log: $DSGYM_RETRIEVAL_LOG ($(wc -l < $DSGYM_RETRIEVAL_LOG) lines)"
}

run_eval "claude_sonnet_smartmem_easy_test_v2" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "claude_sonnet_smartmem_hard_test_v2" "dspredict-hard-test" "docker-dspredict-hard.yml"

echo "All done at $(date)"
