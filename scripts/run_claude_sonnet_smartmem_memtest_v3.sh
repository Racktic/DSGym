#!/bin/bash
# Claude Sonnet 4.6 + SmartMem (collective memory v3) on easy_test + hard_test.
#
# v3 change (on top of v2): memory file now includes scope_caveats on selected
# entries (walmart-recruiting idx 239, m5-forecasting-accuracy idx 23) targeting
# three failure patterns documented in dev_logs/0415.md §14:
#   - store-sales v2: FE-timeout loop caused by lag_52/lag_104 on 3M rows
#   - recruit-restaurant v1: lag NaN when test horizon > training history lag window
#   - store-sales v1: recursive forecasting error compounding vs non-recursive validation
#
# Retriever code itself is unchanged from v2; only the memory JSON and the
# format_for_prompt renderer (which prints a ⚠ Scope caveat line under the insight)
# differ. This lets us A/B scope-caveat effectiveness vs v2 directly.
#
# Cost estimate: ~$12 per split, ~$20-24 total.
#
# Usage:
#   export LITELLM_API_KEY=...
#   export OPENAI_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=$LITELLM_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_claude_sonnet_smartmem_memtest_v3.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
# v3 memory: original v5 enriched + scope_caveats on 2 entries.
# Embedding .npy is an unchanged copy of v5 (same stem so aide_agent auto-detects it).
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_caveat_enriched.json
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

run_eval "claude_sonnet_smartmem_easy_test_v3" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "claude_sonnet_smartmem_hard_test_v3" "dspredict-hard-test" "docker-dspredict-hard.yml"

echo "All done at $(date)"
