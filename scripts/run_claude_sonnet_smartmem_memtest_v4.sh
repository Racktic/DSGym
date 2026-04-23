#!/bin/bash
# Claude Sonnet 4.6 + SmartMem V4 on easy_test + hard_test.
#
# V4 = V3 caveat-enriched memory + V2 retrieval replay.
# Goal: eliminate retrieval-sampling variance so the remaining delta vs V2 is
# attributable purely to scope_caveat content (on entries V2 did retrieve:
# idx 239 walmart + idx 23 m5-forecasting-accuracy).
#
# Mechanism:
#   - Memory file: cross_task_memory_teacher_v5_caveat_enriched.json (same as V3)
#   - DSGYM_REPLAY_SAMPLES points SmartRetriever at v2_retrieval_replay.json;
#     each retrieve() call returns the exact entry indices V2's agent saw at the
#     matching (challenge, action, call_index) — bypasses random.sample entirely.
#
# Any task whose V2 retrieval didn't include idx 239/23 will render identically
# to V2; only store-sales / recruit-restaurant / house-prices see the caveat
# lines appended in the prompt.
#
# Cost estimate: ~$24 (10 hard + 8 easy tasks).
#
# Usage:
#   export LITELLM_API_KEY=...
#   export OPENAI_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=$LITELLM_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_claude_sonnet_smartmem_memtest_v4.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_caveat_enriched.json
REPLAY=$DSGYM/data/memory/v2_retrieval_replay.json
MODEL=openai/claude-sonnet-4.6

export DSGYM_REPLAY_SAMPLES="$REPLAY"

cd $DSGYM
source .venv/bin/activate

run_eval() {
    local tag="$1"; local dataset="$2"; local compose="$3"

    echo "=============================="
    echo "[${tag}] bring up ${compose} at $(date)"
    echo "=============================="
    cd $DSGYM/executors
    sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
    sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
    sudo docker compose -f $compose up -d
    sleep 10
    curl -s http://localhost:5000/status | head -c 500 || true
    echo

    export DSGYM_RETRIEVAL_LOG=$DSGYM/logs/retrieval_debug_${tag}.jsonl
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
}

run_eval "claude_sonnet_smartmem_easy_test_v4" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "claude_sonnet_smartmem_hard_test_v4" "dspredict-hard-test" "docker-dspredict-hard.yml"

echo "All done at $(date)"
