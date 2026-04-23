#!/bin/bash
# GPT-5.2 + SmartMem V4 on easy_test + hard_test, with NATURAL retrieval (no replay).
# V4 memory now has 7 caveats: 4 original (Claude V1/V2 + Gemini V1) + 3 new GPT V2-derived:
#   - idx 915 battlefin Ridge: don't apply to retail forecasting
#   - idx 40  LANL Ridge: don't generalize to retail forecasting
#   - idx 911 DontGetKicked LR: don't fall back to LR on large imbalanced fraud
#
# No replay — GPT's natural retrieval pattern applies.
# No opt-out.
# Hyperparams match GPT V2: num-drafts=3, max-turns=12, max-tokens=4096.
#
# Usage:
#   export LITELLM_API_KEY=... OPENAI_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=... OPENAI_API_KEY=... bash /data/fnie/qixin/DSGym/scripts/run_gpt52_smartmem_v4_gpt_memtest.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

# NO replay, NO opt-out
unset DSGYM_REPLAY_SAMPLES
unset DSGYM_OPT_OUT_TASKS

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_v4snap_enriched.json  # V4 + 3 GPT-derived caveats
MODEL=openai/gpt-5.2

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
    curl -s http://localhost:5000/status | head -c 300 || true
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

run_eval "gpt52_smartmem_easy_test_v4" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "gpt52_smartmem_hard_test_v4" "dspredict-hard-test" "docker-dspredict-hard.yml"
echo "GPT-5.2 V4 (natural retrieval) all done at $(date)"
