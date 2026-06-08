#!/bin/bash
# gpt5.2 + ROUND-2 Δ+caveat memory (real flywheel turn, GPT line).
#
# Memory: cross_task_memory_teacher_v5_mixed_gpt_round2_caveat_enriched.json
#   = M_2 gpt round-2 (1354 entries, 6 GPT-derived caveats) + 1 NEW caveat:
#       - idx 1339 (recruit-restaurant round-2 delta entry, "store×day-of-year mean log-visitors"):
#         warns "this exact technique caused val RMSLE 0.484→0.252 but Kaggle pct 48.5→9.4 in round-1 mixed"
#
# Caveat is the gpt5.2 analog of Sonnet's recruit caveat (same temporal-leakage
# pattern via target encoding on time-series). Caveats targeting other gpt5.2
# round-2 regressions (titanic, s5e3) were not derivable — the agent chose
# methods that no memory entry suggested, so no entry to caveat.
set -e
export PYTHONUNBUFFERED=1

OPENAI_KEY="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"
export OPENAI_API_KEY="$OPENAI_KEY"
export LITELLM_API_KEY="$OPENAI_KEY"
export LITELLM_BASE_URL="https://api.openai.com/v1"
export DSGYM_METADATA_MODEL="gpt-4o-mini"
unset KAGGLE_USERNAME KAGGLE_KEY
export KAGGLE_API_TOKEN="KGAT_eacf487e63f24bebc62cdd0bcd52f598"

DSGYM=/srv/home/bohanlyu/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixed_gpt_round2_caveat_enriched.json
MODEL=openai/gpt-5.2
cd $DSGYM
source .venv/bin/activate

run_eval() {
    local tag="$1"; local dataset="$2"; local manager="$3"
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
        --api-key "$OPENAI_KEY" \
        --base-url https://api.openai.com/v1 \
        --manager-url "$manager" \
        --num-drafts 3 \
        --max-turns 12 \
        --max-tokens 4096 \
        --max-workers 8 \
        --best-node-strategy best \
        --output-dir evaluation_results/${tag} \
        2>&1 | tee logs/${tag}.out
    echo "${tag} finished at $(date)"
}

run_eval "$1" "$2" "$3"
