#!/bin/bash
# gpt5.2 + ROUND-2 Δ-only memory (no new caveats this run).
#
# Memory: cross_task_memory_teacher_v5_mixed_gpt_round2_enriched.json
#   = 1223 round_origin=0 (M₀ + 6 GPT-derived caveats from v4)
#   + 75 round_origin=1 (delta_gpt52_new from gpt5.2 v2 runs)
#   + 56 round_origin=2 (delta_gpt52_round2 from gpt5.2 mixed round-1 runs)
#   Total 1354 entries; same-challenge round_origin>=1 entries are admitted
#   by the round-aware filter (capped at max_per_task=2).
#
# This is the gpt5.2 analog of sonnet46_round2_*_test (Δ-only flywheel turn).
# Used to test whether Δ-only round-2 also regresses for gpt5.2 (cross-model
# replication of the "must pair Δ with caveat" finding from Sonnet).
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
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixed_gpt_round2_enriched.json
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
