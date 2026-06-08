#!/bin/bash
# gpt5.2 + R1-ATTRIBUTION caveat memory.
#
# Memory: cross_task_memory_teacher_v5_mixed_gpt_r1attrib_enriched.json
#   = mixed_gpt (1298 entries, 6 GPT-derived caveats) + 8 NEW caveats from
#     gpt5.2 round-1 mixed regression attribution (per skill
#     gecm-caveat-attribution):
#       idx 619  (M₀ s4e1) — titanic: don't generalize bank-churn interactions to <2k rows
#       idx 737  (M₀ s3e11) — s3e19: don't use random 5-fold KFold on time-series
#       idx 505  (M₀ s4e11) — s5e3: don't stack class_weight=balanced on regularized LR
#       idx 506  (M₀ s3e24) — s4e1: don't drop CatBoost for LightGBM at noise-level AUC
#       idx 862  (M₀ quora-insincere) — nlp: don't apply quora threshold tuning to <15k text
#       idx 1261 (R1 home-data) — house-prices: don't use Huber loss on <3k rows
#       idx 1232 (R1 s3e25 self) — s3e25: don't widen max_leaf_nodes without tightening min_samples_leaf
#       idx 1291 (R1 ventilator) — recruit-restaurant: don't generalize Ridge from physics signals to panel demand
#
# 5/8 caveats attached to M₀ teacher entries (round_origin=0), 3/8 to R1 delta
# entries — consistent with the principle that retrieval rate determines which
# entries are most-likely-misapplied (M₀ has highest cosine rate).
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
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixed_gpt_r1attrib_enriched.json
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
