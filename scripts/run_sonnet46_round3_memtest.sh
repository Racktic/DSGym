#!/bin/bash
# Sonnet 4.6 + ROUND-3 flywheel turn (Δ + caveat both updated from round-2 trajectories).
#
# Memory: cross_task_memory_teacher_v5_round3_caveat_enriched.json
#   = M_2 (1333 entries, 13 caveats) + 40 round-3 delta entries (round_origin=3,
#     from Sonnet round-2 caveat trajectories) + 6 NEW caveats attributed to
#     regressions observed in Sonnet round-2 caveat vs round-1 mixed:
#       idx 1242 (s5e3 R1 CatBoost) — "iterations=2000 + 10-fold can crash kernel"
#       idx 1252 (mens-march R1 SOS) — "fillna(median) on StdR before scaling"
#       idx 1265 (recruit R1 store×month, +follow-up) — "median blend fallback also fails; use LightGBM+lag+reservation"
#       idx 1297 (s3e19 R2 year-trend) — "SMAPE val 11 → Kaggle 0: verify submission format"
#       idx 1322 (mens-march R2 nonlinear) — "NaN-safe imputer needed before nonlinear features"
#       idx 1325 (novozymes R2 source TE) — "val Spearman 0.6 → Kaggle 0.2: source-specific bias overfit"
#
# Caveat mix is intentionally broader (5+ as requested by user): some are pure
# attribution (recruit, novozymes), some are practical sanity-check guidance
# (s3e19 submission format, mens-march NaN handling, s5e3 memory limits).
# Total 18 scope_caveats. 1373 entries; embeddings unchanged.
set -e
export PYTHONUNBUFFERED=1

ANTHROPIC_KEY="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"
OPENAI_KEY="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"
export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
export OPENAI_API_KEY="$OPENAI_KEY"
export LITELLM_API_KEY="$OPENAI_KEY"
export LITELLM_BASE_URL="https://api.openai.com/v1"
export DSGYM_METADATA_MODEL="gpt-4o-mini"
unset KAGGLE_USERNAME KAGGLE_KEY
export KAGGLE_API_TOKEN="KGAT_eacf487e63f24bebc62cdd0bcd52f598"

DSGYM=/srv/home/bohanlyu/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_round3_caveat_enriched.json
MODEL=anthropic/claude-sonnet-4-6
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
        --api-key "$ANTHROPIC_KEY" \
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
