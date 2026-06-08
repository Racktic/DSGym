#!/bin/bash
# Sonnet 4.6 + ROUND-2 with auto-derived caveats (real flywheel turn).
#
# Memory: cross_task_memory_teacher_v5_mixed_claude_round2_caveat_enriched.json
#   = M_2 round-2 memory + 2 NEW caveats attached to round-1 delta entries that
#     were attributed to the round-1 regressions:
#       - idx 1223 (digit-recognizer round-1 entry):
#         warns "plain CNN at 0.993 val is NOT enough; M₀ SEResNet+TTA hits 0.998"
#         (Sonnet round-1 mixed regressed 96.5 → 74.3 pct)
#       - idx 1265 (recruit-restaurant round-1 entry):
#         warns "store×month target encoding validation drop does NOT transfer to test"
#         (Sonnet round-1 mixed regressed 33.8 → 19.8 pct: val 0.482→0.393 but Kaggle dropped)
#
# Caveats were authored by inspecting Sonnet baseline vs round-1 mixed trajectories,
# the same way the original 11 caveats were generated. Total 13 scope_caveats now
# (11 original + 2 new). Memory entries stay at 1333; embeddings unchanged.
#
# This is the test of "Δ + caveat together" — pairing the round-2 delta append with
# attribution-derived caveats on the round-1 entries that mis-applied.
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
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixed_claude_round2_caveat_enriched.json
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
