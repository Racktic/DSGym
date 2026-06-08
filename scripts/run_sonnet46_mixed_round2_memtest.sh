#!/bin/bash
# Sonnet 4.6 + MIXED ROUND-2 memory (GECM flywheel, 2nd turn).
#
# Memory composition:
#   - 1223 entries with round_origin=0 (M₀ + 11 caveats, the base)
#   - 63 entries with round_origin=1 (delta_claude_new from Sonnet v2 runs — round-1 delta)
#   - 47 entries with round_origin=2 (delta_claude_round2 from Sonnet mixed#2 runs — round-2 delta)
#   - Total 1333 entries.
#
# Same round-aware filter as round-1 mixed: same-challenge entries with
# round_origin>=1 are admitted (capped at max_per_task=2). So at this round
# both round-1 AND round-2 same-challenge entries can be retrieved — earlier
# rounds' lessons survive into the next round's prompt.
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
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixed_claude_round2_enriched.json
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
