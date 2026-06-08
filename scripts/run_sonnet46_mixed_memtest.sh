#!/bin/bash
# Claude Sonnet 4.6 + MIXED memory (caveat + delta, GECM flywheel round-1).
#
# Memory composition:
#   - 1223 entries from cross_task_memory_teacher_v5_caveat_enriched.json
#     (M₀ teacher base, 11 of them carry scope_caveats). All tagged
#     round_origin=0 in the mixed file.
#   - 63 Sonnet-derived delta insights from delta_claude_new_enriched.json
#     (extracted from claude_sonnet_smartmem_*_test_v2 trajectories the same
#     offline way M₀ was built). All tagged round_origin=1.
#   - Total 1286 entries; embeddings stacked in matching order.
#
# Retriever change (smart_retriever.py _build_pool, see commented block there):
# the same-challenge exclusion is now round-aware. M₀ (round_origin=0) same-
# challenge entries stay hidden (every prior experiment did this); but
# round_origin>=1 same-challenge entries are admitted (capped at
# max_per_task=2 in the 15-slot stable pool). This is the entry-level analog
# of Dynamic Cheatsheet's M_i={1..i-1} safeguard, lifted to round granularity.
#
# Everything else mirrors run_sonnet46_delta_memtest.sh: agent via Anthropic
# native (no base-url), retriever metadata/embeddings via OpenAI, Bearer KGAT
# Kaggle scoring.
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
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixed_claude_enriched.json
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
