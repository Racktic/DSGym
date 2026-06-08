#!/bin/bash
# GPT-5.2 + MIXED memory (caveat + delta, GECM flywheel round-1, GPT line).
#
# Memory composition:
#   - 1223 entries from cross_task_memory_teacher_v5_v4snap_enriched.json
#     (V4 snapshot: M₀ + 6 GPT-derived scope_caveats — the same caveat base
#     used by gpt52_smartmem_*_v4). All tagged round_origin=0 in the mixed file.
#   - 75 GPT-derived delta insights from delta_gpt52_new_enriched.json
#     (extracted from gpt52_smartmem_*_test_v2 trajectories the same offline
#     way M₀ was built). All tagged round_origin=1.
#   - Total 1298 entries.
#
# Retriever change (smart_retriever.py _build_pool, see commented block there):
# the same-challenge exclusion is now round-aware. M₀ (round_origin=0) same-
# challenge entries stay hidden (every prior experiment did this); but
# round_origin>=1 same-challenge entries are admitted (capped at
# max_per_task=2 in the 15-slot stable pool). Entry-level analog of Dynamic
# Cheatsheet's M_i={1..i-1} safeguard, lifted to round granularity.
#
# All other config mirrors run_gpt52_delta_memtest.sh exactly: agent via
# direct OpenAI (anthropic key not used here), retriever metadata/embedding
# via OpenAI, Bearer KGAT Kaggle scoring.
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
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixed_gpt_enriched.json
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
