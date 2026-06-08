#!/bin/bash
# Sonnet 4.6 + MIXED-LINE step5 memory.
#
# This is the "caveat update from the very first step" ablation line (NOT branched
# off mixed). Memory = M0 (1223 entries) + caveat1 (12 scope_caveats, the step2
# base = cross_task_memory_teacher_v5_caveat_enriched.json) + caveat2 (3 new scope
# caveats attributed from the PURE-caveat step2 run claude_sonnet_smartmem_{easy,
# hard}_test_v5, attached to the mis-applied M0 entries + their top-recalled
# accomplice). No new entries are added (pure scope refinement).
#
# Total 1223 entries, 18 scope_caveats. New caveats:
#   - playground-series-s3e13 regression -17.7  -> idx 752 (s3e26) + 445 (s4e2)
#   - playground-series-s4e1   regression -4.6   -> idx 782 (s5e3)  + 473 (s3e3)
#   - ventilator-pressure-pred regression -4.0   -> idx 23 (m5)     + 915 (battlefin)
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
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixedline_step5_claude_enriched.json
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
