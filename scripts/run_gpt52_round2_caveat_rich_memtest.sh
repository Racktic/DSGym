#!/bin/bash
# gpt5.2 + ROUND-2 Δ + 6 caveat memory (richer caveat attribution).
#
# Memory: cross_task_memory_teacher_v5_mixed_gpt_round2_caveat_rich_enriched.json
#   = M_2 gpt round-2 (1354 entries, 6 GPT-derived caveats) + 6 NEW caveats:
#       idx 1240 (s5e3 round-1 LR)   — "don't switch to deep tree models, val plateau but Kaggle drops"
#       idx 1250 (titanic round-1 GBC)— "don't push capacity beyond GBC n_est=800"
#       idx 1253 (ieee round-1 LR)   — "don't abandon identity merge / time-aware split"
#       idx 1267 (spaceship round-1 HGBC) — "don't add OOF GroupTransportRate leakage feature"
#       idx 1322 (digit round-2 RBF SVM)  — "don't add PCA before SVM, spatial info loss"
#       idx 1339 (recruit round-2 store×DOY) — "target encoding val drop doesn't transfer to test"
#
# Each caveat is derived from a gpt5.2 round-2 Δ-only regression case (≥10 pct
# drop vs baseline or round-1 mixed), attributing the failure to a memory entry
# that either (a) recommended the failing technique, or (b) describes the
# successful round-1 approach that the round-2 agent abandoned.
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
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_mixed_gpt_round2_caveat_rich_enriched.json
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
