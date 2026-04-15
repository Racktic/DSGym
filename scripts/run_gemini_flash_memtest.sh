#!/bin/bash
# Collective intelligence experiment:
# Does Gemini 3 Flash (a teacher itself) benefit from cross-task memory?
#
# Runs 3 evals:
#   A. Gemini no-mem on hard_test   (fills teacher baseline hole — we only had train)
#   B. Gemini + SmartMem on easy_test   (vs known 52.1 pct baseline)
#   C. Gemini + SmartMem on hard_test   (vs A)
#
# Config matches original teacher distillation: --max-turns 12, --num-drafts 3,
# --max-tokens 4096, --backend litellm. No --no-think (closed-source model).
#
# SmartRetriever reads enriched memory with --no-cross-memory-write.
# The 'writeeveryturn' prompt rule is now in SYSTEM_PROMPT_DSPREDICT, so applies here too.

set -e
export PYTHONUNBUFFERED=1

export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_enriched.json
MODEL=openai/gemini-3-flash-preview

cd $DSGYM
source .venv/bin/activate

run_eval() {
    local tag="$1"
    local dataset="$2"
    local compose="$3"
    local extra_flags="$4"

    echo "=============================="
    echo "[${tag}] bring up ${compose} at $(date)"
    echo "=============================="
    cd $DSGYM/executors
    sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
    sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
    sudo docker compose -f $compose up -d
    sleep 10
    curl -s http://localhost:5000/status | head -c 500 || true
    echo

    cd $DSGYM
    echo "Starting ${tag} at $(date)"
    # shellcheck disable=SC2086
    dsgym eval \
        --model $MODEL \
        --dataset $dataset \
        --backend litellm \
        --agent aide \
        --memory-version v6 \
        --no-task-memory \
        --api-key $LITELLM_API_KEY \
        --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
        --num-drafts 3 \
        --max-turns 12 \
        --max-tokens 4096 \
        --max-workers 8 \
        --best-node-strategy best \
        $extra_flags \
        --output-dir evaluation_results/${tag} \
        2>&1 | tee logs/${tag}.out
    echo "${tag} finished at $(date)"
}

# A. Gemini no-mem on hard_test
run_eval "gemini_flash_nomem_hard_test" "dspredict-hard-test" "docker-dspredict-hard.yml" \
    "--no-cross-memory"

# B. Gemini + SmartMem on easy_test
run_eval "gemini_flash_smartmem_easy_test" "dspredict-easy-test" "docker-dspredict-easy.yml" \
    "--memory-path $MEMORY --no-cross-memory-write"

# C. Gemini + SmartMem on hard_test
run_eval "gemini_flash_smartmem_hard_test" "dspredict-hard-test" "docker-dspredict-hard.yml" \
    "--memory-path $MEMORY --no-cross-memory-write"

echo "All done at $(date)"
