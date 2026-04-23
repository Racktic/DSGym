#!/bin/bash
# Gemini 3 Flash + SmartMem V4 on easy_test + hard_test.
#
# V4 = caveat-enriched memory + Gemini V1 retrieval replay.
# Goal: eliminate retrieval-sampling variance so the delta vs Gemini V1 is
# attributable to scope_caveat content on entries V1 actually retrieved.
#
# Memory: data/memory/cross_task_memory_teacher_v5_caveat_enriched.json
#   - Caveats on idx 23 (m5-forecasting-accuracy draft_success)
#   - Caveats on idx 175 (bnp-paribas draft_success) ← added for Gemini ieee-fraud
#   - Caveats on idx 239 (walmart draft_success)
# Replay: data/memory/gemini_v1_retrieval_replay.json
#
# A-group tasks (caveat actually visible in V4 prompt):
#   - ieee-fraud-detection        (idx 175 new caveat, draft calls 0, 6)
#   - recruit-restaurant          (idx 239 + 23, draft calls 0–3, 6–7)
#   - ventilator-pressure         (idx 239 + 23, draft calls 0–6)
#   - store-sales                 (idx 239 + 23, draft calls 0–3)
#   - playground-series-s3e19     (idx 239 + 23, draft calls 0–2)   [easy]
#   - titanic                     (idx 175, final_submission call 0) [easy]
#
# B-group tasks (V4 = V1 memory identically → LLM-decode null):
#   digit, home-data, house-prices, mens-march, nlp, novozymes,
#   playground-{s3e13, s3e25, s4e1, s4e3, s5e3}, spaceship-titanic  (12 tasks)
#
# Cost estimate: Gemini 3 Flash is cheaper than Claude Sonnet; expect <$10 total.
#
# Usage:
#   export LITELLM_API_KEY=...
#   export OPENAI_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=$LITELLM_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_gemini_flash_smartmem_memtest_v4.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_caveat_enriched.json
REPLAY=$DSGYM/data/memory/gemini_v1_retrieval_replay.json
MODEL=openai/gemini-3-flash-preview

export DSGYM_REPLAY_SAMPLES="$REPLAY"

cd $DSGYM
source .venv/bin/activate

run_eval() {
    local tag="$1"; local dataset="$2"; local compose="$3"

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

    export DSGYM_RETRIEVAL_LOG=$DSGYM/logs/retrieval_debug_${tag}.jsonl
    : > "$DSGYM_RETRIEVAL_LOG"

    cd $DSGYM
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
        --api-key $LITELLM_API_KEY \
        --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
        --num-drafts 3 \
        --max-turns 12 \
        --max-tokens 4096 \
        --max-workers 8 \
        --best-node-strategy best \
        --output-dir evaluation_results/${tag} \
        2>&1 | tee logs/${tag}.out
    echo "${tag} finished at $(date)"
}

run_eval "gemini_flash_smartmem_easy_test_v4" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "gemini_flash_smartmem_hard_test_v4" "dspredict-hard-test" "docker-dspredict-hard.yml"

echo "All done at $(date)"
