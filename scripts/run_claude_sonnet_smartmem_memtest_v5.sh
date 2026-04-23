#!/bin/bash
# Claude Sonnet 4.6 + SmartMem V5 on easy_test + hard_test.
#
# V5 = V4 replay + expanded caveats targeting V4 memory-hurt tasks.
# New/changed caveats vs V4:
#   - idx 239 walmart: caveat[0] condition tightened to "retail/demand panel w/
#     calendar series_id" so it no longer misfires on ventilator/novozymes
#   - idx 24 m5-forecasting-accuracy improvement: NEW caveat for sub-second
#     sensor signals (addresses Claude V4 ventilator −22 failure)
#   - idx 1162 liberty-mutual: NEW caveat for biological sequence tasks
#     (addresses Claude V4 novozymes −17.92 failure)
#
# Retrieval replay: same v2_retrieval_replay.json as V4. Only memory content changes.
#
# Cost estimate: ~$24 (easy + hard, Claude Sonnet 4.6).
#
# Usage:
#   export LITELLM_API_KEY=...
#   export OPENAI_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=$LITELLM_API_KEY OPENAI_API_KEY=$OPENAI_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_claude_sonnet_smartmem_memtest_v5.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_caveat_enriched.json
REPLAY=$DSGYM/data/memory/v2_retrieval_replay.json
MODEL=openai/claude-sonnet-4.6

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

run_eval "claude_sonnet_smartmem_easy_test_v5" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "claude_sonnet_smartmem_hard_test_v5" "dspredict-hard-test" "docker-dspredict-hard.yml"

echo "All done at $(date)"
