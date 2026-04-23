#!/bin/bash
# GPT-5.2 + SmartMem V4-style on easy_test + hard_test.
# V4 config: caveat-enriched memory + V2 retrieval replay, NO opt-out.
#
# Memory file: cross_task_memory_teacher_v5_caveat_enriched.json (current caveat state)
# Replay:      data/memory/v2_retrieval_replay.json (Claude V2's retrieval sequence)
# NOTE: novozymes opt-out is intentionally NOT set here (matches Claude V4 exactly).
#       Caveats whose derived_for_tasks is fully in opt_out_tasks won't be skipped
#       because opt_out_tasks is empty.
#
# Hyperparams match Claude/Gemini V4: num-drafts=3, max-turns=12, max-tokens=4096.
#
# Usage:
#   export LITELLM_API_KEY=...
#   export OPENAI_API_KEY=...
#   ssh research-common-22 "LITELLM_API_KEY=... OPENAI_API_KEY=... bash /data/fnie/qixin/DSGym/scripts/run_gpt52_smartmem_v4_memtest.sh"

set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"

# V4 == caveat memory + V2 replay, NO opt-out
export DSGYM_REPLAY_SAMPLES=/data/fnie/qixin/DSGym/data/memory/v2_retrieval_replay.json
unset DSGYM_OPT_OUT_TASKS

DSGYM=/data/fnie/qixin/DSGym
# V4 SNAPSHOT: only the 4 caveats that existed at V4 time (idx 23, 175, 239×2).
# Current in-place caveat file has V7 state (12 caveats) which would NOT be a V4 reproduction.
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_v4snap_enriched.json
MODEL=openai/gpt-5.2

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
    curl -s http://localhost:5000/status | head -c 300 || true
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

run_eval "gpt52_smartmem_easy_test_v4" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "gpt52_smartmem_hard_test_v4" "dspredict-hard-test" "docker-dspredict-hard.yml"
echo "GPT-5.2 V4 all done at $(date)"
