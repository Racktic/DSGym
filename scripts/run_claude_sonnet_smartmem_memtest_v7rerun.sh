#!/bin/bash
# Claude Sonnet 4.6 + SmartMem V6.
# V6 = V5 + Gemini-V4-derived caveats (idx 1094 GiveMeSomeCredit + idx 1083 DontGetKicked
# scale_pos_weight) + idx 88 stanford-covid-vaccine bio-domain caveat +
# NOVOZYMES OPT-OUT (coverage gap — no protein entries in memory).
# Retrieval replay: same V2 replay as V4/V5.
set -e
export PYTHONUNBUFFERED=1
export LITELLM_API_KEY="${LITELLM_API_KEY:?}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?}"
DSGYM=/data/fnie/qixin/DSGym
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_caveat_enriched.json
REPLAY=$DSGYM/data/memory/v2_retrieval_replay.json
MODEL=openai/claude-sonnet-4.6
export DSGYM_REPLAY_SAMPLES="$REPLAY"
export DSGYM_OPT_OUT_TASKS="novozymes-enzyme-stability-prediction"
cd $DSGYM && source .venv/bin/activate

run_eval() {
    local tag="$1"; local dataset="$2"; local compose="$3"
    echo "=== [${tag}] ${compose} at $(date) ==="
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
    dsgym eval --model $MODEL --dataset $dataset --backend litellm --agent aide \
        --memory-version v6 --no-task-memory --memory-path $MEMORY --no-cross-memory-write \
        --api-key $LITELLM_API_KEY --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
        --num-drafts 3 --max-turns 12 --max-tokens 4096 --max-workers 8 \
        --best-node-strategy best --output-dir evaluation_results/${tag} \
        2>&1 | tee logs/${tag}.out
    echo "${tag} done at $(date)"
}

run_eval "claude_sonnet_smartmem_easy_test_v7rerun" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "claude_sonnet_smartmem_hard_test_v7rerun" "dspredict-hard-test" "docker-dspredict-hard.yml"
echo "Claude V6 all done at $(date)"
