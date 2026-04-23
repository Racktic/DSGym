#!/bin/bash
# Qwen3-Coder-480B baseline (no memory) on dspredict-easy-test + dspredict-hard-test.
# Inference via Together AI — no local GPU needed, any node with docker works.
# Hyperparams match Claude/Gemini memtest v6/v7: num-drafts=3, max-turns=12, max-tokens=4096.
#
# Usage:
#   export TOGETHER_API_KEY=...
#   ssh research-common-22 "TOGETHER_API_KEY=$TOGETHER_API_KEY bash /data/fnie/qixin/DSGym/scripts/run_coder480_baseline_memtest.sh"

set -e
export PYTHONUNBUFFERED=1
export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MODEL=together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8

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
    sudo docker restart executors-manager-1 2>/dev/null || sudo docker restart executors_manager_1 2>/dev/null || true
    sleep 15
    curl -s http://localhost:5000/status | head -c 300 || true
    echo

    cd $DSGYM
    echo "Starting ${tag} at $(date)"
    dsgym eval \
        --model $MODEL \
        --dataset $dataset \
        --backend litellm \
        --agent aide \
        --memory-version v6 \
        --no-cross-memory \
        --no-task-memory \
        --num-drafts 3 \
        --max-turns 12 \
        --max-tokens 4096 \
        --max-workers 4 \
        --best-node-strategy best \
        --output-dir evaluation_results/${tag} \
        2>&1 | tee logs/${tag}.out
    echo "${tag} finished at $(date)"
}

run_eval "coder480_baseline_easy_test" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "coder480_baseline_hard_test" "dspredict-hard-test" "docker-dspredict-hard.yml"
echo "Coder480 baseline all done at $(date)"
