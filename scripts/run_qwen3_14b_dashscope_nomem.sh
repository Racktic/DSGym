#!/bin/bash
# Qwen3-14B via Alibaba DashScope (OpenAI-compatible) — no-memory baseline.
# Runs easy_test + hard_test with the AIDE agent, no cross/task memory.
#
# Why DashScope: local GPUs are busy (no multi-vllm), so use the hosted
# Qwen3-14B chat endpoint through litellm's OpenAI-compatible path. The
# existing litellm backend already forwards --no-think -> enable_thinking=False
# via extra_body, which DashScope requires for non-streaming calls.
#
# Usage:
#   export DASHSCOPE_API_KEY=sk-...
#   bash /data/fnie/qixin/DSGym/scripts/run_qwen3_14b_dashscope_nomem.sh

set -e
export PYTHONUNBUFFERED=1

export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY}"

DSGYM=/data/fnie/qixin/DSGym
MODEL=qwen3-14b
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

cd $DSGYM
source .venv/bin/activate

run_eval() {
    local tag="$1"
    local dataset="$2"
    local compose="$3"

    echo "=============================="
    echo "[${tag}] bring up ${compose} at $(date)"
    echo "=============================="
    cd $DSGYM/executors
    sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
    sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
    sudo docker compose -f $compose up -d
    sleep 15
    curl -s http://localhost:5000/status | head -c 500 || true
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
        --api-key $DASHSCOPE_API_KEY \
        --base-url $BASE_URL \
        --num-drafts 3 \
        --max-turns 12 \
        --max-tokens 4096 \
        --max-workers 4 \
        --no-think \
        --best-node-strategy best \
        --output-dir evaluation_results/${tag} \
        2>&1 | tee logs/${tag}.out
    echo "${tag} finished at $(date)"
}

run_eval "qwen3_14b_dashscope_nomem_easy_test" "dspredict-easy-test" "docker-dspredict-easy.yml"
run_eval "qwen3_14b_dashscope_nomem_hard_test" "dspredict-hard-test" "docker-dspredict-hard.yml"

echo "All done at $(date)"
