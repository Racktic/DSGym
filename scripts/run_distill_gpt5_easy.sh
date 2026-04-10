#!/bin/bash
# Distill with GPT-5.2 via LiteLLM proxy on dspredict-easy
# Usage: cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_distill_gpt5_easy.sh

set -e
export PYTHONUNBUFFERED=1

DSGYM=/data/fnie/qixin/DSGym

# Reset container manager
sudo docker restart executors-manager-1 2>/dev/null || sudo docker restart executors_manager_1 2>/dev/null || true
sleep 5

dsgym eval \
    --model openai/gpt-5.2 \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key $LITELLM_API_KEY \
    --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
    --num-drafts 3 \
    --max-turns 8 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir "$DSGYM/evaluation_results/distill_gpt5_easy" \
    2>&1 | tee "$DSGYM/logs/distill_gpt5_easy.out"
