#!/bin/bash
# Distill with Gemini 3 Pro via LiteLLM proxy on dspredict-easy
# Usage: cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_distill_gemini_pro_easy.sh

set -e
export PYTHONUNBUFFERED=1

DSGYM=/data/fnie/qixin/DSGym

dsgym eval \
    --model openai/gemini-3-pro-preview \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --api-key $LITELLM_API_KEY \
    --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir "$DSGYM/evaluation_results/distill_gemini_pro_easy" \
    2>&1 | tee "$DSGYM/logs/distill_gemini_pro_easy.out"
