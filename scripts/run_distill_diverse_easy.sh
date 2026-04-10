#!/bin/bash
# Diverse teacher distillation on dspredict-easy: Claude Sonnet, GPT-5.2, Gemini Flash (sequential)
# Usage: cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_distill_diverse_easy.sh

set -e
export PYTHONUNBUFFERED=1

DSGYM=/data/fnie/qixin/DSGym
LITELLM_KEY="$LITELLM_API_KEY"
LITELLM_URL="https://litellm.nbdevenv.xiaoaojianghu.fun"

# --- Claude Sonnet 4.6 ---
echo "=============================="
echo "Starting Claude Sonnet 4.6 at $(date)"
echo "=============================="
dsgym eval \
    --model openai/claude-sonnet-4.6 \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key "$LITELLM_KEY" \
    --base-url "$LITELLM_URL" \
    --num-drafts 3 \
    --max-turns 8 \
    --max-workers 8 \
    --best-node-strategy best \
    --max-tokens 4096 \
    --output-dir "$DSGYM/evaluation_results/distill_claude_sonnet_easy" \
    2>&1 | tee "$DSGYM/logs/distill_claude_sonnet_easy.out"

echo "Claude Sonnet finished at $(date)"

# --- GPT-5.2 ---
echo "=============================="
echo "Starting GPT-5.2 at $(date)"
echo "=============================="
dsgym eval \
    --model openai/gpt-5.2 \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key "$LITELLM_KEY" \
    --base-url "$LITELLM_URL" \
    --num-drafts 3 \
    --max-turns 8 \
    --max-workers 8 \
    --best-node-strategy best \
    --max-tokens 4096 \
    --output-dir "$DSGYM/evaluation_results/distill_gpt5_easy" \
    2>&1 | tee "$DSGYM/logs/distill_gpt5_easy.out"

echo "GPT-5.2 finished at $(date)"

# --- Gemini 3 Flash ---
echo "=============================="
echo "Starting Gemini 3 Flash at $(date)"
echo "=============================="
dsgym eval \
    --model openai/gemini-3-flash-preview \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key "$LITELLM_KEY" \
    --base-url "$LITELLM_URL" \
    --num-drafts 3 \
    --max-turns 8 \
    --max-workers 8 \
    --best-node-strategy best \
    --max-tokens 4096 \
    --output-dir "$DSGYM/evaluation_results/distill_gemini_flash_easy" \
    2>&1 | tee "$DSGYM/logs/distill_gemini_flash_easy.out"

echo "All done at $(date)"
