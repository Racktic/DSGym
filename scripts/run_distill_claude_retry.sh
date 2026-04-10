#!/bin/bash
# Diverse teacher distillation: Claude retry (15 failed) + GPT-5.2 full + Gemini Flash full
# All on dspredict-easy with hard container config (3600s timeout, 12 turns)
#
# Before running:
#   cd /data/fnie/qixin/DSGym/executors
#   sudo docker compose -f docker-dspredict-easy.yml down
#   sudo docker compose -f docker-dspredict-hard.yml up -d
# Then:
#   cd /data/fnie/qixin/DSGym && source .venv/bin/activate && bash scripts/run_distill_claude_retry.sh

set -e
export PYTHONUNBUFFERED=1

DSGYM=/data/fnie/qixin/DSGym
LITELLM_KEY="$LITELLM_API_KEY"
LITELLM_URL="https://litellm.nbdevenv.xiaoaojianghu.fun"

# Reset container manager
sudo docker restart executors-manager-1 2>/dev/null || sudo docker restart executors_manager_1 2>/dev/null || true
sleep 5

# --- Claude Sonnet 4.6 (15 failed tasks retry) ---
echo "=============================="
echo "Starting Claude Sonnet retry at $(date)"
echo "=============================="
dsgym eval \
    --model openai/claude-sonnet-4.6 \
    --dataset dspredict-easy-claude-retry \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key "$LITELLM_KEY" \
    --base-url "$LITELLM_URL" \
    --num-drafts 3 \
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir "$DSGYM/evaluation_results/distill_claude_sonnet_retry" \
    2>&1 | tee "$DSGYM/logs/distill_claude_sonnet_retry.out"

echo "Claude Sonnet retry finished at $(date)"

# --- GPT-5.2 (full easy) ---
echo "=============================="
echo "Starting GPT-5.2 at $(date)"
echo "=============================="
sudo docker restart executors-manager-1 2>/dev/null || sudo docker restart executors_manager_1 2>/dev/null || true
sleep 5

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
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir "$DSGYM/evaluation_results/distill_gpt5_easy" \
    2>&1 | tee "$DSGYM/logs/distill_gpt5_easy.out"

echo "GPT-5.2 finished at $(date)"

# --- Gemini 3 Flash (full easy) ---
echo "=============================="
echo "Starting Gemini 3 Flash at $(date)"
echo "=============================="
sudo docker restart executors-manager-1 2>/dev/null || sudo docker restart executors_manager_1 2>/dev/null || true
sleep 5

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
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir "$DSGYM/evaluation_results/distill_gemini_flash_easy" \
    2>&1 | tee "$DSGYM/logs/distill_gemini_flash_easy.out"

echo "All done at $(date)"
