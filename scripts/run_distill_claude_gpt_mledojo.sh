#!/bin/bash
# Distill mledojo with Claude Sonnet 4.6 + GPT-5.2 (sequential)
# Requires: mle containers (docker-dspredict-mledojo.yml)
# Usage: cd /data/fnie/qixin/DSGym && source .venv/bin/activate && export LITELLM_API_KEY=xxx && bash scripts/run_distill_claude_gpt_mledojo.sh

set -e
export PYTHONUNBUFFERED=1

DSGYM=/data/fnie/qixin/DSGym
LITELLM_URL="https://litellm.nbdevenv.xiaoaojianghu.fun"

# --- Claude Sonnet 4.6 ---
echo "=============================="
echo "Starting Claude Sonnet 4.6 mledojo at $(date)"
echo "=============================="
sudo docker restart executors-manager-1 2>/dev/null || sudo docker-compose -f $DSGYM/executors/docker-dspredict-mledojo.yml restart executors-manager-1 2>/dev/null || true
sleep 5

dsgym eval \
    --model openai/claude-sonnet-4.6 \
    --dataset dspredict-mledojo \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key $LITELLM_API_KEY \
    --base-url "$LITELLM_URL" \
    --num-drafts 3 \
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir "$DSGYM/evaluation_results/distill_claude_sonnet_mledojo" \
    2>&1 | tee "$DSGYM/logs/distill_claude_sonnet_mledojo.out"

echo "Claude Sonnet mledojo finished at $(date)"

# --- GPT-5.2 ---
echo "=============================="
echo "Starting GPT-5.2 mledojo at $(date)"
echo "=============================="
sudo docker restart executors-manager-1 2>/dev/null || sudo docker-compose -f $DSGYM/executors/docker-dspredict-mledojo.yml restart executors-manager-1 2>/dev/null || true
sleep 5

dsgym eval \
    --model openai/gpt-5.2 \
    --dataset dspredict-mledojo \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key $LITELLM_API_KEY \
    --base-url "$LITELLM_URL" \
    --num-drafts 3 \
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir "$DSGYM/evaluation_results/distill_gpt5_mledojo" \
    2>&1 | tee "$DSGYM/logs/distill_gpt5_mledojo.out"

echo "All done at $(date)"
