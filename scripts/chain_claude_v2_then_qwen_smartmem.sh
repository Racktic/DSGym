#!/bin/bash
# One-shot chain: wait for the running claude_sonnet_smartmem_memtest_v2 on this
# node to finish, then immediately launch run_qwen3_14b_dashscope_smartmem.sh.
#
# Env vars required (inherited from the outer launcher):
#   DASHSCOPE_API_KEY  — DashScope inference for qwen3-14b
#   OPENAI_API_KEY     — SmartRetriever embeddings
#   LITELLM_API_KEY    — SmartRetriever task classify via LiteLLM proxy
#
# Usage (from a workstation):
#   ssh research-common-22 \
#     "DASHSCOPE_API_KEY=... OPENAI_API_KEY=... LITELLM_API_KEY=... \
#      nohup bash /data/fnie/qixin/DSGym/scripts/chain_claude_v2_then_qwen_smartmem.sh \
#      > /tmp/chain_claude_v2_then_qwen.log 2>&1 &"

set +e
export PYTHONUNBUFFERED=1

: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${LITELLM_API_KEY:?Set LITELLM_API_KEY}"

SENTINEL="run_claude_sonnet_smartmem_memtest_v2"

echo "[$(date '+%F %T')] chain started on $(hostname); polling for '$SENTINEL' to finish"

# Poll every 60s until no matching process remains.
while pgrep -f "$SENTINEL" > /dev/null; do
    sleep 60
done

echo "[$(date '+%F %T')] '$SENTINEL' finished; launching qwen3-14b dashscope smartmem"
bash /data/fnie/qixin/DSGym/scripts/run_qwen3_14b_dashscope_smartmem.sh
echo "[$(date '+%F %T')] qwen3-14b dashscope smartmem finished with exit=$?"
