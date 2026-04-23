#!/bin/bash
# Sequential chain on a single node:
#   1. Claude V7 rerun (easy+hard)
#   2. Gemini V7 (easy+hard)
#   3. Qwen3-Coder-480B baseline (easy+hard, no memory)
#   4. GPT-5.2 baseline (hard only, no memory)
# Each script switches docker compose between easy/hard, so they MUST run sequentially on one node.
#
# Env vars required:
#   LITELLM_API_KEY    (Claude, Gemini, GPT-5.2 via LiteLLM proxy)
#   OPENAI_API_KEY     (SmartRetriever embeddings for Claude + Gemini)
#   TOGETHER_API_KEY   (Qwen3-Coder-480B via Together AI)
#
# Usage:
#   ssh research-common-22 \
#     "LITELLM_API_KEY=... OPENAI_API_KEY=... TOGETHER_API_KEY=... \
#      nohup bash /data/fnie/qixin/DSGym/scripts/chain_claude_v7rerun_gemini_v7_coder480.sh \
#      > /tmp/chain_v7rerun_gemv7_coder480.log 2>&1 &"

set +e  # keep chain going even if one step fails
export PYTHONUNBUFFERED=1

: "${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

echo "[$(date)] === Chain start: claude_v7rerun -> gemini_v7 -> coder480_baseline -> gpt52_baseline_hard ==="

echo "[$(date)] STEP 1/4: Claude V7 rerun"
bash /data/fnie/qixin/DSGym/scripts/run_claude_sonnet_smartmem_memtest_v7rerun.sh
echo "[$(date)] STEP 1 finished (exit=$?)"

echo "[$(date)] STEP 2/4: Gemini V7"
bash /data/fnie/qixin/DSGym/scripts/run_gemini_flash_smartmem_memtest_v7.sh
echo "[$(date)] STEP 2 finished (exit=$?)"

echo "[$(date)] STEP 3/4: Qwen3-Coder-480B baseline"
bash /data/fnie/qixin/DSGym/scripts/run_coder480_baseline_memtest.sh
echo "[$(date)] STEP 3 finished (exit=$?)"

echo "[$(date)] STEP 4/4: GPT-5.2 baseline hard_test"
bash /data/fnie/qixin/DSGym/scripts/run_gpt52_baseline_hard_test.sh
echo "[$(date)] STEP 4 finished (exit=$?)"

echo "[$(date)] === Chain complete ==="
