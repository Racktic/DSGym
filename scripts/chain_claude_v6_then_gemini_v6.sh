#!/bin/bash
# One-shot chain on a single node: Claude V6 → Gemini V6.
# Both use the same memory JSON (which now includes Claude+Gemini V4-derived caveats).
set +e
export PYTHONUNBUFFERED=1
: "${LITELLM_API_KEY:?Set LITELLM_API_KEY}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
echo "[$(date)] Chain start: Claude V6 then Gemini V6"
bash /data/fnie/qixin/DSGym/scripts/run_claude_sonnet_smartmem_memtest_v6.sh
echo "[$(date)] Claude V6 finished, starting Gemini V6"
bash /data/fnie/qixin/DSGym/scripts/run_gemini_flash_smartmem_memtest_v6.sh
echo "[$(date)] Gemini V6 finished — chain complete"
