#!/bin/bash
cd /data/fnie/qixin/DSGym
export TOGETHER_API_KEY="tgp_v1_CZUEMzg7WnjCfNkZ6ActxCwcGgDWXbbuaQbdqpCzZfk"
export PYTHONUNBUFFERED=1
mkdir -p evaluation_results/aide_current_score_test
.venv/bin/dsgym eval \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 2 \
    --limit 2 \
    --output-dir /data/fnie/qixin/DSGym/evaluation_results/aide_current_score_test \
    > /data/fnie/qixin/DSGym/logs/aide_current_score_test.out 2>&1
echo "Done: $?"
