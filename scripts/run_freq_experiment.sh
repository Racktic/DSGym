#!/bin/bash
cd /data/fnie/qixin/DSGym
export TOGETHER_API_KEY="tgp_v1_CZUEMzg7WnjCfNkZ6ActxCwcGgDWXbbuaQbdqpCzZfk"
export PYTHONUNBUFFERED=1
mkdir -p evaluation_results/aide_memory_v2_bestnode_freq evaluation_results/aide_memory_v2_bestnode_freq_easy
nohup .venv/bin/dsgym eval \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --best-node-strategy best \
    --memory-path /data/fnie/qixin/DSGym/evaluation_results/aide_memory_v2_bestnode_freq/cross_task_memory.json \
    --output-dir /data/fnie/qixin/DSGym/evaluation_results/aide_memory_v2_bestnode_freq_easy \
    > /data/fnie/qixin/DSGym/logs/aide_memory_v2_bestnode_freq.out 2>&1 &
echo "PID: $!"
