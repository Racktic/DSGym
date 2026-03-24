#!/bin/bash
#SBATCH --job-name=eet-memory
#SBATCH --partition=oneday
#SBATCH --nodelist=research-common-03
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/data/fnie/qixin/DSGym/logs/eet_memory_%j.out
#SBATCH --error=/data/fnie/qixin/DSGym/logs/eet_memory_%j.err

set -e

cd /data/fnie/qixin/DSGym

# Together AI API key — set before submitting or export in your shell
if [ -z "$TOGETHER_API_KEY" ]; then
    echo "ERROR: TOGETHER_API_KEY not set"
    exit 1
fi

MEMORY_PATH="/data/fnie/qixin/DSGym/evaluation_results/eet_memory_v1/cross_task_memory.json"
OUTPUT_DIR="/data/fnie/qixin/DSGym/evaluation_results/eet_memory_qwen3_235b_easy_v1"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$MEMORY_PATH")"

echo ">>> Starting EET + Memory evaluation"
echo ">>> Memory path: $MEMORY_PATH"
echo ">>> Output dir: $OUTPUT_DIR"
echo ">>> Time: $(date)"

.venv/bin/dsgym eval \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-easy \
    --backend litellm \
    --agent eet \
    --max-turns 20 \
    --max-workers 8 \
    --memory-path "$MEMORY_PATH" \
    --output-dir "$OUTPUT_DIR"

echo ">>> EET + Memory evaluation completed at $(date)"
