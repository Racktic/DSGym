#!/bin/bash
#SBATCH --job-name=vllm-qwen3-8b
#SBATCH --nodelist=research-common-17
#SBATCH --partition=oneday
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/fnie/qixin/DSGym/logs/vllm_%j.out
#SBATCH --error=/data/fnie/qixin/DSGym/logs/vllm_%j.err

VLLM_VENV=/data/fnie/qixin/vllm-env
MODEL=/data/huggingface/qwen3-8b-yarn/
PORT=8000

# Create venv and install vllm if not already done
if [ ! -f "$VLLM_VENV/bin/vllm" ]; then
    echo "Creating vllm venv..."
    uv venv "$VLLM_VENV" --python 3.12
    uv pip install --python "$VLLM_VENV/bin/python" vllm
fi

echo "Starting vLLM server on $(hostname):${PORT}"
echo "Model: $MODEL"

"$VLLM_VENV/bin/vllm" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name qwen3-8b \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768
