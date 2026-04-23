#!/bin/bash
#SBATCH --job-name=qwen14b_sft_v5ckpt16_nomem_train
#SBATCH --nodelist=research-common-33
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --partition=oneday
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/fnie/qixin/DSGym/logs/qwen3_14b_sft_v5ckpt16_nomem_train_slurm_%j.out
#SBATCH --error=/data/fnie/qixin/DSGym/logs/qwen3_14b_sft_v5ckpt16_nomem_train_slurm_%j.err

# SFT Qwen3-14B (distill_diverse_v5_with_hard_truncAF checkpoint-16) evaluated on
# dspredict-easy-train + dspredict-hard-train with NO cross-task memory.
#
# Model: /data/fnie/LLaMA-Factory/saves/qwen3-14b-distill_diverse_v5_with_hard_truncAF/full/sft/checkpoint-16
# Submit via:   sbatch scripts/run_qwen3_14b_sft_v5ckpt16_nomem_train.sh

set -e
export PYTHONUNBUFFERED=1

DSGYM=/data/fnie/qixin/DSGym
MODEL=/data/fnie/LLaMA-Factory/saves/qwen3-14b-distill_diverse_v5_with_hard_truncAF/full/sft/checkpoint-16
TAG=qwen3_14b_sft_v5ckpt16_nomem

cd $DSGYM
source .venv/bin/activate

# --- easy_train ---
echo "=============================="
echo "[easy_train] switching to docker-dspredict-easy.yml at $(date)"
echo "=============================="
cd $DSGYM/executors
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-easy.yml up -d
sleep 10
curl -s http://localhost:5000/status | head -c 500 || true
echo

cd $DSGYM
echo "Starting ${TAG} easy_train at $(date)"
dsgym eval \
    --model $MODEL \
    --dataset dspredict-easy-train \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --no-task-memory \
    --num-drafts 3 \
    --max-turns 12 \
    --max-workers 8 \
    --temperature 0.7 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/${TAG}_easy_train \
    2>&1 | tee logs/${TAG}_easy_train.out
echo "easy_train finished at $(date)"

# --- hard_train ---
echo "=============================="
echo "[hard_train] switching to docker-dspredict-hard.yml at $(date)"
echo "=============================="
cd $DSGYM/executors
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml up -d
sleep 10
curl -s http://localhost:5000/status | head -c 500 || true
echo

cd $DSGYM
echo "Starting ${TAG} hard_train at $(date)"
dsgym eval \
    --model $MODEL \
    --dataset dspredict-hard-train \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --no-task-memory \
    --num-drafts 3 \
    --max-turns 12 \
    --max-workers 8 \
    --temperature 0.7 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/${TAG}_hard_train \
    2>&1 | tee logs/${TAG}_hard_train.out
echo "hard_train finished at $(date)"

echo "All done at $(date)"
