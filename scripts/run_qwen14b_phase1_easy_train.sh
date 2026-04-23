#!/bin/bash
#SBATCH --job-name=qwen14b_p1_easy
#SBATCH --nodelist=research-common-33
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=/data/fnie/qixin/DSGym/logs/qwen14b_phase1_easy_slurm_%j.out
#SBATCH --error=/data/fnie/qixin/DSGym/logs/qwen14b_phase1_easy_slurm_%j.err

# Phase 1 rollouts on easy_train (30 tasks) with base Qwen3-14B:
#   1. NO memory
#   2. WITH teacher memory (v5_enriched, no caveats — RAG baseline)
# SFT data selection rule: per-task compare Kaggle pct between 2 conditions;
#   memory-helps tasks → use +MEM trajectory as SFT data directly;
#   memory-doesn't-help tasks → DAgger path (teacher correction, later stage).
#
# Submit: sbatch scripts/run_qwen14b_phase1_easy_train.sh

set -e
export PYTHONUNBUFFERED=1

DSGYM=/data/fnie/qixin/DSGym
MODEL=/data/fnie/qixin/models/Qwen3-14B
MEMORY=$DSGYM/data/memory/cross_task_memory_teacher_v5_enriched.json   # no caveats, matches prior Qwen14B SmartMem config
DATASET=dspredict-easy-train
COMPOSE=docker-dspredict-easy.yml

cd $DSGYM
source .venv/bin/activate

cd $DSGYM/executors
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null || true
sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null || true
sudo docker compose -f $COMPOSE up -d
sleep 15
curl -s http://localhost:5000/status | head -c 300 || true
echo

# ---------- 1/2: NO memory ----------
TAG=qwen14b_phase1_easy_train_nomem
cd $DSGYM
echo "Starting ${TAG} at $(date)"
dsgym eval \
    --model $MODEL \
    --dataset $DATASET \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --no-task-memory \
    --num-drafts 3 \
    --max-turns 20 \
    --max-workers 8 \
    --temperature 0.7 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/${TAG} \
    2>&1 | tee logs/${TAG}.out
echo "${TAG} finished at $(date)"

# ---------- 2/2: WITH memory ----------
TAG=qwen14b_phase1_easy_train_mem
echo "Starting ${TAG} at $(date)"
dsgym eval \
    --model $MODEL \
    --dataset $DATASET \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --no-task-memory \
    --memory-path $MEMORY \
    --no-cross-memory-write \
    --num-drafts 3 \
    --max-turns 20 \
    --max-workers 8 \
    --temperature 0.7 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/${TAG} \
    2>&1 | tee logs/${TAG}.out
echo "${TAG} finished at $(date)"

echo "Phase 1 easy_train all done at $(date)"
