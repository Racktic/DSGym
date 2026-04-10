#!/bin/bash
#SBATCH --job-name=aide-v4-test
#SBATCH --partition=oneday
#SBATCH --nodelist=research-common-03
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/data/fnie/qixin/DSGym/logs/aide_v4_test_%j.out
#SBATCH --error=/data/fnie/qixin/DSGym/logs/aide_v4_test_%j.err

set -e

cd /data/fnie/qixin/DSGym

export TOGETHER_API_KEY="${TOGETHER_API_KEY:?Set TOGETHER_API_KEY}"

OUTPUT_DIR="/data/fnie/qixin/DSGym/evaluation_results/aide_v4_best_test"
MEMORY_PATH="${OUTPUT_DIR}/cross_task_memory.json"
TRAJ_DIR="${OUTPUT_DIR}/trajectories"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TRAJ_DIR"

echo ">>> Starting docker containers..."
cd /data/fnie/qixin/DSGym/executors
docker compose -f docker-dspredict-easy.yml up -d
cd /data/fnie/qixin/DSGym

# Wait for containers to be ready
echo ">>> Waiting for containers to start..."
sleep 15

# Check manager is reachable
echo ">>> Checking container health..."
for port in 60000 60001; do
    for i in $(seq 1 10); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "  Container on port ${port} is healthy"
            break
        fi
        echo "  Waiting for port ${port}... (attempt ${i})"
        sleep 3
    done
done

echo ">>> Starting AIDE V4 (best-node strategy) test — limit 2"
echo ">>> Memory path: $MEMORY_PATH"
echo ">>> Output dir: $OUTPUT_DIR"
echo ">>> Time: $(date)"

.venv/bin/dsgym eval \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 2 \
    --limit 2 \
    --best-node-strategy best \
    --memory-path "$MEMORY_PATH" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo ">>> Test completed at $(date)"
echo ">>> Checking results..."

# Print task memory from trajectory files
echo ""
echo "=== Trajectory files ==="
ls -la "$TRAJ_DIR"/ 2>/dev/null || echo "No trajectory files found in $TRAJ_DIR"

# Check if any trajectory has default dir
ls -la "$OUTPUT_DIR"/aide_trajectories/ 2>/dev/null || true

echo ""
echo "=== Cross-task memory ==="
cat "$MEMORY_PATH" 2>/dev/null || echo "No cross-task memory file"

echo ""
echo "=== Summary JSON ==="
cat "$OUTPUT_DIR"/summary.json 2>/dev/null || echo "No summary.json"

echo ""
echo ">>> Stopping docker containers..."
cd /data/fnie/qixin/DSGym/executors
docker compose -f docker-dspredict-easy.yml down
echo ">>> Done."
