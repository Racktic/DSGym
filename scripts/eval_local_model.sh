#!/bin/bash
# =============================================================================
# DSGym: 使用本地模型进行多worker评测的脚本
# =============================================================================
#
# 架构说明:
#   vLLM serve (HTTP服务, 多GPU tensor parallel)
#       ↑ ↑ ↑ ↑ (HTTP请求, 并发)
#   dsgym eval --backend litellm --max-workers N
#       ├── Worker 1 → Container 1 (执行代码)
#       ├── Worker 2 → Container 2 (执行代码)
#       └── ...
#
# 前提条件:
#   1. 已安装 DSGym: uv sync --extra vllm
#   2. 已启动 Docker containers: docker compose -f docker-dspredict-hard.yml up -d
#   3. Container Manager 在 localhost:5000 运行中
#
# 代码改动 (已在 litellm_backend.py 中完成):
#   - generate() 方法中显式传递 api_base 和 api_key 给 litellm.completion()
#   - _setup_litellm() 方法中读取 OPENAI_API_BASE 环境变量
#
# =============================================================================

set -e

# ========================== 配置区域 ==========================

# 模型路径
MODEL_PATH="/data/fnie/LLaMA-Factory/saves/qwen3-qr-dab-all-10epochs-16accu-nef10-lr2e-5/full/sft/checkpoint-96"

# vLLM serve 配置
SERVE_GPUS="0,1,2,3"           # 用哪些GPU跑推理
TENSOR_PARALLEL_SIZE=4          # tensor parallel 数量, 需与GPU数量一致
PORT=8001                       # vLLM serve 端口
MODEL_NAME="qwen3-4b-sft"      # serve对外暴露的模型名
GPU_MEMORY_UTILIZATION=0.8      # GPU显存利用率
MAX_MODEL_LEN=32768             # 最大序列长度

# eval 配置
DATASET="dspredict-easy"        # 评测数据集
MAX_WORKERS=6                   # 并发worker数, 不能超过container数量(默认8个)
OUTPUT_DIR="results/qwen3_4b_sft2k"

# ========================== 步骤1: 启动 vLLM serve ==========================

echo ">>> 步骤1: 启动 vLLM serve (GPU: ${SERVE_GPUS}, TP: ${TENSOR_PARALLEL_SIZE})"

CUDA_VISIBLE_DEVICES=${SERVE_GPUS} .venv/bin/vllm serve ${MODEL_PATH} \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served-model-name ${MODEL_NAME} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --disable-log-requests \
    &

VLLM_PID=$!
echo ">>> vLLM serve PID: ${VLLM_PID}"

# 等待 server 就绪
echo ">>> 等待 vLLM serve 启动..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo ">>> vLLM serve 已就绪 (${i}s)"
        break
    fi
    if [ $i -eq 120 ]; then
        echo ">>> 错误: vLLM serve 120秒内未启动"
        kill ${VLLM_PID} 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ========================== 步骤2: 检查 Container 状态 ==========================

echo ">>> 步骤2: 检查 Container Manager 状态"
CONTAINER_STATUS=$(curl -s http://localhost:5000/status)
AVAILABLE=$(echo ${CONTAINER_STATUS} | python3 -c "import sys,json; print(json.load(sys.stdin)['available_containers'])")
ALLOCATED=$(echo ${CONTAINER_STATUS} | python3 -c "import sys,json; print(json.load(sys.stdin)['allocated_containers'])")

echo ">>> 可用容器: ${AVAILABLE}, 已分配: ${ALLOCATED}"

if [ "${AVAILABLE}" -lt "${MAX_WORKERS}" ]; then
    echo ">>> 警告: 可用容器(${AVAILABLE})少于worker数(${MAX_WORKERS}), 调整 MAX_WORKERS=${AVAILABLE}"
    MAX_WORKERS=${AVAILABLE}
fi

# ========================== 步骤3: 运行评测 ==========================

echo ">>> 步骤3: 启动评测 (workers: ${MAX_WORKERS}, dataset: ${DATASET})"

OPENAI_API_BASE=http://localhost:${PORT}/v1 \
OPENAI_API_KEY=dummy \
.venv/bin/dsgym eval \
    --model openai/${MODEL_NAME} \
    --backend litellm \
    --dataset ${DATASET} \
    --max-workers ${MAX_WORKERS} \
    --output-dir ${OUTPUT_DIR} \
    --api-key dummy

echo ">>> 评测完成, 结果保存在: ${OUTPUT_DIR}"

# ========================== 步骤4: 清理 ==========================

echo ">>> 停止 vLLM serve..."
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null
echo ">>> 完成"
