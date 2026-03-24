#!/bin/bash
# =============================================================================
# DSPredict-Easy 全量评测脚本
#
# 流程:
#   Step 1: 释放GPU
#   Step 2: 并行跑两个 single-worker eval (GPU 0-3: Qwen3-4B, GPU 4-7: SFT)
#   Step 3: 并行跑两个 multi-worker eval (GPU 0-3: Qwen3-4B, GPU 4-7: SFT)
# =============================================================================

# 不使用 set -e, 因为 deallocate/pkill 等清理命令可能返回非零退出码

# ========================== 配置 ==========================
QWEN3_MODEL="/data/fnie/qixin/models/Qwen3-4B-Instruct-2507"
SFT_MODEL="/data/fnie/LLaMA-Factory/saves/qwen3-qr-dab-all-10epochs-16accu-nef10-lr2e-5/full/sft/checkpoint-96"

DATASET="dspredict-easy"
MULTI_WORKERS=4
QWEN3_SERVE_PORT=8001
SFT_SERVE_PORT=8002

# 输出目录
QWEN3_SINGLE_OUT="results/qwen3_4b_instruct_single"
SFT_SINGLE_OUT="results/qwen3_4b_sft_single"
QWEN3_MULTI_OUT="results/qwen3_4b_instruct_multi"
SFT_MULTI_OUT="results/qwen3_4b_sft_multi"

# ========================== Step 1: 释放GPU ==========================
echo "=========================================="
echo "Step 1: 释放GPU"
echo "=========================================="

# 杀掉所有 dsgym eval 和 vllm 进程
pkill -f "dsgym eval" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true
sleep 3
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 3

echo ">>> GPU状态:"
nvidia-smi | grep "MiB.*/" | head -8

# 释放所有stuck containers
for i in 0 1 2 3 4 5 6 7; do
    curl -s --max-time 5 -X POST http://localhost:5000/deallocate/$i 2>/dev/null || true
done
echo ">>> Container状态:"
curl -s http://localhost:5000/status
echo ""

# ========================== Step 2: 并行 single-worker eval ==========================
echo ""
echo "=========================================="
echo "Step 2: 并行 single-worker eval"
echo "  GPU 0-3: Qwen3-4B-Instruct"
echo "  GPU 4-7: SFT checkpoint"
echo "=========================================="

mkdir -p ${QWEN3_SINGLE_OUT} ${SFT_SINGLE_OUT}

# Qwen3-4B single worker (GPU 0-3)
echo ">>> 启动 Qwen3-4B single-worker eval..."
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_TENSOR_PARALLEL_SIZE=4 \
    .venv/bin/dsgym eval \
    --model ${QWEN3_MODEL} \
    --dataset ${DATASET} \
    --backend vllm \
    --max-workers 1 \
    --output-dir ${QWEN3_SINGLE_OUT} \
    > logs/qwen3_single.log 2>&1 &
QWEN3_SINGLE_PID=$!
echo ">>> Qwen3-4B PID: ${QWEN3_SINGLE_PID}"

# SFT single worker (GPU 4-7)
echo ">>> 启动 SFT single-worker eval..."
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_TENSOR_PARALLEL_SIZE=4 \
    .venv/bin/dsgym eval \
    --model ${SFT_MODEL} \
    --dataset ${DATASET} \
    --backend vllm \
    --max-workers 1 \
    --output-dir ${SFT_SINGLE_OUT} \
    > logs/sft_single.log 2>&1 &
SFT_SINGLE_PID=$!
echo ">>> SFT PID: ${SFT_SINGLE_PID}"

echo ">>> 等待两个 single-worker eval 完成..."
echo ">>> 可用 tail -f logs/qwen3_single.log 或 tail -f logs/sft_single.log 查看进度"

wait ${QWEN3_SINGLE_PID}
QWEN3_SINGLE_EXIT=$?
echo ">>> Qwen3-4B single-worker 完成 (exit: ${QWEN3_SINGLE_EXIT})"

wait ${SFT_SINGLE_PID}
SFT_SINGLE_EXIT=$?
echo ">>> SFT single-worker 完成 (exit: ${SFT_SINGLE_EXIT})"

# 确保GPU和container释放
sleep 5
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 3
for i in 0 1 2 3 4 5 6 7; do
    curl -s --max-time 5 -X POST http://localhost:5000/deallocate/$i 2>/dev/null || true
done

# ========================== Step 3: 并行 multi-worker eval ==========================
echo ""
echo "=========================================="
echo "Step 3: 并行 multi-worker eval"
echo "  GPU 0-3: Qwen3-4B (port ${QWEN3_SERVE_PORT}, ${MULTI_WORKERS} workers)"
echo "  GPU 4-7: SFT (port ${SFT_SERVE_PORT}, ${MULTI_WORKERS} workers)"
echo "=========================================="

mkdir -p ${QWEN3_MULTI_OUT} ${SFT_MULTI_OUT}

# 启动两个 vLLM serve
echo ">>> 启动 vLLM serve (Qwen3-4B, GPU 0-3, port ${QWEN3_SERVE_PORT})..."
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/vllm serve ${QWEN3_MODEL} \
    --host 0.0.0.0 --port ${QWEN3_SERVE_PORT} \
    --served-model-name qwen3-4b-instruct \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --disable-log-requests \
    > logs/vllm_serve_qwen3.log 2>&1 &
QWEN3_VLLM_PID=$!

echo ">>> 启动 vLLM serve (SFT, GPU 4-7, port ${SFT_SERVE_PORT})..."
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/vllm serve ${SFT_MODEL} \
    --host 0.0.0.0 --port ${SFT_SERVE_PORT} \
    --served-model-name qwen3-4b-sft \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --disable-log-requests \
    > logs/vllm_serve_sft.log 2>&1 &
SFT_VLLM_PID=$!

# 等待两个 server 就绪
echo ">>> 等待两个 vLLM serve 启动..."
QWEN3_READY=0
SFT_READY=0
for i in $(seq 1 120); do
    if [ ${QWEN3_READY} -eq 0 ] && curl -s http://localhost:${QWEN3_SERVE_PORT}/health > /dev/null 2>&1; then
        echo ">>> Qwen3-4B serve 就绪 (${i}s)"
        QWEN3_READY=1
    fi
    if [ ${SFT_READY} -eq 0 ] && curl -s http://localhost:${SFT_SERVE_PORT}/health > /dev/null 2>&1; then
        echo ">>> SFT serve 就绪 (${i}s)"
        SFT_READY=1
    fi
    if [ ${QWEN3_READY} -eq 1 ] && [ ${SFT_READY} -eq 1 ]; then
        break
    fi
    if [ $i -eq 120 ]; then
        echo ">>> 错误: vLLM serve 120秒内未全部启动"
        kill ${QWEN3_VLLM_PID} ${SFT_VLLM_PID} 2>/dev/null
        exit 1
    fi
    sleep 1
done

# 并行运行两个 eval
echo ">>> 启动 Qwen3-4B multi-worker eval..."
OPENAI_API_BASE=http://localhost:${QWEN3_SERVE_PORT}/v1 OPENAI_API_KEY=dummy \
    .venv/bin/dsgym eval \
    --model openai/qwen3-4b-instruct \
    --backend litellm \
    --dataset ${DATASET} \
    --max-workers ${MULTI_WORKERS} \
    --output-dir ${QWEN3_MULTI_OUT} \
    --api-key dummy \
    > logs/qwen3_multi.log 2>&1 &
QWEN3_EVAL_PID=$!

echo ">>> 启动 SFT multi-worker eval..."
OPENAI_API_BASE=http://localhost:${SFT_SERVE_PORT}/v1 OPENAI_API_KEY=dummy \
    .venv/bin/dsgym eval \
    --model openai/qwen3-4b-sft \
    --backend litellm \
    --dataset ${DATASET} \
    --max-workers ${MULTI_WORKERS} \
    --output-dir ${SFT_MULTI_OUT} \
    --api-key dummy \
    > logs/sft_multi.log 2>&1 &
SFT_EVAL_PID=$!

echo ">>> 等待两个 multi-worker eval 完成..."
echo ">>> 可用 tail -f logs/qwen3_multi.log 或 tail -f logs/sft_multi.log 查看进度"

wait ${QWEN3_EVAL_PID}
echo ">>> Qwen3-4B multi-worker 完成"
wait ${SFT_EVAL_PID}
echo ">>> SFT multi-worker 完成"

# 停止两个 vLLM serve
kill ${QWEN3_VLLM_PID} ${SFT_VLLM_PID} 2>/dev/null
sleep 3
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 3

# ========================== 汇总结果 ==========================
echo ""
echo "=========================================="
echo "全部完成! 结果分析:"
echo "=========================================="

for dir in ${QWEN3_SINGLE_OUT} ${SFT_SINGLE_OUT} ${QWEN3_MULTI_OUT} ${SFT_MULTI_OUT}; do
    echo ""
    echo "--- ${dir} ---"
    if ls ${dir}/*_results.json 1>/dev/null 2>&1; then
        python3 scripts/analyze_dspredict.py ${dir}
    else
        echo "  (无结果文件)"
    fi
done
