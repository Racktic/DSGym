# DSGym Eval 参数说明

## 用法

```bash
cd /data/fnie/qixin/DSGym && source .venv/bin/activate
dsgym eval [参数]
```

---

## 模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必填) | 模型名称或路径。API: `together_ai/Qwen/...`, `openai/claude-sonnet-4.6`。本地: `/path/to/checkpoint` |
| `--backend` | `litellm` | 推理后端。`litellm`=API 调用, `vllm`=本地单卡, `multi-vllm`=本地多卡(每 GPU 一个 TP=1 实例), `sglang`=SGLang |
| `--temperature` | 0.0 | 采样温度 |
| `--max-tokens` | 1524 (litellm) | 每次生成的最大 token 数。Claude/GPT 建议 4096 |
| `--no-think` | False | 关闭 Qwen3 的 thinking mode（`enable_thinking=False`） |

## API 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api-key` | None | API key，不传则从环境变量读取（`TOGETHER_API_KEY`, `OPENAI_API_KEY` 等） |
| `--base-url` | None | 自定义 API 端点（如 LiteLLM proxy `https://litellm.nbdevenv.xiaoaojianghu.fun`）。设置后强制走 OpenAI 兼容协议 |

## 数据集配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | (必填) | 数据集。常用: `dspredict-easy`, `dspredict-hard`, `dspredict-swap`, `dspredict-mledojo`, `dspredict-hard-swap`, `dspredict-hard-rejected`, `dspredict-easy-claude-retry` |
| `--limit` | None (全部) | 限制评测的样本数 |

## 评测配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `./evaluation_results` | 结果输出目录 |
| `--max-turns` | 15 | 每个 task 的最大轮次 |
| `--max-workers` | 自动 | 并行 worker 数。litellm 默认 24，multi-vllm 默认 8，vllm/sglang 默认 1 |
| `--manager-url` | `http://localhost:5000` | Docker 容器 manager 地址 |

## Agent 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent` | `react` | Agent 类型: `react` 或 `aide`（draft-improve-debug） |

### AIDE 专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-drafts` | 5 | 初始 draft 轮数（前 N 轮全部 draft，之后 draft/improve/debug 自动选择） |
| `--best-node-strategy` | `latest` | Improve 时参考哪个节点。`latest`=最近一个有 score 的, `best`=score 最高的 |

### Memory 配置（AIDE）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--memory-version` | `v4` | Memory 版本。`v4`=LLM summary, `v5`=V2 style cross-task, `v6`=简化输出（只有 `<goal>` + `<python>`） |
| `--memory-path` | None | Cross-task memory 文件路径。**启用 cross-task memory 必须同时传 `--memory-version` 和 `--memory-path`** |
| `--no-cross-memory` | False | 完全禁用 cross-task memory 的读写（即使设了 memory-version） |
| `--no-task-memory` | False | 不将 in-task memory 注入 prompt（summary LLM 仍会运行） |
| `--no-draft-memory` | False | Draft 阶段不注入 cross-task memory |
| `--log-degradation` | False | V5/V6: 将 improve 失败（score 下降）也记录到 cross-task memory |

## Memory 参数组合速查

| 场景 | 参数 |
|------|------|
| 裸 AIDE（无任何 memory） | 不传 `--memory-version`, 不传 `--memory-path` |
| V6 格式 + 无 cross-task | `--memory-version v6 --no-cross-memory` |
| V6 格式 + 无 cross-task + 无 in-task | `--memory-version v6 --no-cross-memory --no-task-memory` |
| V6 + cross-task memory | `--memory-version v6 --memory-path /path/to/cross_task_memory.json` |
| V6 + cross-task + best strategy | `--memory-version v6 --memory-path /path/to/memory.json --best-node-strategy best` |

## 常见用法示例

### API 蒸馏（无 memory）
```bash
dsgym eval \
    --model openai/claude-sonnet-4.6 \
    --dataset dspredict-easy \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key YOUR_KEY \
    --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
    --num-drafts 3 \
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir evaluation_results/distill_claude_easy
```

### 本地 SFT 模型评测（与训练数据一致）
```bash
dsgym eval \
    --model /path/to/checkpoint \
    --dataset dspredict-easy \
    --backend multi-vllm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --no-task-memory \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --no-think \
    --best-node-strategy best \
    --output-dir evaluation_results/sft_eval
```

### LiteLLM Proxy 蒸馏（多模型 diverse，无 memory）
```bash
export LITELLM_API_KEY=your_key
dsgym eval \
    --model openai/gemini-3-flash-preview \
    --dataset dspredict-mledojo \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --no-cross-memory \
    --api-key $LITELLM_API_KEY \
    --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
    --num-drafts 3 \
    --max-turns 12 \
    --max-tokens 4096 \
    --max-workers 8 \
    --best-node-strategy best \
    --output-dir evaluation_results/distill_gemini_flash_mledojo
```
支持的 LiteLLM proxy 模型：`openai/claude-sonnet-4.6`, `openai/gpt-5.2`, `openai/gemini-3-flash-preview`。
模型名必须加 `openai/` 前缀（让 litellm 走 OpenAI 兼容协议），proxy 收到后会去掉前缀路由到实际模型。

### Together AI 蒸馏（带 cross-task memory）
```bash
export TOGETHER_API_KEY=xxx
dsgym eval \
    --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
    --dataset dspredict-hard \
    --backend litellm \
    --agent aide \
    --memory-version v6 \
    --best-node-strategy best \
    --memory-path evaluation_results/xxx/cross_task_memory.json \
    --num-drafts 5 \
    --max-turns 20 \
    --max-workers 8 \
    --output-dir evaluation_results/xxx
```

## Docker 容器配置

| Split | Compose 文件 | 镜像 | 内存 | CPU | GPU | Timeout |
|-------|-------------|------|------|-----|-----|---------|
| easy / swap / hard-swap | `docker-dspredict-easy.yml` | executor-kaggle | 2G | 0.5 | 无 | 600s |
| hard / hard-rejected | `docker-dspredict-hard.yml` | executor-kaggle | 24G | 8 | 有 | 3600s |
| mledojo / mle-bench | `docker-dspredict-mledojo.yml` | executor-mle | 24G | 8 | 有 | 3600s |

**executor-kaggle**: 基础 ML 包（sklearn, xgboost, lightgbm, catboost 等 29 个包）
**executor-mle**: 全栈 ML 包（额外含 transformers, torch, tensorflow 等 92 个包）

⚠️ httpx client timeout = 1800s（30 分钟），比容器 timeout 先生效。

详细文档见 [docker_executor_guide.md](docker_executor_guide.md)。

切换容器：
```bash
cd /data/fnie/qixin/DSGym/executors
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null
sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null
sudo docker compose -f docker-dspredict-mledojo.yml down 2>/dev/null
# 启动目标配置（三选一）
sudo docker compose -f docker-dspredict-easy.yml up -d
```
