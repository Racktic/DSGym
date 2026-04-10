# DSPredict-Easy 评测完整指南

本文档描述如何在一台新机器上从零配置并运行 DSPredict-Easy 评测。读完此文档你应该能独立完成整个流程。

---

## 1. 项目结构概览

```
DSGym/
├── dsgym/
│   ├── agents/           # Agent 实现
│   │   ├── vgs/          # EET, AIDE, VGS agent
│   │   │   ├── eet_agent.py
│   │   │   ├── eet_prompts.py
│   │   │   ├── aide_agent.py
│   │   │   └── ...
│   │   ├── react_ds_agent.py
│   │   └── dspredict_react_agent.py
│   ├── cli/
│   │   └── eval.py       # CLI 入口
│   ├── datasets/         # 数据集加载
│   └── eval/             # 评测框架
├── executors/            # Docker 容器配置
│   ├── docker-dspredict-easy.yml          # 8 容器 compose 文件
│   ├── container_config_dspredict_easy.json  # 容器 URL 配置（Docker 内部网络）
│   ├── container_config.json              # 当前活跃配置（也指向 Docker 内部网络）
│   └── generate_compose.py               # compose 文件生成器
├── data/
│   ├── data/dspredict-easy/   # 38 个 Kaggle 竞赛数据集
│   └── task/dspredict/        # 任务定义 JSON
├── submissions/               # 每个容器的 submission 输出目录
├── evaluation_results/        # 评测结果输出
├── logs/                      # 实验日志
└── .venv/                     # Python 虚拟环境
```

## 2. 环境准备

### 2.1 Python 环境

```bash
cd /data/fnie/qixin/DSGym
# 虚拟环境已存在，激活即可
source .venv/bin/activate
# 验证 CLI 可用
dsgym --help
```

**注意**: 运行评测命令必须用 `.venv/bin/dsgym eval`，不能用 `python -m dsgym`。

### 2.2 API Key 配置

**Together AI**（用于调用 Qwen3-235B 等模型）:
```bash
export TOGETHER_API_KEY="你的key"
```

**Kaggle**（用于自动提交并获取 leaderboard 分数）:
- 配置文件位于 `~/.kaggle/kaggle.json`，格式: `{"username":"xxx","key":"xxx"}`
- 权限必须为 600: `chmod 600 ~/.kaggle/kaggle.json`

### 2.3 Docker 容器

DSPredict-Easy 使用 **8 个** executor 容器 + 1 个 manager 容器。

**必须使用的镜像**:
- Executor: `executor-kaggle`（87GB，预装所有 ML/Kaggle 库）
- Manager: `manager-prebuilt`

**绝对不要用 `executor-prebuilt` 跑 dspredict 任务**，它缺少 ML 库，agent 会浪费 turn 自己装包且大概率失败。

#### 启动容器

```bash
cd /data/fnie/qixin/DSGym/executors
sudo docker compose -f docker-dspredict-easy.yml up -d
```

#### 验证容器健康

```bash
# 检查所有容器运行中
sudo docker compose -f docker-dspredict-easy.yml ps

# 检查 manager 可达（从宿主机访问）
curl http://localhost:5000/status
# 应返回: {"available_containers":8, "allocated_containers":0, ...}

# 检查容器内数据挂载正常（重要！容器运行久了数据挂载可能变空）
sudo docker exec executors-executor-000-1 ls /data/dspredict-easy/ | head -3
# 应该能看到数据集目录名，如 house-prices-advanced-regression-techniques

# 如果数据挂载为空，重启容器
sudo docker compose -f docker-dspredict-easy.yml restart
```

#### 容器网络架构

```
宿主机 (localhost:5000) --> Manager 容器 (port 5000)
                               |
                               v
                    Docker 内部网络 (bridge)
                               |
         ┌─────────┬─────────┬─────────┐
         v         v         v         v
    executor-000  executor-001  ...  executor-007
    (内部 8432)   (内部 8432)       (内部 8432)
```

- Manager 通过 Docker 内部 URL 访问 executor: `http://executor-000:8432`
- 宿主机通过 `localhost:5000` 访问 manager
- `container_config_dspredict_easy.json` 中的 URL 是 Docker 内部地址，不要改成 localhost

## 3. 运行评测

### 3.1 基本评测命令

```bash
export TOGETHER_API_KEY="你的key"

.venv/bin/dsgym eval \
  --dataset dspredict-easy \
  --agent eet \
  --backend litellm \
  --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
  --max-turns 20 \
  --max-workers 8 \
  --output-dir evaluation_results/你的实验名 \
  2>&1 | tee logs/你的实验名_$(date +%Y%m%d_%H%M%S).log
```

### 3.2 参数说明

| 参数 | 说明 | DSPredict-Easy 推荐值 |
|------|------|----------------------|
| `--dataset` | 数据集名称 | `dspredict-easy` |
| `--agent` | Agent 类型 | `react`, `eet`, `aide` |
| `--backend` | 推理后端 | `litellm`（API 调用）, `vllm`（本地模型） |
| `--model` | 模型名 | `together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput` |
| `--max-turns` | 每个任务最大轮数 | `20` |
| `--max-workers` | 并行 worker 数 | `8`（必须 <= 容器数） |
| `--limit` | 只跑前 N 个任务 | 不设=全部 38 个 |
| `--output-dir` | 结果输出目录 | 每次跑必须用新目录，避免结果污染 |
| `--no-terminate` | EET 专用: 禁用 terminate action | 仅 explore/exploit |
| `--num-drafts` | AIDE 专用: draft 轮数 | 默认 5 |

### 3.3 Agent 类型

- **react**: 基础 ReAct agent，无结构化决策
- **eet**: Explore-Exploit-Terminate，结构化搜索决策
  - 加 `--no-terminate` 可禁用 terminate，强制模型用完所有 turn
- **aide**: Draft-Improve-Debug 三阶段 agent

### 3.4 先测试 2 个任务

**每次跑全量之前，务必先用 `--limit 2` 测试**:

```bash
.venv/bin/dsgym eval \
  --dataset dspredict-easy \
  --agent eet \
  --backend litellm \
  --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
  --max-turns 20 \
  --max-workers 2 \
  --limit 2 \
  --output-dir evaluation_results/test_run \
  2>&1 | tee logs/test_run.log
```

检查日志确认无报错后再跑全量。

## 4. 关键注意事项

### 4.1 绝对不能犯的错误

1. **Container 数量**: DSPredict-Easy 固定用 **8 个** container，不要自作主张改成 24 个
2. **Docker 镜像**: 必须用 `executor-kaggle`，不能用 `executor-prebuilt`
3. **Output-dir**: 每次实验用新目录，旧结果会污染新结果
4. **实验被 kill 后**: 必须检查容器状态并手动释放

### 4.2 容器管理

```bash
# 查看容器分配状态
curl http://localhost:5000/status

# 手动释放所有容器（实验中断后必做）
for i in $(seq 0 7); do curl -s -X POST "http://localhost:5000/deallocate/$i"; done

# 重启容器（数据挂载出问题时）
cd executors && sudo docker compose -f docker-dspredict-easy.yml restart
```

### 4.3 监控实验

实验可能跑 1-2 小时，必须监控日志:

```bash
# 实时查看日志
tail -f logs/实验日志.log

# 查看进度（看百分比）
grep "Evaluating" logs/实验日志.log | tail -1

# 查看已完成的 trajectory
ls evaluation_results/实验目录/*_trajectory.json | wc -l
```

### 4.4 日志必须写进文件

所有实验命令必须用 `2>&1 | tee logs/xxx.log` 重定向日志，否则无法排查问题。日志写到 `/data/fnie/qixin/DSGym/logs/` 目录。

## 5. 结果分析

### 5.1 结果文件

评测完成后，output-dir 下会生成:
- `*_results.json` — 每个任务的完整结果（含 Kaggle score、percentile、medal）
- `*_metrics.json` — 汇总指标（success_rate、avg_turns 等）
- `*_trajectory.json` — 每个任务的逐 turn 轨迹（action、score、goal、代码）

### 5.2 关键指标

结果中 Kaggle 指标的路径:
```python
result["metrics"]["kaggle_submission"]["details"]["public_percentile"]  # 公榜百分位
result["metrics"]["kaggle_submission"]["details"]["public_score"]       # 公榜分数
result["metrics"]["kaggle_submission"]["details"]["public_medal"]       # 奖牌
result["metrics"]["kaggle_submission"]["score"]                         # 原始 score
```

### 5.3 分析脚本

```bash
# 对比两个实验的结果
.venv/bin/python scripts/compare_eet_v3_v4.py

# 分析 agent 特性（需要修改脚本中的目录路径）
.venv/bin/python scripts/analyze_agent_characteristics.py
```

## 6. 已完成的实验

| 实验 | 目录 | Agent | 说明 |
|------|------|-------|------|
| React v2 | `react_qwen3_235b_easy_v2/` | react | 基线 |
| EET v3 | `eet_qwen3_235b_easy_v3/` | eet | 原版 EET |
| EET v4 | `eet_qwen3_235b_easy_v4/` | eet | 加了 system best_score tracking（效果中性） |
| AIDE v1 | `aide_qwen3_235b_easy_v1/` | aide | t20 配置 |
| AIDE t10d4 | `aide_qwen3_235b_easy_t10d4/` | aide | t10 + 4 drafts |

### EET 已知问题

- EET 的 score improvement curve 在 turn 9 后变平（不是 bug，是模型搜索能力天花板）
- 约 30% 的 turn 是"浪费的"（在 best score 之后无改善）
- 根因: explore/exploit 在后期无法产生增益，不是 score tracking 的问题
- `--no-terminate` 模式可禁用 terminate 研究这个问题
