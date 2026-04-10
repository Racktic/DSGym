# Docker Executor 配置指南

DSGym 使用 Docker 容器执行 agent 生成的代码。每个 task 在一个独立容器里运行 Jupyter kernel，agent 通过 HTTP API 发送代码、接收执行结果。

## 三种配置

### 1. `docker-dspredict-easy.yml` — 轻量级（Easy / Swap）

| 项目 | 值 |
|------|------|
| **镜像** | `executor-kaggle` |
| **内存** | 2G |
| **CPU** | 0.5 核 |
| **GPU** | 无 |
| **超时** | 600 秒（10 分钟） |
| **容器数** | 8 |
| **适用 split** | `dspredict-easy`, `dspredict-swap`, `dspredict-hard-swap` |

**适用场景**：Playground Series 等小型 tabular 竞赛、target-swap 合成任务。数据量小（< 100MB），不需要 GPU，模型训练快。

### 2. `docker-dspredict-hard.yml` — 标准重量级（Hard）

| 项目 | 值 |
|------|------|
| **镜像** | `executor-kaggle` |
| **内存** | 24G |
| **CPU** | 8 核 |
| **GPU** | 有（nvidia） |
| **超时** | 3600 秒（1 小时） |
| **容器数** | 8 |
| **适用 split** | `dspredict-hard`, `dspredict-hard-rejected` |

**适用场景**：真实 Kaggle 竞赛（大数据集、复杂特征工程、需要 GPU 加速训练）。

**注意**：`executor-kaggle` 镜像包含基础 ML 包（sklearn、xgboost、lightgbm、catboost 等，约 29 个包），但**不包含** transformers、tensorflow、torch、cv2 等。如果 task 需要这些，用 mledojo 配置。

### 3. `docker-dspredict-mledojo.yml` — 全栈重量级（MLE Dojo）

| 项目 | 值 |
|------|------|
| **镜像** | `executor-mle` |
| **内存** | 24G |
| **CPU** | 8 核 |
| **GPU** | 有（nvidia） |
| **超时** | 3600 秒（1 小时） |
| **容器数** | 8 |
| **适用 split** | `dspredict-mledojo`, `dspredict-mle-bench` |

**适用场景**：MLE-Dojo benchmark 任务，涉及 NLP、CV、推荐系统等多种领域，需要更全面的包支持。

**`executor-mle` 镜像额外包含**（相比 `executor-kaggle`）：
- `transformers`, `sentence-transformers`, `datasets`
- `torch`, `tensorflow`, `keras`
- `albumentations`, `kornia`, `efficientnet-pytorch`
- `fastai`, `gym`, `implicit`
- `langchain`, `anthropic`
- 共 92 个包（kaggle 只有 29 个）

## 使用方法

### 切换容器配置

在目标节点上操作。**同一时间只能跑一种配置**——切换前必须先 down 当前配置。

```bash
cd /data/fnie/qixin/DSGym/executors

# 停掉当前容器（如果有）
sudo docker compose -f docker-dspredict-easy.yml down 2>/dev/null
sudo docker compose -f docker-dspredict-hard.yml down 2>/dev/null
sudo docker compose -f docker-dspredict-mledojo.yml down 2>/dev/null

# 启动目标配置（三选一）
sudo docker compose -f docker-dspredict-easy.yml up -d       # Easy/Swap
sudo docker compose -f docker-dspredict-hard.yml up -d       # Hard
sudo docker compose -f docker-dspredict-mledojo.yml up -d    # MLE Dojo
```

### 验证容器状态

```bash
# 检查容器是否全部 Up
sudo docker ps --format '{{.Names}} {{.Image}} {{.Status}}' | grep executor

# 检查 manager 状态（应该 8/8 available）
curl -s http://localhost:5000/status | python3 -m json.tool

# 检查具体容器配置
sudo docker inspect executors-executor-000-1 | grep -E '"Memory"|EXECUTION_TIMEOUT'
```

### 重启 manager（释放卡死的容器）

如果 Ctrl+C 中断实验后容器没释放（503 错误），重启 manager：

```bash
sudo docker restart executors-manager-1    # docker compose v2 命名
# 或
sudo docker restart executors_manager_1    # docker compose v1 命名
```

## 容器生命周期

```
allocate_container() → kernel restart → 清理 /submission/* → 执行代码 → deallocate_container()
```

- **allocate**：从 manager 获取空闲容器
- **kernel restart**：清除上一个 task 的内存变量
- **清理 /submission/***：删除残留的 submission.csv（防止容器污染）
- **执行代码**：通过 HTTP POST 发送 Python 代码到容器的 Jupyter kernel
- **deallocate**：归还容器到 pool

### 容器污染问题（已修复）

当 `--max-workers > 1` 时容器被复用。之前会出现：
- 上一个 task 的 kernel 变量残留
- 上一个 task 的 `/submission/submission.csv` 残留

修复：在 `allocate_container()` 中增加了 kernel restart + 删除 `/submission/*` 的逻辑。

## 超时注意事项

| 层面 | 超时 | 说明 |
|------|------|------|
| 容器 EXECUTION_TIMEOUT | 600s / 3600s | 单次代码执行的容器级超时 |
| httpx client timeout | 1800s（30 分钟） | HTTP 客户端超时，**实际生效的上限** |

⚠️ httpx client 的 1800s < 容器的 3600s，所以 hard/mledojo 配置下**实际每轮代码执行最多 30 分钟**（httpx 先超时）。如需更长，修改 `AllocatedCodeToolGroup.__init__` 的 `timeout` 参数。

## 镜像构建

镜像的 Dockerfile 和 requirements 在：
- `executors/container_images/kaggle_image/` — executor-kaggle
- `executors/container_images/mle_image/` — executor-mle
- `executors/container_images/instance/` — executor-prebuilt（基础版，不用于 dspredict）
- `executors/container_images/bio_image/` — executor-bio（生物信息学）

构建命令（在有 Docker 的节点上）：
```bash
cd executors/container_images/kaggle_image
docker build -t executor-kaggle .
```

## 迁移到新集群的检查清单

1. [ ] 在新节点上构建 `executor-kaggle` 和 `executor-mle` 镜像
2. [ ] 构建 `manager-prebuilt` 镜像（`executors/manager/` 目录）
3. [ ] 拷贝 `executors/docker-dspredict-*.yml` 三个 compose 文件到新节点
4. [ ] 拷贝数据到新节点：`data/data/dspredict-easy/`, `dspredict-hard/`, `dspredict-mledojo/`
5. [ ] 确认 compose 文件中的 volume mount 路径匹配新节点的数据目录
6. [ ] `sudo docker compose -f docker-dspredict-easy.yml up -d` 测试启动
7. [ ] `curl http://localhost:5000/status` 验证 manager 正常
8. [ ] 用 `dsgym eval --limit 1` 跑一个 task 验证端到端流程
