# Qwen3-8B SFT Training Results

所有结果基于 **Qwen3-8B** 在 DSPredict benchmark 上评测，分数为 **public leaderboard percentile**。

## Easy Split（38 个 task）

| 模型/Recipe | Memory | Valid | Avg Pct | Above Median | 备注 |
|-------------|--------|-------|---------|--------------|------|
| **Teacher 对照（Qwen3-235B V5 best）** | V5+best | 33/38 | 49.2 | 16/38 | 上限参考 |
| Base Qwen3-8B (think) | - | 18/38 | 25.1 | 2/38 | 无 SFT，48% 任务 IndentationError |
| Base Qwen3-8B (no think) | - | 15/38 | 23.1 | 2/38 | 无 SFT，48% 任务 runtime error |
| **SFT V5 (improved truncA)** | V5+best | 16/38 | 40.0 | 4/38 | 第一版 SFT，V5 数据训练 |
| SFT V5 (improved truncA) | nomem | 14/38 | 41.1 | 6/38 | 同模型，推理时关闭 cross-task memory |
| **SFT V6 (truncAF) epoch 3** | V6+best | 22/38 | 37.4 | 6/38 | V6 格式数据，3 epoch |
| **SFT V6 (truncAF) epoch 5** | V6+best | 13/38 | 43.5 | 4/38 | V6 格式数据，5 epoch（avg pct 提升但 valid 数下降） |
| SFT V6 w_submission epoch 3 | V6+best | 0/38 | - | 0 | **完全失败**：模型陷入死循环重复输出 |
| SFT distill_all_truncAF (hard config) | nomem | - | - | - | 在 hard 容器上跑 easy，验证容器配置兼容性 |

## Hard Split（54 个 task）

| 模型 | Memory | Valid | Avg Pct | Above Median |
|------|--------|-------|---------|--------------|
| Teacher (Qwen3-235B V5) | V5+best | 18/54 | ~22.9 | - |
| Teacher (Coder-480B V6) | V6+best | TBD (49 完成) | TBD | TBD |
| **SFT V6 (truncAF)** | V6+best | 4/54 | 28.8 | 0/54 |

## MLE Dojo Split（60 个 task）

| 模型 | Memory | Valid | Avg Pct |
|------|--------|-------|---------|
| Teacher (Qwen3-235B V5) | V5+best | 33/60 | ~30 |
| **SFT V6 (truncAF)** | V6+best | 1/60 | 31.2 |

## Swap Split（67 个 task）

| 模型 | Memory | Valid | Avg Pct |
|------|--------|-------|---------|
| Teacher (Qwen3-235B V6 4 rounds) | V6+best | ~65/67 each round | high |
| **SFT V6 (truncAF)** | V6+best | 0/67 | - |

## 关键发现

1. **SFT 对 easy 有效但远不及 teacher**：最佳 SFT 模型 avg_pct 43.5，仍低于 teacher 49.2
2. **Hard / MLE Dojo / Swap 上 SFT 几乎完全失败**：valid submission 极少（0-4 个），说明模型泛化能力差
3. **SFT 模型陷入死循环**：遇到 error 后重复输出相同代码，不会 debug。w_submission_epoch3 完全失败就是这个问题
4. **训练数据问题**：
   - 169/473 (36%) 的样本中存在连续相同 assistant（教模型重复）
   - 72% 的第一轮 goal 包含 "baseline"/"simple yet effective" 模板
   - swap 数据占比过高（55%），且模板化严重
5. **Epoch 越多反而越差**：V6 epoch 3 (valid=22, pct=37.4) → epoch 5 (valid=13, pct=43.5)，说明过拟合到训练数据 pattern

## 下一步

- 清理训练数据：去掉连续重复的 assistant（169 对）
- 降低 swap 数据比例：用 `distill_diverse_truncAF.json`（279 条，swap 仅占 23%）
- 验证 diverse 数据训练效果
