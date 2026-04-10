# Iterative Self-Improvement Framework for Data Science Agents

## Core Idea

通过 **iterative trajectory distillation with selective self-improvement** 训练 data science agent。小模型通过迭代地在真实执行环境中生成 trajectory、保留高质量结果、用 teacher 补弱项、用合成 task 扩充训练集，逐步提升自己的 data science 能力，最终在部分任务上超越 teacher。

## 三个支柱

1. **Iterative Self-Improvement**：student 自我进化，每轮用自己生成的高质量 trajectory 替代 teacher 数据
2. **Selective Teacher Fallback**：保底不退化——低于阈值的 task 回退到 teacher trajectory
3. **Targeted Task Synthesis**：哪里弱补哪里——分析失败 pattern，合成同类型新 task

## 详细流程

### Round 0: Seed（初始化）

**目标**：用多个 teacher 模型的 diverse trajectories 训练初始 student。

```
Multiple Teachers (Claude, GPT, Gemini, Qwen3-235B, Coder-480B)
    │
    ├─ 每个 teacher 在 easy/swap/mledojo 上蒸馏（无 cross-task memory）
    │   → 产出 diverse trajectory pool
    │
    ├─ 转换为 truncAF SFT 数据（截断到 best score + final submission）
    │
    ├─ 从 teacher trajectories 初始化 cross-task memory
    │   → 提取每个 task 的最佳策略、关键经验、常见陷阱
    │
    └─ SFT 训练 → Student_0
```

**关键设计**：
- 多 teacher diverse 蒸馏：每个 teacher 的解题风格不同，student 能学到更多策略
- 不启用 cross-task memory 蒸馏：避免 teacher 间经验富集导致 diversity 下降
- truncAF 截断：只保留到 best score 的路径 + final submission，但包含中间失败轮次（约 75%）

### Round N: Self-Improvement（迭代优化）

**目标**：用 student 自己生成的好 trajectory 替代 teacher 数据，用 teacher 补弱项。

```
Student_N + cross-task memory → 跑所有 tasks
    │
    ├─ task score > threshold → 保留 student trajectory（知识富集）
    │   • student 在这些 task 上已经"学会了"
    │   • 自己生成的 trajectory 更匹配自己的 distribution（无 distribution shift）
    │
    ├─ task score < threshold → 用 teacher trajectory 替代（补短板）
    │   • student 在这些 task 上还没学好
    │   • 保底不会越学越差
    │
    ├─ 分析失败的 task pattern
    │   • 哪类 task 系统性失败？（时序？NLP？大数据集？）
    │   • 失败原因是什么？（代码 bug？策略选择？超时？）
    │
    └─ Targeted Task Synthesis（针对性合成）
        • 从失败类型的数据集合成新 task（target-swap、feature subset 等）
        • teacher 和 student 都在新 task 上跑
        • 好的 trajectory 加入训练池
    
    → 混合数据集（student 好的 + teacher 补的 + 新 task 的）
    → SFT 训练 → Student_{N+1}
    → 更新 cross-task memory
```

### 阈值设计

**方案 A：相对于 teacher**
- `student_score >= teacher_score` → 保留 student trajectory
- 优点：直接衡量 student 是否学会了
- 缺点：需要 teacher 也跑同一个 task

**方案 B：Percentile 阈值**
- `kaggle_percentile >= 50%` → 保留 student trajectory
- 优点：不需要 teacher 对照
- 缺点：不同 task 难度不同

**推荐**：方案 A 用于有 teacher baseline 的 task，方案 B 用于新 task。

### Cross-Task Memory 在迭代中的角色

```
Round 0: 从 teacher trajectories 提取初始 memory
    → 每个 task 的最佳方法、关键参数、常见错误

Round N: Student 跑 task 时读 memory → 利用之前积累的经验
    → 好的 task 的新经验写入 memory（更新）
    → 差的 task 的经验保留（不覆盖）

Round N+1: 用更新后的 memory 指导下一轮
```

**每轮是否重新积累 memory？**
- 建议：不完全重置。保留 teacher 的初始经验作为基础，student 的新经验叠加。
- 原因：student 自己发现的 pattern 比继承的更匹配，但 teacher 的经验有保底作用。

## Task Synthesis（任务合成）

### 合成方式

1. **Target-Swap**：复用已有数据集，换预测目标列
   - 已验证可行（`scripts/generate_hard_swap_tasks.py`）
   - 从 hard split 的 tabular 数据集生成了 10 个 swap task

2. **Feature Subset**：从原始数据集中选不同 feature 子集
   - 同一个数据，不同的特征组合 → 不同的建模策略

3. **数据增强**：对训练集做采样/噪声注入
   - 同一个 task 但数据分布略有不同

4. **Cross-domain Transfer**：从外部数据源引入新 task
   - MLE-Bench（47 个独有 task 待集成）
   - OpenML、UCI 等公开数据集

### Targeted Synthesis 策略

```
分析 Round N 失败的 task:
    │
    ├─ 时序预测类 → 合成更多时序 target-swap task
    ├─ NLP/文本类 → 引入 MLE-Bench 的 NLP task
    ├─ 大数据集超时 → 合成小样本版本，练采样策略
    └─ 特征工程类 → 合成不同 feature subset task
```

## 与现有工作的区别

| 方法 | 区别 |
|------|------|
| **Dynamic Cheatsheet** | DC 是 test-time learning（不改参数），我们是 train-time + self-improvement |
| **STaR / ReST** | 它们在静态 QA 上做 self-improvement，我们在 agentic code execution + 真实 Kaggle 评估上做 |
| **AIDE** | AIDE 是固定算法（不进化），我们的 agent 通过迭代训练进化 |
| **MLE-Bench** | MLE-Bench 只是 benchmark，我们是方法（benchmark + agent + memory + distillation + self-improvement） |

## 核心假设与验证

| 假设 | 验证方式 |
|------|---------|
| Student 自己生成的好 trajectory 比 teacher 的更适合训练 | 对比：round 1 student-only data vs teacher-only data 训练效果 |
| Cross-task memory 能帮助 student 在新 task 上更快收敛 | 对比：有 memory vs 无 memory 的 student 表现 |
| Targeted task synthesis 能补齐 student 的弱点 | 对比：有 synthesis vs 无 synthesis 的 round N→N+1 提升 |
| 迭代几轮后 student 能超越 teacher | 跟踪每轮的 avg percentile，看是否超过 teacher baseline |

## 实验计划

### Phase 1: Seed Data（已完成）
- [x] Claude / GPT / Gemini / Qwen3-235B 多 teacher 蒸馏 easy
- [x] Qwen3-235B 蒸馏 swap (4 rounds)、mledojo
- [x] Gemini / Coder-480B 蒸馏 mledojo
- [x] 转换为 truncAF SFT 数据（312 条 diverse + 442 条含 swap）

### Phase 2: Student_0 训练与评估（进行中）
- [x] Qwen3-8B SFT 训练（多个 checkpoint）
- [x] 在 easy/hard 上评估
- [ ] 分析 student_0 的失败 pattern

### Phase 3: Iterative Self-Improvement
- [ ] Student_0 + memory 跑 easy/swap/mledojo
- [ ] 按阈值筛选好/差 trajectory
- [ ] Targeted task synthesis
- [ ] Round 1 训练 → Student_1
- [ ] 对比 Student_1 vs Student_0 vs Teacher

### Phase 4: Scaling
- [ ] 多轮迭代（Round 2, 3, ...）
- [ ] 扩展到 hard split 和 MLE-Bench
- [ ] 验证收敛性和 teacher-surpassing
