# Group Evolving through Collective Memory — 方法文档

## 1. 问题设定

我们研究这样一个问题：让多种底层模型（如 Claude、Gemini、GPT、Qwen 等）在同一套 Kaggle 风格机器学习基准上独立完成任务，能否通过一个**共享的、可演化的经验记忆池**，使整个 agent 群体随时间不断变强，而不只是依赖单个模型自身的学习能力。

传统的 in-context RAG 把记忆当作只读知识库 —— 每个任务只"读"不"写"，或只允许同一 agent 单向积累。我们关心的是一个更接近群体学习的设定：

- **共享载体**：所有 agent（不论模型家族）共用同一份记忆。
- **异构参与者**：不同模型的认知偏好不同，它们会在相同记忆上做出不同选择，也会在不同类型的任务上失败。
- **演化驱动**：群体每一轮的失败成为下一轮记忆的修正信号，记忆内容按"使用 → 失败 → 修正"的循环逐渐长出对特定任务的精细指导。

我们将这个框架称为 **Group Evolving through Collective Memory (GECM)**。

---

## 2. 方法总览

GECM 由三个模块组成：

| 模块 | 作用 |
|------|------|
| **Collective Memory Pool**  | 存储所有历史任务的求解片段（draft / improve / debug 级别的 insight） |
| **SmartRetriever** | 面向当前任务动态检索一小组相关条目注入到 agent 的 prompt |
| **Group-CL Caveat Synthesis** | 从群体的失败中抽取"作用域警示"（scope caveats），挂到相应记忆条目上 |

一次完整的群体演化循环（`epoch`）如下：

```
epoch_t: 记忆 M_t  ──▶  [模型 A, B, C ...] 并行求解任务集 T
                              │
                              ▼
                       收集每个模型的失败样本
                              │
                              ▼
                   Group-CL Critic 总结 caveats
                              │
                              ▼
                       M_{t+1} = M_t + caveats
                              │
                              ▼
epoch_{t+1}: 新一轮 agent 使用 M_{t+1} ...
```

---

## 3. Collective Memory Pool

### 3.1 条目结构

记忆池是一个扁平的 entry 列表，每条 entry 描述"某个模型在某个任务的某个阶段做了什么并取得了什么效果"：

```json
{
  "challenge_name": "store-sales-time-series-forecasting",
  "task_name": "...",
  "action": "draft" | "improve" | "debug",
  "entry_type": "draft_success" | "improvement" | "debug_fix",
  "model_type": "LightGBM",
  "score": 0.766,
  "insight": "自然语言 summary — 方法、关键超参、结果 ...",
  "domain": "time_series",
  "task_type": "regression",
  "metric": "rmsle",
  "data_size": "large",
  "scope_caveats": [ ... ]   // 演化层，见 §5
}
```

前两层字段（求解过程 + 数据/任务分类元数据）由**教师模型**（强模型跑 baseline 时的轨迹）产出并统一离线富化。分类元数据的枚举空间是固定的（`domain ∈ {tabular, image, text, time_series, ...}` 等），便于后续检索时做 hard filter。

### 3.2 条目向量化

每条 entry 的 `insight`（或任务描述）用 `text-embedding-3-small` 离线编码为 1536 维向量，与 `.npy` 文件一一对齐持久化；检索时直接加载、无需重新调用 embedding API。

---

## 4. SmartRetriever

SmartRetriever 将"根据当前任务动态挑选几条最相关的历史经验"这件事拆成三步，关键是**结构化过滤 + 语义排序 + 稳定随机**：

### 4.1 任务分类（Hard Filter）

当前任务的 query description 先由 LLM 分类进同一套枚举空间：

```
{domain, task_type, metric, data_size}
```

只有在 `domain` 和 `task_type` 上与 query 兼容（或一方为 `"other"` 通配）的记忆条目才能进入候选池。这一步在 tabular/image/text/time_series 等完全不同的任务家族之间形成硬隔离，避免了"用 NLP 经验指导图像任务"的跨域污染。

### 4.2 相似度排序 + Challenge Name Boost

通过 hard filter 后，候选集与 query embedding 做批量余弦相似度，取 top-`candidate_pool` (默认 60)：

$$s(i) = \cos(\mathbf{v}_\text{query}, \mathbf{v}_i)$$

若候选条目的 `challenge_name` 与当前 challenge 共享长度 ≥ 4 的公共词（如 `titanic`、`forecasting`），该条目的分数被乘 2 作为 soft boost，让**同系列/同数据源**的经验优先出现，但不硬性独占。

### 4.3 Per-Task Cap 与稳定随机采样

排序后的候选池做两件事：

1. **Per-task cap**：同一个 challenge 最多保留 `max_per_task = 2` 条，避免单个任务的多条记录霸屏。
2. **Stable Pool + Random Sample**：对 `(task, action)` 维度缓存一个大小为 `pool_size = 15` 的稳定池；每一次调用都从中均匀随机采样 `top_k = 3` 条注入 prompt。

这样做的目的是：

- **稳定**：同任务同 action 的整体可见经验集合不变 —— 方便复现、实验对齐。
- **多样**：同一任务的 draft / improve / debug 多轮调用看到的具体 3 条略有差异，agent 的探索空间不被锁死在头部。

### 4.4 按阶段分层（Action-Aware）

AIDE-style agent 有三个固定 action：`draft` / `improve` / `debug`。SmartRetriever 按当前 action 只召回同种 `entry_type` 的条目 —— 写新草稿时看过往 draft，修代码时看 debug fix，而不会把 debug 经验用来指导初始建模。

---

## 5. Group-CL Caveat Synthesis — 演化层

### 5.1 为什么需要 Caveat

只有 `insight` 的记忆条目是**脱离上下文的经验陈述**。它在原任务上是对的，但放到语义相近的任务上未必对 —— 比如：

- battlefin 上 `Ridge + 2 特征`在 200 天 × 198 目标的金融序列上成立；到 store-sales 3M 行零售多序列就会塌陷。
- DontGetKicked 上 LogisticRegression 作为 ~70k 行二分类 baseline 合理；到 ieee-fraud 590k × 400 维 3.5% 正样本，直接套用 LR 会 AUC 掉到 0.57。

这类**作用域错配**不是 insight 本身写错了，而是检索时该条目被语义相似度带到了它不该指导的任务上。我们不能删除该条目（在原任务上它仍然是对的），因此需要在条目上**附加前向警示**。

### 5.2 Caveat 结构

Caveat 是条目的嵌套字段，结构化记录"该经验何时不该用"：

```json
"scope_caveats": [
  {
    "condition": "large-scale imbalanced binary classification (rows > 100k AND positive rate < 10%)",
    "caveat": "DontGetKicked LR 是 ~70k 行的 baseline；不要把它推广到 ieee-fraud 上 ...",
    "source": "ieee-fraud-detection GPT-5.2 V2 failure (2026-04-21, AUC 0.57 vs baseline 0.93)",
    "derived_for_tasks": ["ieee-fraud-detection"]
  }
]
```

四个字段分工：

- **`condition`**：自然语言判据 —— agent 读到时可以判断"我这个任务是否落入该条件"。
- **`caveat`**：应做什么 / 不应做什么，以及背后的失败机制。
- **`source`**：证据出处（哪次实验、什么任务、什么指标）。可溯源、便于后续复核是否过时。
- **`derived_for_tasks`**：该 caveat 当初是为哪些任务被写出来的 —— 用于自动跳过（见 §5.4）。

### 5.3 Caveat 的来源：Group-CL Critic

每个 epoch 结束时，对参与的每个模型，抽取其在各任务上的 per-task percentile 并与 baseline 对比，识别大幅回退样本（例如 percentile 降低 > 25）。对每个回退样本：

1. 定位该 agent 当时的 trajectory 中出现过的记忆条目（由 retrieval log 记录）；
2. 判断失败是否由**条目的 insight 被错误推广**引起；
3. 如是，由 critic（强 LLM）在该条目上新增一条 `scope_caveats`，说明"哪类任务/数据规模下不要照搬"。

这个过程是**群体驱动**的：多个模型的失败共同更新同一记忆池。Claude V2 暴露过的 store-sales 漏洞写出的 caveat，Gemini V3 和 GPT V3 都会读到；反之亦然。

### 5.4 Opt-Out 与 Caveat 渲染策略

并非所有任务都适合读取任何条目。我们为记忆接入了 `opt_out_tasks` 集合（已知"读记忆反而变差"的任务）：

- **任务级 opt-out**：`retrieve()` 对这些任务直接返回空，即不注入任何 memory block。
- **Caveat-级自动跳过**：若一条 caveat 的 `derived_for_tasks` 完全落在 opt-out 集合内，则 `format_for_prompt` 不渲染它 —— 因为它服务的目标任务已经不再用 memory，继续渲染只会污染语义上碰巧检索到的其他任务。

这是一个简单但重要的"守门人"：它保证随着 opt-out 集合变化，caveat 的可见性自适应收缩，不会在记忆中堆积死代码。

### 5.5 渲染到 prompt

检索结果按 challenge 分组注入 prompt，每条 entry 的 insight 后紧跟其 caveat：

```
--- Task: DontGetKicked ---
  [DRAFT] Model: LogisticRegression | Score: 0.49 | <insight ...>
    ⚠ Scope caveat — when large-scale imbalanced binary classification ...: <msg>
```

`⚠` + 条件从句的格式刻意使 caveat 与主体 insight 视觉上区分开，让 agent 在读到条件从句时先判断"我是否落入此条件"，再决定是否按主体 insight 执行。

---

## 6. 群体演化的闭环

整套 GECM 形成一个可迭代闭环：

1. **Base**：教师模型产出 `M_0`（只有 insights，无 caveats）。
2. **Epoch 1**：agent 集合 {Claude, Gemini, GPT, ...} 各自在 `M_0` 上跑 easy_test + hard_test。
3. **Critic**：每个模型的大幅回退案例 → caveat，合并到 `M_1`。
4. **Epoch 2**：同 agent 集合（或加入新模型）在 `M_1` 上重跑。部分 caveat 被触发，部分失败类型消失，新的失败暴露出来。
5. **迭代**：`M_{t+1} = M_t ∪ caveats_t`，直到 group-level metric 收敛。

这个设计使得：

- **单个 agent 弱点可被群体看到**：GPT 跌倒的地方 Claude 下次能躲开。
- **经验不被覆盖、只被加约束**：原 insight 永远保留，caveat 只缩小其适用域，知识单调增加。
- **来源可追溯**：每条 caveat 都能回查具体失败记录，未来对抗过时知识时可以精准淘汰。

---

## 7. 设计原则小结

| 设计决策 | 目的 |
|---------|------|
| 结构化元数据硬过滤 | 避免跨域污染，让相似度排序只在同类任务内生效 |
| Cosine ranking + challenge-name boost | 相似度主导，系列任务 soft 优先 |
| Per-task cap + stable pool + random sample | 稳定而不单调，可复现而不僵化 |
| Action-aware 分层（draft / improve / debug） | 让相关经验进入相应阶段 |
| Caveat 附加而非覆盖 | 知识单调可演化，原 insight 不丢 |
| Caveat 自带 `condition` 与 `derived_for_tasks` | 使用者可判断适用性，维护者可回收 |
| Opt-out 机制 | 任务层面与 caveat 层面双重隔离，防止错挂 |

---

## 8. 讨论：方法目前的边界

本方法尚待解决的两个结构性问题（留作后续研究动机，方便与后续实验节合并）：

1. **Caveat 渲染的精准性**：当前 caveat 只要其所挂 entry 被检索到就会渲染，依赖 agent 自行判断 `condition` 是否命中。如果 agent 对 `condition` 的判定偏保守或偏激进，caveat 会成为干扰或被忽略。一个自然的扩展是：在渲染前对 `condition` 与当前任务语义做一次轻量 gate（规则 / 小模型判别）。

2. **Caveat 的汇总与去重**：随着 epoch 增加，同类失败的 caveat 可能累积 —— 需要定期做 caveat merge（同 condition 的多源 caveat 折叠成一条），避免 prompt 膨胀与同义重复。

以上构成 GECM 第二阶段的工作基础。
