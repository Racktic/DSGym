# AIDE Memory Versions: V4 / V5 / V6 对比

所有实验使用 **Qwen3-235B-A22B-Instruct-2507-tput** 在 **dspredict-easy**（38 个 task）上评测，分数为 **public leaderboard percentile**。

| Version | Strategy | Valid | Avg Pct | Above Median | 特点 |
|---------|----------|-------|---------|--------------|------|
| **V4** | latest | 35/38 | 45.2 | 17/38 | LLM summary memory，task_memory 用结构化字段（step/score/best_score/baseline_score/notes）记录每轮，**没有 cross-task memory** |
| **V4** | best | 34/38 | 45.2 | 14/38 | 同上，improve 时参考 best score 节点而非最近节点 |
| **V5** | latest | 34/38 | 48.1 | 16/38 | 在 V4 基础上加入 **cross-task memory**（V2-style 结构化 insight），跨 task 共享经验，模型输出仍含 `<search_state>/<step>/<best_score>/<baseline_score>` |
| **V5** | best | 33/38 | **49.2** | 16/38 | V5 的最优配置，improve 选 best score 节点 + cross-task memory |
| V5 (no task memory) | best | 29/38 | 48.6 | 14/38 | 关闭 in-task memory 注入，valid 数明显下降 |
| V5 (log degradation) | best | 30/38 | 48.7 | 13/38 | 在 cross-task memory 中**额外记录失败的 improve 尝试**（score 下降的步骤），让模型看到"反例" |
| **V6** | best | 27/38 | 48.4 | 16/38 | 简化输出格式：assistant 只输出 `<goal>` + `<python>`，分数追踪改由 summary LLM 在 task_memory 里维护 `is_best` |

## 关键观察

- **V4 → V5**：cross-task memory 把 avg_pct 从 45 提到 48-49（+3-4 个点），说明跨任务经验共享有效
- **V5 best vs latest**：best strategy 略好（49.2 vs 48.1），improve 时参考 best score 节点比参考最近节点更有效
- **V6 valid 数偏低（27）** 是因为 container 污染（同期跑的实验，前 24 个被 Kaggle rejected），不是 V6 本身的问题。按 percentile 看 V6 ≈ V5
- **in-task memory 的重要性**：V5 best (33 valid) vs no_task_memory (29 valid)，关闭 in-task memory 后 valid 数明显下降，说明它对模型稳定输出有帮助

## 版本设计差异

### V4（LLM Summary）
- 每轮调用 summary LLM 生成结构化的 task_memory entry
- 字段：`step`, `model`, `score`, `best_score`, `baseline_score`, `notes`
- **无 cross-task memory**

### V5（V2-style Cross-task）
- 在 V4 基础上加入 cross-task memory
- 跨 task 共享 V2 风格的结构化 insight（improvement / debug_fix / task_summary）
- 模型输出格式仍包含 `<search_state><step>N</step><best_score>X</best_score><baseline_score>Y</baseline_score><goal>...</goal></search_state>` + `<python>`
- 引入 `--best-node-strategy` 选项（latest / best）

### V6（简化输出 + is_best 追踪）
- **模型输出大幅简化**：只有 `<goal>` + `<python>`，去掉了 `<search_state>` 包裹和 step/best_score/baseline_score 标签
- 分数追踪改由 summary LLM 维护：每个 task_memory entry 加 `is_best` 字段
- 减少模型输出的 noise，更适合 SFT 训练数据
- summary prompt 接收 task_description 前 3000 字符 + code，便于判断 metric 方向
