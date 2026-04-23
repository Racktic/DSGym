# Caveat 迭代日志（V3 → V5）

从 2026-04-16 开始做 `scope_caveats` 的 Group Continual Learning 初版。这份文档只记录 caveat-相关的进展，不重复 V1/V2 retriever 改动（见 dev_logs/0415.md §13）。

---

## 0. 背景

V1/V2 都是"一次性注入 retrieval memory"的设计：`cross_task_memory_teacher_v5_enriched.json` 里每条 entry 有固定 `insight` 文本，retriever 按 cosine 召回后直接喂给 agent，没有任何"适用条件"的标注。

dev_logs/0415.md §2、§7、§14 诊断出一个共同 failure 模式：**memory 的 insight 在源任务上是对的，但在当前 task 上适用条件不满足 → agent 照搬 → 伤害当前 task**。

典型例子：
- walmart `draft_success` 的 lag_52/lag_104 insight 对 45 stores × 99 depts 的 Walmart 是对的，但套给 3M 行的 store-sales → groupby 超时 + test horizon 不够 → Claude V1 store-sales −26 pct vs NOMEM
- m5-forecasting-accuracy `draft_success` 的 recursive 28-day forecasting 需要 recursive validation，但 agent 做了 direct validation → val/test 机制失配 → store-sales V1 val 0.378 / Kaggle 0.534

**单纯扔 insight 不够，得告诉 agent "这条经验在什么条件下不适用"**。

## 1. 方法：`scope_caveats` 字段 + prompt 渲染 + replay

### 1.1 数据格式

在每条 memory entry 上加一个可选字段 `scope_caveats: List[{condition, caveat, source}]`，不动 `insight` 文本，不重算 embedding：

```json
{
  "challenge_name": "walmart-recruiting-store-sales-forecasting",
  "entry_type": "draft_success",
  "insight": "...same-week-last-year lag features (lag_52 and lag_104)...",
  "scope_caveats": [
    {
      "condition": "retail/demand/sales panel data ... AND training rows > 1M AND unique series count > 500",
      "caveat": "Computing lag_52/lag_104 via groupby('series_id').shift() timed out on store-sales (3M rows, 1782 series). For large retail panels, ...",
      "source": "store-sales-time-series-forecasting v2 failure (2026-04-17)"
    },
    ...
  ]
}
```

### 1.2 代码改动（commit 层面）

- `dsgym/agents/vgs/memory.py`：`MemoryEntry` 类加 optional `scope_caveats` 字段，`from_dict`/`to_dict` 支持；向后兼容（没有字段时默认空 list）
- `dsgym/agents/vgs/smart_retriever.py`：`format_for_prompt` 渲染 `⚠ Scope caveat — when {condition}: {caveat}` 行，只在 entry 有 caveat 时渲染
- Memory 另存为 `cross_task_memory_teacher_v5_caveat_enriched.json`（保留原 `_enriched.json` 不动）；`.npy` embedding 复制一份 `_caveat_embeddings.npy`，aide_agent.py 自动根据 `_enriched.json → _embeddings.npy` 命名规则加载

### 1.3 Replay 机制（V4 引入）

SmartRetriever 默认行为：`retrieve()` 从 pool-15 里 `random.sample(3)`。**两次 run 的 RNG state 不同** → 同一 task agent 看到的 3 条 insight 会飘 → caveat A/B 实验被 retrieval 方差污染。

Fix（commit 层面）：
- `SmartRetriever.__init__` 加 `replay_samples_path` 参数；若 set 则跳过 `_build_pool` + `random.sample`，按 `(challenge, action, call_count)` 从预先提取的 lookup 返回固定 indices
- `aide_agent.py` 读 env var `DSGYM_REPLAY_SAMPLES`
- `scripts/extract_v2_retrieval_samples.py` 从已有 trajectory 反向提取每个 `(challenge, action, call_idx) → [entry_indices]`，100% match（625/625 for Claude V2，627/627 for Gemini V1）

**V4 起**固定用 V2 / V1 的 replay，让后续所有 caveat 实验的 memory input 严格对齐，delta 纯归因到 caveat 内容。

## 2. 各版本的 caveat 状态（精确定义）

| version | ranking | threshold | caveats（cum） | retrieval |
|---------|---------|-----------|---------------|-----------|
| V1 | `0.8 cos + 0.2 score_norm` | 无 | — | 随机 3-from-15 |
| V2 | `1.0 cos` | 无 | — | 随机 3-from-15 |
| V3 | `1.0 cos` | 无 | idx 23 + idx 239 | 随机 3-from-15 |
| V4 | `1.0 cos` | 无 | V3 + idx 175 | **replay from V2** |
| V5 | `1.0 cos` | 无 | V4 + idx 24 + idx 1162 + **idx 239[0] 条件收紧** | replay from V2 |

## 3. V3 — 第一次加 caveat（Claude-derived）

### 3.1 诊断基础

V1 vs NOMEM paired delta 找到 3 个 memory-hurt case（dev_logs/0415.md §13）：

| task | NOMEM | V1 | Δ |
|------|-------|-----|----|
| recruit-restaurant | 66.37 | 33.81 | −32.56 |
| store-sales | 68.53 | 42.41 | −26.12 |
| novozymes | 24.53 | 4.53 | −20.00 |

读 trajectory，store-sales 和 recruit-restaurant 都能定位到具体 culprit entry（walmart idx 239 + m5-accuracy idx 23）。novozymes 是 coverage gap（memory 里没 protein entry），这轮先不管。

### 3.2 Patch 内容

- **idx 239** walmart-recruiting draft_success 加 2 条 caveat：
  - large-panel compute cost（针对 store-sales V1 groupby 超时）
  - horizon-history 覆盖（针对 recruit-restaurant V1 lag NaN）
- **idx 23** m5-forecasting-accuracy draft_success 加 1 条 caveat：
  - recursive forecasting 必须用 recursive rollout 验证（针对 store-sales V1 val/test 机制失配）

### 3.3 V3 结果

V3 是随机抽样、没开 replay，跨 run 的 retrieval 漂移让 single-run 信号很难读：

| task | NOMEM | V2 | V3 | Δv3-v2 |
|------|-------|-----|-----|--------|
| recruit-restaurant | 66.37 | 53.91 | 55.28 | +1.37 |
| store-sales | 68.53 | 10.32 | 24.63 | +14.30 |
| novozymes | 24.53 | 2.90 | 3.87 | +0.97 |

Store-sales 涨 +14.30 算有信号。但 V3 hard mean 44.01 比 V2 的 52.25 低 8 pct（被 spaceship-titanic −81 和 mens-march −35 两个方差 outlier 拖的）。

单 run A/B 太嘈杂 → 引入 replay 做 V4。

## 4. V4 — 引入 replay + Gemini caveat（后来证明无效）

### 4.1 Replay 把 retrieval 固定到 V2

`extract_v2_retrieval_samples.py` 从 V2 trajectory 反向提取 625 条 entry indices，存成 `v2_retrieval_replay.json`。V4 run 时设 `DSGYM_REPLAY_SAMPLES=...` → 每个 `(challenge, action, call_idx)` 精确复现 V2 当时 agent 看到的 3 条。

A/B 分组：
- **A group**（V2 retrieval 里有 idx 23 / 239）：ventilator, recruit-restaurant, store-sales, playground-s3e19 —— V4 会在 prompt 里渲染 caveat
- **B group**（V2 retrieval 里没命中 patched entries）：其余 14 个 task —— V4 memory 和 V2 完全一致，只有 LLM decode 不同

B group 的 V4-vs-V2 delta 就是**纯 LLM decode 方差 null baseline**。

### 4.2 Gemini-driven speculation：idx 175 caveat

之前分析 Gemini V1 ieee-fraud（NOMEM 30.48 → V1 17.32，−13.15）时，发现 Gemini V1 prompt 里出现了 bnp-paribas draft_success（idx 175）+ homesite-quote，两条都 normalize "wide + LightGBM = fine"。Claude V1 ieee-fraud 实际上是 +4.64 涨的，所以是"Gemini-specific 失败"。

我基于 hypothesis"bnp-paribas 的 wide-feature framing 把 Gemini 从 'feature minimalism' 推开了" 给 idx 175 加了 1 条 caveat。结果（见 §5 Gemini V4）证明无效。

### 4.3 V4 结果

Claude V4 vs V2（hard，10 tasks）：

| task | V2 | V4 | Δ | caveat 命中？|
|------|-----|-----|----|-------------|
| store-sales | 10.32 | **77.11** | **+66.79** | ✅ |
| recruit-restaurant | 53.91 | N/A（pipeline 崩）| — | ✅ |
| ventilator-pressure | 34.74 | 15.39 | **−19.35** ⚠️ | ✅ 但反作用 |
| （B-group 12 个 task）| — | — | mean +4.81, stdev **27.08** | ❌ |

**Headline**: 
- **store-sales 是第一个明确的 caveat 大信号**，+66.79 pct 远超 B-group null stdev 27
- **ventilator 被误伤**：V2 只 −2.76，V4 变 −19.35。caveat 出问题，见 §5
- **B-group stdev 27** 量化了 LLM decode 方差量级，之后所有 delta < 27 都要对照这个 null band 解读

Gemini V4 vs V1 类似做法（replay from Gemini V1）：

| task | V1 | V4 | Δ |
|------|-----|-----|----|
| recruit-restaurant | 15.30 | 25.86 | **+10.56** ✅ |
| ieee-fraud（idx 175 caveat target）| 17.32 | 15.09 | −2.23 ❌ |
| B-group | — | — | stdev **7.34** |

Gemini null stdev 远小于 Claude（7.34 vs 27.08）——**Gemini 的 decode 更稳定**，对 caveat A/B 来说更敏感。

**idx 175 caveat 对 Gemini ieee-fraud 无信号**（Δ=−2.23 完全在 null 内）→ 假设证伪，这条 caveat 之后可以 rollback，暂时先保留。

## 5. V5 — Claude V4 failure 驱动的新 caveat

### 5.1 V4 新浮现的失败点

V4 下仍然 < NOMEM 的（Claude，memory 还在害的）：

| task | NOMEM | V4 | Δ | 诊断 |
|------|-------|-----|----|------|
| novozymes | 24.53 | 6.60 | −17.92 | 自 V1 起从未修好（coverage gap）|
| **ventilator-pressure** | 37.50 | 15.39 | **−22.11** | V3 只 −3.69，V4 突然 −22 → caveat 反作用 |

Ventilator 的 V2→V4 跳水是 V3→V4 独有的新问题，要诊断。

### 5.2 Ventilator 反作用诊断

读 Claude V4 ventilator trajectory：
- memory block 里渲染了 idx 239 caveat："`training rows > 1M OR series count > 500` → 大 panel lag_52/104 超时"
- Ventilator 6M 行 × 75k breath series，**命中条件**
- 但 caveat 内容讲的是 retail panel (series_id = store × product)，**和 ventilator 的 breath_id sensor 完全不是一回事**
- Agent 被文本分散注意力 + 内容不适用 → 加剧 6M 行 timeout 挣扎，最终 degrade 到 50-bin lookup table（MAE 1.82 vs NOMEM LightGBM MAE 0.52）

**根因**：idx 239 caveat[0] 的 `condition` 写得太宽，误伤 non-retail 的大数据时序 task。

另一方面，ventilator retrieval 里 **idx 24**（m5 improvement about `roll_std_7 = groupby.shift.rolling`）在 improve phase 出现 3 次。这个 recipe 是 triple-stacked groupby+shift+rolling，对 6M 行 ventilator 直接 expensive，而且 "roll_std_7" 的 weekly 假设对 sub-second 呼吸信号毫无意义。之前没 caveat。

### 5.3 Novozymes 诊断

读 Claude V4 novozymes trajectory：retrieved entries 里全是 tabular/insurance/time-series（playground-s3e21、s3e25、afsis-soil、LANL、liberty-mutual、loan-default、stanford-covid-RNA...）。**没有一条 protein/sequence 领域**。每条 entry 都在推 tabular-ML recipe，agent 照搬 → LightGBM + k-mer + bigram → Kaggle pct 6.60 vs NOMEM 24.53。

**根因**：coverage gap。单条 caveat 难救，因为问题是"所有 retrieved entries 都不适用"而不是"某条 entry 误导"。

### 5.4 V5 patch 内容

1. **idx 239 caveat[0] condition 收紧**：
   - 从 `"training rows > 1M OR series count > 500"` 改成 `"retail/demand/sales panel data with calendar series_id (e.g. store x product x date) AND training rows > 1M AND unique series count > 500"`
   - caveat 内容末尾加一句 "applies ONLY to calendar-indexed retail/demand panel data — NOT to sub-second sensor signals, protein/DNA sequences, or event-level tabular data"
   - 预期：ventilator / novozymes 不再命中此 caveat

2. **idx 24 m5-forecasting-accuracy improvement 新 caveat**：
   - condition: "task is sub-second sensor signal or any time series WITHOUT daily/weekly calendar periodicity"
   - caveat: 警告 roll_std_7 的 weekly 假设不适用 + 建议 within-event rolling

3. **idx 1162 liberty-mutual draft_success 新 caveat**：
   - condition: "task is biological sequence data (protein, DNA, RNA) or molecular structure prediction"
   - caveat: 提示 insurance tabular recipes 不适用，建议用 BLOSUM / hydrophobicity / structural context

### 5.5 V5 结果

Claude V5 vs V4（hard，10 tasks）：

| task | V4 | V5 | Δv5-v4 | V5 vs NOMEM | 判断 |
|------|-----|-----|--------|-------------|------|
| **ventilator-pressure** | 15.39 | **26.22** | **+10.83** | −11.29 | ✅ idx 239 收紧 + idx 24 新 caveat 止血（还没完全修到 NOMEM）|
| **novozymes** | 6.60 | 3.10 | −3.50 | −21.43 | ❌ idx 1162 无效（coverage gap）|
| store-sales | 77.11 | **81.59** | **+4.48** | +13.06 | ✅ 收紧后仍然保持正效 |
| recruit-restaurant | N/A | 38.23 | — | −28.14 | 这次跑通但仍低 |
| B-group（7 tasks）| — | — | — | — | 方差范围 ±22 |

Easy（8 tasks）mean V4 63.72 → V5 60.24（−3.48），被 titanic −22.51 和 playground-s3e13 −15.24 这俩 B-group 方差摆动拖的。

V5 hard mean 53.68 vs V4 53.89（基本持平，但目标 task 有信号）。

## 6. 6 条 caveat 的逐条有效性审计

截至 V5 结束：

| idx | entry | 加入版本 | 有效性验证 |
|-----|-------|----------|-----------|
| 239 | walmart draft_success（大 panel + horizon）| V3 | ✅ V4 store-sales +66.79，V5 store-sales 仍 +13.06 vs NOMEM |
| 23 | m5-accuracy draft_success（recursive validation）| V3 | ✅ 和 239 联合起作用 |
| 175 | bnp-paribas draft_success（wide+LightGBM）| V4 | ❌ Gemini V4 ieee-fraud 无信号，Claude V5 也没影响 |
| 239[0] condition 收紧 | walmart | V5 | ✅ ventilator V4→V5 +10.83 |
| 24 | m5-accuracy improvement（roll_std_7 groupby+shift+rolling）| V5 | ✅（和 239 收紧联合）|
| 1162 | liberty-mutual draft_success（insurance vs biology）| V5 | ❌ novozymes 没反应（coverage gap 救不了）|

**4 条 Claude-derived caveat 全部有信号；1 条 Gemini-speculative 无效；1 条针对 coverage gap 无效**。

## 7. 跨模型泛化观察

同一批 caveat 在两个模型上的效果：

| caveat | Claude 有效？| Gemini 有效？|
|--------|-------------|--------------|
| idx 23 + 239（store-sales / recruit-restaurant）| ✅ store-sales +66, recruit +1 | ✅ recruit-restaurant V1→V4 +10.56（小但过 Gemini null 7.34）|
| idx 175（ieee-fraud）| — 未 target Claude | ❌ Gemini V4 ieee-fraud 无信号 |
| idx 24 + 239 收紧（ventilator）| ✅ Claude V4→V5 +10.83 | 未测（V5 没跑 Gemini）|
| idx 1162（novozymes）| ❌ | 未测 |

**Claude-derived caveat 在 Gemini 上也有效（recruit-restaurant）**，说明 scope caveat 有跨模型 transferability。

## 8. 方法论收获

### 8.1 "发现 memory 误导即加 caveat" 是对的迭代启发式
user 的原则比 "signal 超 null band 才加 caveat" 更能推动迭代。null band 分析用于**解读**，不用于 **gate**。idx 175 这种即使无效也应该先加，通过实验证明 → rollback 是正常流程。

### 8.2 单 run A/B 方差巨大，replay 是关键工程
Claude 的 B-group null stdev 27、Gemini 的 7.3，**没有 replay 把 retrieval 锁死，就没法判断 caveat 是否真有信号**。V3 随机抽样的结果几乎不可读，V4 起 replay 模式下 delta 才有意义。

### 8.3 Caveat condition 的宽度是双刃剑
idx 239 caveat[0] 从 "rows > 1M OR series > 500" 放到"retail panel + rows > 1M + series > 500" 把 ventilator 从 −19.35 救到 −11.29。condition 写得太宽 → 误伤无关 task；写得太窄 → 没人命中。

**经验法则**：condition 应该把 insight 生效所依赖的**domain 前提 + 数据规模前提**都显式写出来，让 agent 自己判断当前 task 是否满足。

### 8.4 Coverage gap ≠ scope 问题
novozymes 在 V1-V5 五个版本全被 memory 害（−17 ~ −22），但不是某条 entry 误导——是**所有 retrieved entries 都 off-domain**。这种情况下**单条 caveat 无效**，需要的是：
- 要么在 memory 里补 protein / sequence 领域的 entries
- 要么在 prompt header 加 meta-caveat："如果 retrieved entries 没覆盖当前任务 domain，视为弱先验"
- 要么在 retriever 层加 cosine threshold（低于阈值就返回空），让 agent fallback 到 no-mem

这三个方向都是 V6 以后的工作。

### 8.5 Caveat 需要覆盖 agent 决策发生的 phase
idx 1162 在 novozymes 只在 draft call 4/5 出现（后期），agent 已经锁定 LightGBM 路线。对 novozymes 无效有一部分是 caveat 触发时机太晚。类比：idx 239+23 对 store-sales 在 draft × 10 + final × 1 都触发，几乎每个决策点 agent 都能看到 → 有效。

**改进方向**：patched entries 最好能同时在 `draft_success` / `improvement` / `final_submission` 多个 entry_type 里出现，确保 caveat 在多个 phase 被 agent 看到。

## 9. 待办

- [ ] V5 recruit-restaurant V4→V5 实际仍 −28 vs NOMEM，读 V5 trajectory 看是否有**新 culprit**
- [ ] Novozymes coverage gap 的 prompt-header meta-caveat 方案
- [ ] 跨模型验证：把 V5 的新 caveat（idx 24, 1162, 239-tightened）在 Gemini 上跑一次看效果是否 transfer
- [ ] 更系统的 caveat 生成：写 critic LLM prompt，用现有 6 条 caveat 当 few-shot（3 条有效 + 3 条无效），让 LLM 自动生成新 caveat
- [ ] 清理：idx 175（无效 Gemini speculative）保留还是 rollback？—— 倾向保留作负例，等 critic LLM 训练时当反样本

## 10. 文件索引

- Memory 文件
  - 原始：`data/memory/cross_task_memory_teacher_v5_enriched.json`（不动，V1/V2 都读这个）
  - 带 caveat：`data/memory/cross_task_memory_teacher_v5_caveat_enriched.json`（V3+V4+V5 都读这个，in-place 更新）
  - Replay lookup：`data/memory/v2_retrieval_replay.json`（Claude V2 retrieval）、`gemini_v1_retrieval_replay.json`（Gemini V1 retrieval）
- 代码改动
  - `dsgym/agents/vgs/memory.py`：`MemoryEntry.scope_caveats` 字段
  - `dsgym/agents/vgs/smart_retriever.py`：`format_for_prompt` 渲染 caveat 行 + `__init__` 加 `replay_samples_path` + `retrieve()` 开头 replay 短路
  - `dsgym/agents/vgs/aide_agent.py`：读 `DSGYM_RETRIEVAL_LOG` + `DSGYM_REPLAY_SAMPLES` env var
  - `scripts/extract_v2_retrieval_samples.py`：从 trajectory 反向提取 retrieval replay JSON
- 运行 script
  - V3：`scripts/run_claude_sonnet_smartmem_memtest_v3.sh`
  - V4：`scripts/run_claude_sonnet_smartmem_memtest_v4.sh`（+ Gemini V4：`run_gemini_flash_smartmem_memtest_v4.sh`）
  - V5：`scripts/run_claude_sonnet_smartmem_memtest_v5.sh`
- 结果
  - `evaluation_results/claude_sonnet_smartmem_{easy,hard}_test_{v3,v4,v5}/`
  - `evaluation_results/gemini_flash_smartmem_{easy,hard}_test_v4/`
  - Retrieval debug dumps：`logs/retrieval_debug_*_v{3,4,5}.jsonl`
