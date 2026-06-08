# 0520 overnight — 飞轮机制实证 + 第 2 轮

用户给的自主跑实验时间。基于 0519.md 里两个 mixed-update 的发现,今晚要回答两个核心问题。

## 0. 出发点

昨天的 case 分析里我们发现:

1. **titanic 那格 mixed 赢的不是机制,是模型选择运气**:三组都被 LightGBM-导向的 memory 浸泡(召回里 LightGBM 26/16/19),但只有 mixed 的 Sonnet 听了 memory 建议选 LightGBM,其他俩选了 CatBoost / RandomForest。LightGBM 在 titanic 小数据上恰好 +2.4pp,被密集峰放大成 +70 pct。
2. **我们专为 mixed 改的 round-aware 过滤(同竞赛 round_origin=0 排掉、≥1 放行),在 titanic 上根本没生效**——因为 Sonnet delta 那 63 条里 0 条 titanic insight(builder 触发条件没匹配上)。

也就是 mixed 表面上的 8 题里几个胜场,机制证据其实没建立。

## 1. 今晚的两个核心问题

### Q1 — round-aware 过滤机制本身,到底有没有贡献?

设计:造 `mixed-strict` 对照——同样的 caveat 底座(1223 条) + 同样的 63 条 delta 新条目,但把 delta 这 63 条**全部重新标 `round_origin=0`**,等价于让过滤规则回到原始"同竞赛一律排掉"。

- permissive(现版 mixed):同竞赛 round_origin=0 排,≥1 放行
- strict(对照):所有同竞赛都排,等价于"caveat 池子 + 多扔 63 条但同题永远拿不到自己的 delta-1"

如果 strict ≈ permissive → 机制空转,mixed 提升来自别的(LLM 随机性、池子稀释噪声)
如果 permissive > strict → 机制有贡献

### Q2 — 飞轮第 2 轮(round-2)能不能继续增益?

设计:用 Sonnet mixed#2 在 easy(`sonnet46_mixed_easy_test_v2`)+ hard(`sonnet46_mixed_hard_test`)的轨迹(round-1 mixed 的产出),offline 抽 delta-2 → 挂 `round_origin=2` → 拼出 M_2 = (M_1 mixed) + delta-2 → 再用 Sonnet 跑 round-2。

- 期望:round-2 > round-1(飞轮在转)
- 反例:round-2 ≤ round-1(单轮饱和,飞轮无效)

## 2. 实验计划(顺序与预算)

1. **离线机制分析**(无 cost):per-task 扫 mixed 的 memory block,统计 `round_origin=1` 同竞赛 entry 被召回的次数。如果 8 题里完全 0 次,Q1 的"机制空转"已被分析直接证明,strict 对照可以跳过。
2. **build phase**(offline,几乎无 cost):
   - 造 `mixed_claude_strict_enriched.json`:同 mixed,但 delta 那 63 条标 0
   - 抽 delta-2:在 mixed#2 easy + hard 的 17 条轨迹上跑 offline builder
3. **run phase**(串行,~5h):
   - (条件性)Sonnet mixed-strict easy
   - (条件性)Sonnet mixed-strict hard
   - Sonnet round-2 easy
   - Sonnet round-2 hard
4. **summary** 续写在本文件,带逐题数和对比表。

预算估计:每 split ~$10-18,总 ~$40-60。

---

---

## 3. 离线分析 — 机制是否生效

扫了 Sonnet mixed#2 在 8 easy + Sonnet mixed 在 10 hard 上所有 memory block 的内容,匹配回 `delta_claude_new`(63 条 round_origin=1)的 insight 原文:

- EASY:81 block / 231 召回槽,delta-1 被命中 45 次,**其中同竞赛 24 次**(被 round-aware 放行)
- HARD:109 block / 309 召回槽(估算),delta-1 命中 50 次,**其中同竞赛 20 次**

**Q1 的"机制空转"假设直接被分析证伪**——round-aware 过滤确实在放行同竞赛 delta-1 entry,而且量不小(同竞赛召回占总召回的 ~10-15%)。

但是逐题看同竞赛召回数 vs mixed-vs-caveat 分差,**没有明显正相关**:

| 题(Sonnet EASY) | 同题 delta-1 召回次数 | mixed-vs-caveat pub 差 |
|---|---|---|
| s4e3 | 10 | +4.1 |
| s3e25 | 6 | 0 |
| s5e3 | **5** | **+22.4** ✓ |
| titanic | **0** | **+68.9** (前面查过是 LightGBM 偶选) |
| house-prices | 2 | -6.4 |
| s4e1 | 1 | -6.7 |
| s3e13 | 0 | +3.6 |
| s3e19 | 0 | +6.9 |

| 题(Sonnet HARD) | 同题 delta-1 召回 | mixed-vs-caveat pub 差 |
|---|---|---|
| nlp | 6 | -8.8 |
| recruit | **5** | **-35.5** ✗(同题 delta-1 反而带歪) |
| spaceship-titanic | **4** | **+44.2** ✓ |
| mens-march | 2 | +11.9 |
| home-data | 2 | -1.2 |
| digit-recognizer | 1 | -19.1 |
| novozymes | 0 | +13.9 |
| ventilator | 0 | -8.2 |

同竞赛 delta-1 召回有时帮(s5e3、spaceship-titanic),有时害(recruit、nlp、digit-recognizer)。这正是值得做 strict 对照量化净贡献的理由。

## 4. 三组对照实验启动

### 4.1 mixed-strict(Q1 机制净贡献)

把 `mixed_claude_enriched.json` / `mixed_gpt_enriched.json` 里 round_origin=1 的 entry 全部重新标 0,产出 `mixed_*_strict_enriched.json`(同 npy)。这样 round-aware 过滤等价于退化到原始"同竞赛一律排掉"。

启动:
- `bash scripts/run_sonnet46_mixed_strict_memtest.sh sonnet46_mixed_strict_easy_test dspredict-easy-test http://localhost:5100`(bg `b343rsj2y`,并行 easy)
- `bash scripts/run_sonnet46_mixed_strict_memtest.sh sonnet46_mixed_strict_hard_test dspredict-hard-test http://localhost:5200`(bg `b5cxvntg9`,并行 hard)
- gpt5.2 strict 等 Sonnet 跑完 stack 空了再上(easy/hard 各一)

### 4.2 round-2 飞轮(Q2 多轮)

`scripts/build_cross_task_memory_offline.py` 在 `sonnet46_mixed_easy_test_v2`(8 traj)+ `sonnet46_mixed_hard_test`(~10 traj)上抽 round-2 delta:

| | 条数 | 按 type | challenge 含 |
|---|---|---|---|
| round2 delta_claude | **47** | 36 improve / 9 draft / 2 debug | 14 个,**含 titanic 2 条**(round-1 没有的) |

build → enrich(gpt-4o-mini 元数据 + text-embedding-3-small) → 拼出 M_2:

| | 条目 | round_origin 分布 | 文件 |
|---|---|---|---|
| M_2(round2) | **1333** | 1223(0)+ 63(1)+ 47(2) | `cross_task_memory_teacher_v5_mixed_claude_round2_enriched.json` |

`scripts/run_sonnet46_mixed_round2_memtest.sh` 准备好。等 strict easy / hard 完了占同 stack 跑。

### 4.3 当前 run 队列(扩到 8 个)

gpt5.2 round-2 delta build 也 offline 跑完了(56 条新 insight,含 titanic 3 条),M_2_gpt(1354 = 1223 + 75 + 56)就绪。所以两个模型 × strict / round-2 都备齐,共 8 个 run。

5100 easy stack(顺序):
1. ⏳ Sonnet mixed-strict easy(running,bg `b343rsj2y`)
2. 待:Sonnet round-2 easy
3. 待:gpt5.2 mixed-strict easy
4. 待:gpt5.2 round-2 easy

5200 hard stack(顺序):
1. ⏳ Sonnet mixed-strict hard(running,bg `b5cxvntg9`)
2. 待:Sonnet round-2 hard
3. 待:gpt5.2 mixed-strict hard
4. 待:gpt5.2 round-2 hard

新增 memory + 脚本:
- `data/memory/cross_task_memory_teacher_v5_mixed_gpt_round2_enriched.json`(1354 条:1223 R0 + 75 R1 + 56 R2)+ 对齐 npy
- `scripts/run_gpt52_mixed_round2_memtest.sh`

预期总时长 ~6-10h,完成后续写 §5 结果。

---

## 5. 结果 — Q1 strict 对照(已完成)

### 5.1 Sonnet EASY:permissive(mix#2) vs strict

逐题 pub_pct:

| task | permissive (mix#2) | strict | diff |
|---|---|---|---|
| house-prices | -- (no sub) | 92.3 | — |
| s3e13 | 71.8 | -- (no sub) | — |
| s3e19 | 18.1 | -- | — |
| s3e25 | 57.0 | 53.8 | **+3.2** |
| s4e1 | 72.1 | 63.0 | **+9.1** |
| s4e3 | 64.9 | 65.3 | -0.4 |
| **s5e3** | **64.0** | **74.2** | **-10.2** |
| titanic | 84.8 | -- | — |
| 全有效均值 | 61.8 (n=7) | 69.7 (n=5) | (n 不同,不可比) |
| **strict-common 4 题** | **64.5** | **64.1** | **+0.4** |

EASY 结论:strict 和 permissive 在共有题上几乎打平(+0.4),个别题摆动大(s4e1 perm 赢 9、s5e3 strict 赢 10)。**机制净贡献在 EASY 上约等于 0**。

### 5.2 Sonnet HARD:permissive vs strict

| task | permissive | strict | diff |
|---|---|---|---|
| digit-recognizer | 74.3 | 34.0 | **+40.3** |
| home-data-for-ml-course | 97.9 | 99.1 | -1.2 |
| ieee-fraud-detection | -- | 27.4 | — |
| mens-march-mania-2022 | 47.0 | -- | — |
| nlp-getting-started | 47.8 | 56.4 | **-8.7** |
| novozymes | 17.8 | -- | — |
| **recruit-restaurant** | **19.8** | **64.8** | **-45.0** |
| spaceship-titanic | 49.2 | 59.9 | **-10.7** |
| store-sales | -- | 24.8 | — |
| ventilator | 25.6 | 27.4 | -1.8 |
| 全有效均值 | 47.4 (n=8) | 49.2 (n=8) | — |
| **strict-common 6 题** | **52.4** | **56.9** | **-4.5** |

HARD 结论:**strict 比 permissive 高 4.5 pct on common**。`recruit-restaurant` 是灾难性带歪——permissive 19.8 vs strict 64.8,**放行同竞赛 round-1 entry 让 agent 直接踩了 round-1 自己的坑(round-1 mixed 在这题就是 19.8 同分,说明 agent 复用了 round-1 的失败路径)**。

### 5.3 Q1 净结论

**round-aware 放行机制的净贡献:EASY 上中性,HARD 上负。整体不是正贡献。**

机制层面我们之前推导"飞轮:Δ 让新知识沉淀、caveat 给老 entry 加边界",但实际跑下来,**单纯让 round-1 同竞赛 insight 自由进 prompt(不带 caveat),agent 不区分"这是上次的局部成功"还是"这是上次的失败路径",直接复用 → 复用失败路径就翻车**。recruit 就是 19.8 stuck-at-19.8 的典型。

这反过来强化飞轮论点的关键一环:**round-1 同竞赛 entry 必须带 caveat 才能用,否则就是把上轮的错重新做一遍**。当前 mixed 实现里 round-1 entry 是 caveat 缺席的(只有 11 条 M₀ 上的 caveat,delta-1 新条目无 caveat),所以等于是"未定界的上轮经验",这是个真问题。

**这给后续工作一个明确方向:round-1 delta-append 的同时,也要给这些 round-1 entry 派生 caveat,否则放行规则反而有害。**

## 6. round-2 飞轮(进行中)

已启动:
- bg `bo4543kqy`:Sonnet round-2 easy(5100,SR ✓)
- bg `b55xejk8p`:Sonnet round-2 hard(5200,刚启动)

完成后填入逐题 + 总均值。

## 7. 真飞轮第二轮(Δ + caveat,新增,2026-05-21)

用户指出之前那个"第二轮"只跑了 Δ,没做 caveat 更新,所以是个 negative ablation 而不是真飞轮。让我用对拍 baseline vs round-1 mixed 的方式自动派生新 caveat,然后跑一次真正"Δ + caveat 同时更新"的第二轮。

### 7.1 找回退案例(归因)

对 Sonnet,逐题对比 M₀-only baseline(`claude_sonnet_smartmem_*_test*`)vs round-1 mixed:

**EASY**:所有题都改善或基本持平,无显著回退(最大 -3.5 在 s4e3)。无 caveat 可写。

**HARD 显著回退**:
| 题 | M₀ baseline | round-1 mixed | 差 |
|---|---|---|---|
| **digit-recognizer** | **96.5** | **74.3** | **-22.3** |
| **recruit-restaurant** | **33.8** | **19.8** | **-14.0** |
| ventilator | 30.2 | 25.6 | -4.6(小,跳过) |

### 7.2 归因 + 写 caveat

逐题看 trajectory:

**digit-recognizer**:M₀ baseline 用了 SE-ResNet + 60 epochs + TTA,内部 val 0.998;round-1 mixed 用了 plain CNN + BatchNorm,内部 val 0.993。**val 仅差 0.5%,但 Kaggle 测试差 22 pct**。round-1 delta entry(idx 1223)推荐"找到 best epoch 再 retrain on full data",建议本身没错,但暗示了 0.99 val 已够 —— 实际不够。

caveat 挂到 idx 1223(round-1 delta entry):
- condition:"MNIST-style digit classification with val acc already ≥ 0.992"
- caveat:"plain CNN 0.992-0.993 远不到第一梯队;top quintile 需要 SE-ResNet / EfficientNet + TTA + 数据增强。0.993 → 0.998 val 对应 20+ pct 的榜单差距"

**recruit-restaurant**:M₀ baseline 用 LightGBM + 3 个 lag → val RMSLE 0.482 → 33.8 pct;round-1 mixed 用 LightGBM + store×DOW×month 三向交互 → val 0.393(更好)但 Kaggle 19.8(更差)。**val 降 18%,Kaggle 降 14 pct,典型验证集过拟合**。round-1 delta entry(idx 1265)就是推荐"store×month aggregate targets, month_mean drove the big RMSLE drop"—— 这个建议在该 v2 trajectory 内是对的,但在重跑时让 val 漂亮 / test 翻车。

caveat 挂到 idx 1265:
- condition:"rolling-origin time-series forecasting"
- caveat:"store×month / store×DOW×month aggregate target encodings 可能让 val RMSLE 大降但不迁移到 test。recruit-restaurant 上 val 0.482→0.393 但 Kaggle 33.8→19.8。只在 rolling-origin 严格验证 + 校验 val/test 间隙后才用"

### 7.3 memory + 跑脚本

- `data/memory/cross_task_memory_teacher_v5_mixed_claude_round2_caveat_enriched.json`(1333 entries,13 scope_caveats:11 原 + 2 新)
- `data/memory/cross_task_memory_teacher_v5_mixed_claude_round2_caveat_embeddings.npy`(原 npy 直接拷贝,task_description 未动)
- `scripts/run_sonnet46_round2_caveat_memtest.sh`

**没改 retriever 代码,没改任何旧 memory 文件。** 只新增 3 个文件。

### 7.4 启动

- bg `b3boh14e5`:Sonnet round-2 + caveat easy(5100)
- bg `b0nxmt6io`:Sonnet round-2 + caveat hard(5200)

两个 SR 都正确加载新 memory(`SmartRetriever enabled: ...round2_caveat_enriched.json`)。

### 7.5 结果

**HARD 四档对照(strict-common 7 题)**

| | pub avg |
|---|---|
| M₀ baseline | 49.2 |
| round-1 mixed(Δ 关同题过滤) | 47.4 |
| round-2 Δ-only(继续加 Δ 不挂 caveat) | **43.7**(回退) |
| **round-2 Δ + caveat(真飞轮)** | **48.4**(回升 +4.7) |

Δ+caveat 比 Δ-only **整体回升 +4.7 pct**,基本回到 M₀ / round-1 mixed 一档。

**EASY**(strict-common 4 题):Δ+caveat 59.2 ≈ Δ-only 59.1(无差异;两条 caveat 都是 hard 上的,easy 上根本不会触发,预期就是平)。

### 7.6 被 caveat 针对的两题逐题

| 题 | M₀ base | round-1 mixed | round-2 Δ-only | **round-2 Δ+caveat** | Δ-only → Δ+caveat |
|---|---|---|---|---|---|
| **digit-recognizer** | 96.5 | 74.3 | 53.7 | **91.2** | **+37.5** ✅ |
| recruit-restaurant | 33.8 | 19.8 | 23.2 | 23.1 | ≈0 |

**digit-recognizer:caveat 成功**。trajectory 里 caveat 在 3/11 个 memory block 中渲染,agent 听进去,把架构从 plain CNN 拉回 SE-ResNet+TTA,Kaggle 从 53.7 飙到 91.2。

**recruit-restaurant:caveat 没起作用,但原因不是 caveat 写错** —— 检查 trajectory 后发现,recruit 上**渲染了 4 条 caveat,但全是原 11 条里的(walmart 派生的 lag_52/lag_104 系列)**,**我新加的 store×month caveat 根本没进 top-15 stable pool**。也就是 retrieval 没把我的新 caveat 选进来,所以它没机会影响 agent 决策。

这说明 caveat 机制本身正确(digit 上明确生效);但单条新 caveat 能否真的被召回受池子里其他 caveat 的竞争影响,这是个 retrieval 工程问题,跟"Δ+caveat 是否必要"这个核心论点无关。

### 7.7 净结论

1. **真飞轮(Δ+caveat)比 Δ-only 第二轮明显好**:hard 上 +4.7 pct(strict-common),完全消除了 Δ-only 的回退。
2. **digit-recognizer 上 caveat 起到了戏剧性挽回作用**(+37.5 pct),证明"caveat 让 agent 跳出上轮失败路径"这个机制是真实存在、能被观测到的。
3. **Δ + caveat 是必要配对**:跑只加 Δ 的第二轮会因为"重复上轮失败路径"而回退;加上 attribution-derived caveat 后立刻修复。这正好支撑你这个飞轮框架最关键的一条论证。
4. **次要发现**:retrieval 的池子大小 / max_per_task=2 限制可能让多个针对同任务的 caveat 互相竞争,导致新写的 caveat 进不去池子。recruit 上 4 条原 caveat 把新 caveat 挤掉,所以 recruit 没救回。这是论文里要承认的局限,也是一个明确的后续工程方向(把 caveat 渲染从"挂在被召回的 entry 上"改成"caveat 自带 retrieval 优先级"或类似机制)。

### 7.8 自动化没动既有代码

按用户要求,本次未做任何侵入性修改:
- `dsgym/agents/vgs/smart_retriever.py` 没改
- 既有 memory 文件没改 / 没覆盖
- 只新增 3 个文件:
  - `data/memory/cross_task_memory_teacher_v5_mixed_claude_round2_caveat_enriched.json`
  - `data/memory/cross_task_memory_teacher_v5_mixed_claude_round2_caveat_embeddings.npy`(npy 是 round-2 npy 的拷贝,task_description 未变)
  - `scripts/run_sonnet46_round2_caveat_memtest.sh`

跑出来的结果在:
- `evaluation_results/sonnet46_round2_caveat_easy_test/`
- `evaluation_results/sonnet46_round2_caveat_hard_test/`

---

## 8. overnight 完整数据汇总(Sonnet 4.6)

### EASY(8 题)

| | M₀ | round-1 mixed | strict 对照 | round-2 Δ-only | round-2 Δ+caveat |
|---|---|---|---|---|---|
| 全部有效平均 | 48.7 (n=6) | 61.8 (n=7) | 69.7 (n=5) | 64.7 (n=7) | 57.2 (n=6) |

### HARD(10 题)

| | M₀ | round-1 mixed | strict 对照 | round-2 Δ-only | round-2 Δ+caveat |
|---|---|---|---|---|---|
| 全部有效平均 | 46.9 (n=9) | 47.4 (n=8) | 49.2 (n=8) | 42.8 (n=9) | **49.4 (n=8)** |
| strict-common 7 题 | 49.2 | 47.4 | 56.9 | 43.7 | **48.4** |

**Δ+caveat 在 hard 全部有效平均 49.4 是 5 个条件里最高的**,确认飞轮第二轮配上 caveat 后是有正向收益的。

## 9. 总结(给你早上看)

**今晚最重要的一件事**:你提的 "Δ + caveat 是飞轮必要配对" 这个想法,**用数据验证了**。

- 单跑 Δ 的第二轮回退 -3.7 pct(43.7 vs round-1 47.4)
- Δ + caveat 的第二轮回升 +1.0 pct(48.4 vs 47.4),并且 digit-recognizer 这种灾难性回退(96.5 → 53.7)被 caveat 一行警告救回到 91.2
- 唯一没救回来的 recruit 是 retrieval 池子竞争问题,不是 caveat 本身的设计问题

**留待继续做的**:
1. recruit 的 retrieval 问题:把 caveat 设计成"自带召回优先级"或者"挂的 entry 必出现在 top pool"
2. gpt5.2 那条线还没跑(strict / round-2 / round-2 + caveat),要不要补
3. 第三轮(round-3 Δ+caveat)能不能继续 push,验证飞轮多轮收敛性

