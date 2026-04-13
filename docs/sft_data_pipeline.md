# 蒸馏轨迹 → SFT 训练数据 转换指南

## 概述

蒸馏（distillation）产出的是原始 trajectory JSON 文件，需要经过转换才能用于 SFT 训练。转换的核心操作是 **truncAF**：截断到最佳分数轮次 + 追加 final submission 轮。

## 文件位置

| 文件 | 说明 |
|------|------|
| 原始 trajectory | `evaluation_results/<experiment>/*_trajectory.json` |
| 转换脚本 | `scripts/convert_diverse_to_truncAF.py` |
| 输出 SFT 数据 | `data/sft/distill/` |

## 转换流程

### 输入：trajectory JSON

每个 trajectory 文件包含：
- `conversation`: 完整的 system/user/assistant 对话历史
- `task_memory`: 每轮的结构化摘要（step, model, score, best_score, notes）
- `final_best_score`: 最终最佳验证分数
- `success`: 是否成功完成

### 转换步骤（truncAF）

```
原始 trajectory → 找 best turn → 截断 → 清洗 → 追加 final submission → 输出 SFT JSON
```

#### 1. 找 best turn
从 `task_memory` 中找到 `score == final_best_score` 的轮次索引。**如果 best_turn_idx <= 1，跳过该 trajectory**（太短，学习信号不足）。

#### 2. 截断 conversation
保留从开头到 best turn 的所有 system/user/assistant 消息。

#### 3. 清洗消息

**User 消息：**
- 去掉 `[Step X/Y]` 标记
- 去掉 `=== TASK MEMORY ===` 到 `=== END TASK MEMORY ===` 之间的内容
- 去掉 `=== CROSS-TASK EXPERIENCE MEMORY ===` 到 `=== END CROSS-TASK MEMORY ===`
- 去掉 `--- Reference approach ... --- End reference ---`
- 去掉 `No previous attempts yet.`
- 去掉 `Model usage frequency across tasks:...`
- 对 `<information>` 内容做截断：如果包含 warning 且超过 1800 字符，截断到 1800

**Assistant 消息：**
- 转换为 V6 格式：只保留 `<goal>` + `<python>`
- 如果有 `<reasoning>` 代替 `<goal>`，将 `<reasoning>` 当作 `<goal>`
- 去掉 `<search_state>/<step>/<best_score>/<baseline_score>` 包裹
- 去掉 `<answer>` tag（包括未闭合的）
- 如果 `<python>` 未闭合（代码被截断），不补 `</python>`

#### 4. 删除坏 turn
如果某个 assistant 缺少 `<goal>` 或 `<python>`（代码被截断），删掉：
- 该 assistant 前面的 user（Step 指令）
- 该 assistant 本身
- 该 assistant 后面的 user（exec output）

#### 5. 合并连续 user
清洗和删除后可能产生连续的 user 消息，合并为一条。

#### 6. 追加 final submission 轮

**User 消息**：best turn 的执行输出（从完整 conversation 中取 best turn 之后的那个 user 的 `<information>` 块）+ `[ACTION: FINAL SUBMISSION]` 指令

```
<information>best turn 的执行输出</information>

[ACTION: FINAL SUBMISSION]

Generate your final submission now using your best approach. Save predictions to /submission/submission.csv.
```

**Assistant 消息**：取原始 trajectory 的最后一个 assistant（如果有 `<python>`）；如果最后一个 assistant 没有 `<python>`（比如只有 `<answer>`），fallback 到 best turn 的代码。

#### 7. 最终校验
- 每个 assistant 必须有 `<goal>` + `<python>`
- 不能有 `<search_state>/<step>/<best_score>/<baseline_score>/<reasoning>/<answer>/<information>`
- 不能有 `[Step X/Y]`
- 不能有 memory 泄漏
- 不能有连续 user
- best turn index == truncation point

## 使用方法

### 单个目录转换

```bash
python3 scripts/convert_diverse_to_truncAF.py \
    --input-dir evaluation_results/distill_gpt5_easy \
    --split easy \
    --output data/sft/distill/distill_gpt5_easy_truncAF.json
```

参数说明：
- `--input-dir`: 包含 `*_trajectory.json` 的目录
- `--split`: split 标签（写入 meta，用于后续区分来源）
- `--output`: 输出 JSON 文件路径

### 多个目录合并

分别转换后用 Python 合并：

```python
import json, glob

files = glob.glob('data/sft/distill/distill_*_truncAF.json')
all_data = []
for f in files:
    all_data.extend(json.load(open(f)))

with open('data/sft/distill/distill_all_truncAF.json', 'w') as fp:
    json.dump(all_data, fp, indent=2, ensure_ascii=False)
```

## 输出格式（LLaMA-Factory sharegpt）

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert data scientist..."},
    {"role": "user", "content": "**CHALLENGE NAME: ...**\n\n..."},
    {"role": "assistant", "content": "<goal>...</goal>\n\n<python>\n...\n</python>"},
    {"role": "user", "content": "<information>...</information>\n\n[ACTION: DRAFT]..."},
    {"role": "assistant", "content": "<goal>...</goal>\n\n<python>\n...\n</python>"},
    ...
    {"role": "user", "content": "<information>...</information>\n\n[ACTION: FINAL SUBMISSION]..."},
    {"role": "assistant", "content": "<goal>...</goal>\n\n<python>\n...\n</python>"}
  ],
  "meta": {
    "challenge_name": "playground-series-s3e3",
    "split": "easy",
    "teacher": "openai/gpt-5.2",
    "final_best_score": 0.828,
    "best_turn_idx": 4,
    "num_turns_truncated": 5,
    "num_turns_original": 12,
    "turns": [...]
  }
}
```

## 数据质量检查

转换后运行验证：

```python
import json, re

with open('data/sft/distill/xxx_truncAF.json') as f:
    data = json.load(f)

for s in data:
    for m in s['messages']:
        if m['role'] == 'assistant':
            c = m['content']
            assert '<goal>' in c, f"Missing <goal>"
            assert '<python>' in c, f"Missing <python>"
            assert '<search_state>' not in c
            assert '<reasoning>' not in c
            assert '<answer>' not in c
        if m['role'] == 'user':
            assert not re.search(r'\[Step \d+/\d+\]', m['content'])
            assert '=== TASK MEMORY' not in m['content']

    # No consecutive users
    for i in range(1, len(s['messages'])):
        assert not (s['messages'][i]['role'] == s['messages'][i-1]['role'] == 'user')
```

## 当前 SFT 数据一览

| 来源 | Split | 样本数 | 文件 |
|------|-------|--------|------|
| Claude Sonnet 4.6 (easy) | easy | 35 | `distill_claude_sonnet_easy_truncAF.json` |
| Claude Sonnet 4.6 (retry) | easy | 14 | `distill_claude_sonnet_retry_truncAF.json` |
| GPT-5.2 (easy) | easy | 38 | `distill_gpt5_easy_truncAF.json` |
| Gemini Flash (easy) | easy | 38 | `distill_gemini_flash_easy_truncAF.json` |
| Gemini Flash (mledojo) | mledojo | 33 | `distill_gemini_flash_mledojo_truncAF.json` |
| Qwen3-235B V6 (easy) | easy | 35 | `distill_qwen3_235b_v6_easy_truncAF.json` |
| Qwen3-235B V5 (mledojo) | mledojo | 54 | `distill_qwen3_235b_v5_mledojo_truncAF.json` |
| Qwen3-235B V6 (swap r1-r4) | swap | 65/65/62/67 | `distill_qwen3_235b_v6_swap_run{1-4}_truncAF.json` |
| Gemini Flash (mledojo remaining) | mledojo | 20 | `distill_gemini_flash_mledojo_remaining_truncAF.json` |
| Coder 480B (mledojo) | mledojo | 43 | `distill_coder480_mledojo_truncAF.json` |
| Qwen3-235B V6 (hard train) | hard | 23 | `distill_qwen3_235b_v6_hard_train_truncAF.json` |
| Coder 480B V6 (hard train) | hard | 28 | `distill_coder480_v6_hard_train_truncAF.json` |
| Gemini Flash (hard train) | hard | 19 | `distill_gemini_flash_hard_train_truncAF.json` |
| Claude Sonnet 4.6 (hard train) | hard | 22 | `distill_claude_sonnet_hard_train_truncAF.json` |
| GPT-5.2 (hard train) | hard | 36 | `distill_gpt5_hard_train_truncAF.json` |

## 过滤规则

转换时跳过的 trajectory：
1. `success=False` — 任务未成功完成
2. `final_best_score=None` — 没有产出有效分数
3. `best_turn_idx <= 1` — best score 在前两轮就达到了，学习信号不足
4. 最后一轮 assistant 没有 `submission.csv` — 不 fallback，直接跳过
5. assistant 缺少 `<python>`（代码被截断）— 删掉该 turn 和前后 user

## 注意事项

1. **不要用 swap 全部 4 轮**：后几轮因为 cross-task memory 富集，数据不 diverse。推荐只用 run1
2. **训练数据有重复问题**：如果 swap 占比太高（>50%），模型会学到 "I will build a simple yet effective baseline" 的模板化输出
3. **truncAF 包含失败轮次**：约 75% 的轮次是 score 下降或无 score 的——这是正确的，模型需要学会试错和恢复
4. **温度**：SFT 模型推理时建议用 `--temperature 0.7`，否则 temperature=0 容易 mode collapse（重复输出同一段代码）
5. **转换脚本不会修改原始 trajectory**：所有操作都是读取 → 转换 → 写新文件
