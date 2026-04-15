# Distill SFT Data

所有文件均为 truncAF 格式（截断到 best score + final submission），V6 tag（`<goal>` + `<python>`）。

转换脚本：`scripts/convert_diverse_to_truncAF.py`

## 单独来源文件

### Easy Split（4 teachers × 38 tasks）

| 文件 | Teacher | 样本 |
|------|---------|------|
| `distill_claude_sonnet_easy_truncAF.json` | Claude Sonnet 4.6 | 35 |
| `distill_claude_sonnet_retry_truncAF.json` | Claude Sonnet 4.6 (15 failed retry) | 14 |
| `distill_gpt5_easy_truncAF.json` | GPT-5.2 | 38 |
| `distill_gemini_flash_easy_truncAF.json` | Gemini 3 Flash | 38 |
| `distill_qwen3_235b_v6_easy_truncAF.json` | Qwen3-235B | 35 |

### Swap Split（Qwen3-235B × 4 rounds × 67 tasks）

| 文件 | Teacher | 样本 |
|------|---------|------|
| `distill_qwen3_235b_v6_swap_run1_truncAF.json` | Qwen3-235B (round 1) | 65 |
| `distill_qwen3_235b_v6_swap_run2_truncAF.json` | Qwen3-235B (round 2) | 65 |
| `distill_qwen3_235b_v6_swap_run3_truncAF.json` | Qwen3-235B (round 3) | 62 |
| `distill_qwen3_235b_v6_swap_run4_truncAF.json` | Qwen3-235B (round 4) | 67 |

⚠️ Round 2-4 因 cross-task memory 富集导致 diversity 下降，推荐只用 round 1。

### MLE Dojo Split（3 teachers × 60 tasks）

| 文件 | Teacher | 样本 |
|------|---------|------|
| `distill_qwen3_235b_v5_mledojo_truncAF.json` | Qwen3-235B (V5) | 54 |
| `distill_gemini_flash_mledojo_truncAF.json` | Gemini 3 Flash (前 34 个 task) | 33 |
| `distill_gemini_flash_mledojo_remaining_truncAF.json` | Gemini 3 Flash (后 26 个 task) | 20 |
| `distill_coder480_mledojo_truncAF.json` | Qwen3-Coder-480B (45/60 完成) | 43 |
| `distill_claude_sonnet_mledojo_truncAF.json` | Claude Sonnet 4.6 (59/60 完成) | 51 |

## 合并文件

| 文件 | Easy | Swap | MLE Dojo | Hard | 总 | 说明 |
|------|------|------|----------|------|-----|------|
| `distill_all_truncAF.json` | 160 | 259 (r1-r4) | 54 (235B) | - | 473 | 最早版本，swap 4 轮全包含 |
| `distill_diverse_truncAF.json` | 160 | 65 (r1) | 54 (235B) | - | 279 | 去掉 swap r2-r4 |
| `distill_diverse_v2_truncAF.json` | 160 | 65 (r1) | 150 (3 teacher) | - | 375 | mledojo 含全部 3 teacher |
| `distill_diverse_v3_truncAF.json` | 129 | 52 | 133 | - | 314 | v2 去掉 easy test + 无 submission fallback |
| `distill_diverse_v3_with_hard_truncAF.json` | 129 | 52 | 132 | 51 | 364 | v3 + 235B/Coder hard train |
| `distill_diverse_v4_truncAF.json` | 129 | 52 | 132 | 128 | 441 | v3_with_hard + Gemini/Claude/GPT hard train |
| **`distill_diverse_v5_truncAF.json`** | **129** | **52** | **183** | **128** | **492** | **推荐使用**。v4 + Claude Sonnet 4.6 mledojo (51) |

### v4 按 teacher 分布

| Teacher | 样本 |
|---------|------|
| Qwen3-235B | 141 |
| Gemini Flash | 101 |
| Coder 480B | 71 |
| GPT-5.2 | 66 |
| Claude Sonnet 4.6 | 62 |
| **合计** | **441** |

### v5 按 teacher 分布

| Teacher | 样本 |
|---------|------|
| Qwen3-235B | 141 |
| Claude Sonnet 4.6 | 113 |
| Gemini Flash | 101 |
| Coder 480B | 71 |
| GPT-5.2 | 66 |
| **合计** | **492** |
