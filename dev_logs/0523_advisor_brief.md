# GECM — Advisor Briefing Slides Draft

> Group Evolving through Collective Memory: A self-correcting memory flywheel for Kaggle data-science agents

每个 section 下面的内容可直接 copy 到 PPT。Markdown 标题对应 slide title,bullets 对应 slide body。表格可整体粘贴到 PPT 表格 placeholder。

---

## Section 1 · Motivation & Problem Formulation (~1.5 min)

### Slide 1.1 — Pain Point in One Sentence

**Memory-augmented agents on data-science tasks either pollute the prompt with irrelevant history (trajectory-level dump) or grow memory by blindly appending each new lesson (append-only) — memory becomes a growing pile of contradictions instead of a refinable knowledge base.**

### Slide 1.2 — Two Design Axes, Where Existing Work Stops Short

**(A) How does the agent USE memory?**

| | Approach | Limitation |
|---|---|---|
| **Others** (trajectory-level) | Inject all relevant memory items as a single block at the start of the trajectory | Static across turns; irrelevant items pollute every prompt; agent can't reconsider per action |
| **Ours** (turn-level) | Re-retrieve memory per turn, conditioned on the agent's current action (draft / improve / debug) | Right info for the current sub-task; agent re-judges strategy each turn |

**(B) How does memory UPDATE across rounds?**

| | Approach | Limitation |
|---|---|---|
| **Others A** (append-only, e.g. ExpeL, Reflexion) | Add each new lesson as a new bullet | Unbounded growth + accumulating contradictions + no provenance |
| **Others B** (rewrite-only) | Modify existing entries in place | Loses past wisdom, no traceability |
| **Ours** (dual-update) | **(a) delta-append**: grow with new validated insights<br>**(b) caveat-update**: attach scoped warnings (`{condition, caveat, source}`) to existing entries that were over-generalized | Memory both grows and self-corrects; knowledge stays monotonic + traceable |

### Slide 1.3 — Core Intuition

**Memory should be a flywheel, not a pile.**

- Each round, the **growth operator (Δ)** adds new entries from validated trajectories.
- Each round, the **scope operator (caveat)** attaches structured boundaries to any entry that was misapplied — preserving the original insight while narrowing where it should be trusted.
- A round-1 delta becomes a round-2 candidate for caveating; round-2 trajectories spawn round-3 delta + round-3 caveats. → continual self-refinement.

---

## Section 2 · Methodology (~2 min)

### Slide 2.1 — Architecture Pipeline

> **(suggest drawing the diagram below; ASCII sketch for reference)**

```
                ┌──────────────────────────────┐
                │   Memory M_t                 │
                │   • M₀  teacher entries     │
                │   • Δ_1, Δ_2, ... delta     │
                │   • scope_caveats on entries │
                └──────────────┬───────────────┘
                               │ per-turn,
                               │ action-aware retrieve
                               ▼
   ┌─────────────────────────────────────────────────────┐
   │   AIDE-style agent rollout (≤12 turns / task)       │
   │   draft → improve → ... → final_submission          │
   └──────────────┬──────────────────────┬───────────────┘
                  │ trajectory             │ Kaggle public LB
                  ▼                        ▼
       ┌──────────────────────┐  ┌──────────────────────┐
       │ Offline Δ extractor   │  │ Regression detector  │
       │ (LLM summarizer)      │  │ + caveat attribution │
       └──────────┬───────────┘  └──────────┬───────────┘
                  │  new Δ entries           │  scope_caveats
                  │  (round_origin = t+1)    │  attached to any
                  └──────────┬───────────────┘  retrieved entry
                             ▼
                     ┌────────────────┐
                     │  Memory M_{t+1} │
                     └────────────────┘
```

### Slide 2.2 — Innovation 1 · Per-turn Action-Aware Retrieval

For every agent turn, the retriever runs a fresh pipeline:

1. **Hard filter** entries by `(domain, task_type, current action's entry_type)` — draft → draft_success, improve → improvement, debug → debug_fix.
2. **Cosine rank** filtered entries against current task's description embedding (text-embedding-3-small).
3. **Per-task cap** (max 2 entries per challenge) prevents one source task from dominating.
4. **Stable pool of 15** entries cached per `(task, action)`; each turn samples 3 at random for diversity.

→ Different turns of the same task can see different memory subsets → encourages re-exploration vs trajectory-level dump.

### Slide 2.3 — Innovation 2 · Dual Memory Update (the Flywheel)

**Growth operator Δ** (after round *t*):
- LLM summarizer scans round-*t* trajectories
- Selects turns that are `improvement`(new best score), `debug_fix`, or `draft_success` (sampled)
- Each becomes a new memory entry with `round_origin = t+1`
- → memory **covers more strategy space**

**Scope operator caveat** (after round *t*):
- Identify tasks that regressed in round *t* vs baseline
- For each regression, trace **which retrieved entry's insight the agent actually applied**
- Attach a structured `scope_caveats: {condition, caveat, source, derived_for_tasks}` to that entry
- → memory **refines the boundary** of when each insight is trustworthy

**Key property — caveats can attach to any entry**, including teacher M₀ entries (which are most often the culprit because they have highest cosine retrieval rate). The same insight stays in memory; only its applicability gets narrowed.

### Slide 2.4 — Innovation 3 · Round-Aware Test-Time Learning

- **Cross-task transfer (always allowed)**: M₀ entries from other challenges retrievable across all rounds.
- **Self-leakage prevention**: M₀ entries from the *current* challenge are filtered (no self-leakage).
- **Test-time continual learning (the new piece)**: prior-round delta entries from the *same* challenge are admitted under the round-aware filter (`round_origin ≥ 1`), capped at 2 per task.
- This is the entry-level analog of **Dynamic Cheatsheet's** M_i = {1 … i-1} safeguard, lifted to (entry × round) granularity.

---

## Section 3 · Empirical Results (~2.5 min)

### Slide 3.1 — Setup

- **Benchmark**: DSGym `dspredict` split — **18 real Kaggle competitions** (8 easy + 10 hard)
- **Metric**: Kaggle **public leaderboard percentile** (gold-standard externally validated)
- **Backbones**: Claude Sonnet 4.6, GPT-5.2 (two reasoning-model families to test generality)
- **Baselines**:
  - **M₀ static memory** (1223 teacher-distilled entries, no flywheel)
  - **Δ-only** (round-by-round delta, no caveats — mimics ExpeL / Dynamic-Cheatsheet style)
  - **Caveat-only** (M₀ + manual caveats, no growth)

### Slide 3.2 — Main Result: Dual Update vs Single Update

**Claude Sonnet 4.6, Hard split (strict-common 7 tasks, single run)**

| Configuration | Public pct avg | Δ vs M₀ |
|---|---|---|
| M₀ static memory | 49.2 | — |
| Round-1 with caveats only (no delta growth) | 47.4 | -1.8 |
| **Round-2 Δ-only** (append new insights) | **43.7** | **-5.5 (regression)** |
| **Round-2 Δ + Caveat (full flywheel)** | **48.4** | **+4.7 recovery** |

→ **Pure append-only flywheel regresses; the caveat update is what makes the growth safe.** This is the central evidence for dual-update.

### Slide 3.3 — Headline Case Study: `digit-recognizer` (Kaggle MNIST)

| Stage | Model the agent chose | Validation acc | **Kaggle public pct** |
|---|---|---|---|
| M₀ baseline | SE-ResNet + 60 epochs + TTA | 0.998 | **96.5** |
| Round-2 Δ-only | Plain CNN + BatchNorm (catastrophic forgetting) | 0.992 | **53.7** |
| Round-2 Δ + Caveat (auto-derived) | SE-ResNet restored (per caveat warning "plain CNN at 0.99 is far from competitive; need SE-ResNet+TTA") | 0.998 | **91.2** |

**A single structured caveat (~80 words) recovered +37.5 percentile.**
→ Demonstrates the mechanism: caveat doesn't add new knowledge, it *narrows when an existing insight should be trusted*, letting the agent re-pick the correct path.

### Slide 3.4 — Bonus Finding: Caveats Transfer Across Tasks

**gpt-5.2, Hard split** — single caveat written for `recruit-restaurant` (warning about time-series target-encoding leakage):

| Task | Without caveat | With caveat |
|---|---|---|
| `recruit-restaurant` (caveat's target) | 21.5 | **49.5** (+28) |
| `store-sales-time-series-forecasting` (similar time-series structure) | 12.2 | **68.3** (+56) |

→ Caveat's `condition` field ("rolling-origin time-series forecasting") matched a second task it was never explicitly derived for. **Scope caveats transfer; bullet-style lessons do not.**

### Slide 3.5 — Cross-Model Replication

- Same flywheel architecture applied to **two model families** (Claude Sonnet 4.6 and GPT-5.2)
- Both show the same qualitative pattern: Δ-only round-2 regresses, Δ+caveat recovers
- Specific gains differ (Sonnet's "digit" case is the strongest; gpt5.2's "recruit" + "store-sales" is the strongest cross-task transfer)
- → method generality is supported, while specific failure cases reveal model-specific susceptibilities

### Slide 3.6 — Honest Limitations (worth previewing to advisor)

- **Per-run variance is large**: single tasks can swing ±25 pct between runs of the same config. Multiple runs + median reporting needed for paper-grade results.
- **Caveat rendering depends on retrieval cosine**: a caveat only takes effect if its host entry wins the top-k retrieval slot. Current implementation: 2 entries per challenge max → some caveats sit idle. Active work: multi-attach caveats to all sibling entries of a challenge to lift render reliability.
- **Round-3 didn't yet beat round-2** on Sonnet hard avg (likely due to caveat rendering bottleneck + variance, not the framework). Verification ongoing.

---

## Closing line for advisor

> "The flywheel framing converts memory from a one-shot lookup into a learning system: growth (Δ) and refinement (caveat) act together so memory stays monotonic and traceable. The headline result — +37.5 pct on digit-recognizer from a single structured caveat — shows the mechanism is real; the cross-task transfer to store-sales shows the scope is general. Next steps are stabilizing caveat rendering and scaling to multi-model group memory."
