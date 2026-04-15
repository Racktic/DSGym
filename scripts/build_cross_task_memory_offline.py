"""
Build cross-task memory offline from teacher trajectories.

Walks each teacher trajectory, identifies "insight-worthy" turns:
  - improvement: action=improve, not buggy, best_score changed (new best)
  - debug_fix:   action=debug,   not buggy
  - draft_success: action=draft, not buggy, sampled with --draft-prob

For each trigger, calls a strong summarizer LLM (e.g. Claude Sonnet 4.6 via LiteLLM)
to produce a concise "insight" string. Writes a JSON list of MemoryEntry-compatible dicts.

Usage:
  export LITELLM_API_KEY=xxx
  python scripts/build_cross_task_memory_offline.py \
      --input-dirs evaluation_results/distill_* \
      --exclude-glob '*swap*' \
      --summarizer-model openai/claude-sonnet-4.6 \
      --base-url https://litellm.nbdevenv.xiaoaojianghu.fun \
      --output data/memory/cross_task_memory_teacher.json \
      --max-workers 16 \
      --draft-prob 0.25
"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add repo root for imports
DSGYM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, DSGYM_ROOT)

from dsgym.agents.vgs.memory import detect_model_type  # noqa: E402

try:
    from openai import OpenAI
except ImportError:
    print("Need: pip install openai")
    sys.exit(1)


# -----------------------------
# Trigger detection
# -----------------------------

def _is_buggy(exec_output: str) -> bool:
    """Heuristic: detect whether a turn's exec_output indicates an error."""
    if not exec_output:
        return True  # no output = code likely never ran cleanly
    # Common error markers
    err_markers = [
        "Traceback (most recent call last)",
        "[stderr] Traceback",
        "No python code found",
        "Error during execution:",
        "KernelError",
        "TimeoutError",
    ]
    return any(m in exec_output for m in err_markers)


def _extract_goal(raw_response: str) -> str:
    m = re.search(r"<goal>(.*?)</goal>", raw_response or "", re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"<reasoning>(.*?)</reasoning>", raw_response or "", re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_python(raw_response: str) -> str:
    m = re.search(r"<python>(.*?)</python>", raw_response or "", re.DOTALL)
    return m.group(1).strip() if m else ""


def _truncate(s: str, n: int) -> str:
    if not s or len(s) <= n:
        return s or ""
    return s[:n] + f"\n... (truncated, total {len(s)} chars)"


# -----------------------------
# Triggers
# -----------------------------

def find_triggers(
    traj: Dict[str, Any],
    draft_prob: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Walk a trajectory and emit a list of (turn_idx, entry_type, context) triggers."""
    task_memory = traj.get("task_memory", []) or []
    turns = traj.get("turns", []) or []
    if not task_memory or not turns:
        return []
    # Align by index — same length expected
    n = min(len(task_memory), len(turns))

    # First pass: identify all turns that established a *new* best.
    # Conditions: score == best_score (this turn's score IS the current best)
    #             AND best_score != prev_best (it's a fresh best, not a re-achievement)
    #             AND score is not 0 (likely summary-LLM hallucination / default)
    #             AND not buggy
    # The first such turn is the initial baseline; the 2nd, 3rd, ... are improvements.
    best_chain = []
    for i in range(n):
        tm = task_memory[i]
        tr = turns[i]
        score = tm.get("score")
        best_score = tm.get("best_score")
        prev_best = task_memory[i - 1].get("best_score") if i > 0 else None
        if (
            score is not None
            and best_score is not None
            and abs(score) > 1e-9   # exclude 0.0 (likely hallucinated)
            and abs(best_score) > 1e-9
            and abs(score - best_score) < 1e-9
            and (prev_best is None or abs(best_score - prev_best) > 1e-9)
            and not _is_buggy(tr.get("execution_output", "") or "")
        ):
            best_chain.append(i)
    improvement_indices = set(best_chain[1:])  # all but the first
    ref_for = {}  # improvement_idx -> reference (previous best turn) idx
    for k in range(1, len(best_chain)):
        ref_for[best_chain[k]] = best_chain[k - 1]

    triggers = []
    for i in range(n):
        tm = task_memory[i]
        tr = turns[i]
        action = tr.get("phase") or "unknown"
        exec_output = tr.get("execution_output", "") or ""
        buggy = _is_buggy(exec_output)
        score = tm.get("score")
        best_score = tm.get("best_score")

        if buggy:
            continue  # only successful turns

        # Exclude turns whose score is exactly 0.0 (likely summary-LLM hallucination).
        # `None` is fine — only a numeric 0 is suspicious.
        if score is not None and abs(score) < 1e-9:
            continue

        entry_type = None
        # Priority: improvement > debug_fix > draft_success
        if i in improvement_indices:
            entry_type = "improvement"
        elif action == "debug":
            entry_type = "debug_fix"
        elif action == "draft":
            if rng.random() < draft_prob:
                entry_type = "draft_success"

        if entry_type is None:
            continue

        # For improvement: pull reference (previous best turn) data
        ref_turn_idx = None
        ref_raw_response = ""
        ref_exec_output = ""
        ref_notes = ""
        ref_score = None
        if entry_type == "improvement":
            ref_turn_idx = ref_for.get(i)
            if ref_turn_idx is not None:
                rtm = task_memory[ref_turn_idx]
                rtr = turns[ref_turn_idx]
                ref_raw_response = rtr.get("raw_response", "")
                ref_exec_output = rtr.get("execution_output", "") or ""
                ref_notes = rtm.get("notes", "")
                ref_score = rtm.get("score")

        triggers.append({
            "turn_idx": i,
            "entry_type": entry_type,
            "action": action,
            "score": score,
            "best_score": best_score,
            "prev_best": ref_score,
            "raw_response": tr.get("raw_response", ""),
            "exec_output": exec_output,
            "model_notes": tm.get("notes", ""),
            "tm_model": tm.get("model", ""),
            # Reference (only for improvement)
            "ref_turn_idx": ref_turn_idx,
            "ref_raw_response": ref_raw_response,
            "ref_exec_output": ref_exec_output,
            "ref_notes": ref_notes,
        })
    return triggers


# -----------------------------
# LLM summarization
# -----------------------------

INSIGHT_SYSTEM_PROMPT = """You are an experienced ML engineer. Given a single turn from another agent's trajectory on a Kaggle-style ML task, write ONE short, actionable insight (2-3 sentences) that a different agent could use as a hint to improve on similar tasks. Focus on what specifically worked or what specifically was fixed. Avoid generic advice.

Output ONLY the insight text. No headers, no XML, no markdown. Keep it under 60 words."""


PROMPT_BY_TYPE = {
    "improvement": """An agent improved its score on this task.

# Task
{task_description}

# Previous best (score = {prev_best}) — the reference being improved upon
## Reference plan
{ref_plan}

## Reference code
```python
{ref_code}
```

## Reference execution output
{ref_exec_output}

## Reference notes
{ref_notes}

# This turn (new best, score = {score})
## Plan
{plan}

## Code
```python
{code}
```

## Execution output
{exec_output}

## Model's own notes
{notes}

Write ONE actionable insight: what specific change between the reference and this turn caused the score to improve from {prev_best} to {score}? Be concrete (e.g., a new feature, a model swap, a hyperparameter, a fix). 2-3 sentences.""",

    "debug_fix": """An agent fixed a bug in this turn.

# Task
{task_description}

# What the agent's plan was
{plan}

# Code that ran successfully
```python
{code}
```

# Execution output
{exec_output}

# Model's own notes
{notes}

Write ONE actionable insight: what was the underlying bug, and how was it fixed? Be concrete (e.g., wrong API call, dtype mismatch, column alignment). 2-3 sentences.""",

    "draft_success": """An agent drafted a working solution for this task.

# Task
{task_description}

# What the agent's plan was
{plan}

# Code
```python
{code}
```

# Execution output
{exec_output}

# Model's own notes
{notes}

Write ONE actionable insight: what's the key idea of this draft worth referencing? (model choice, feature engineering, validation strategy). 2-3 sentences.""",
}


def call_summarizer(
    client: OpenAI,
    model: str,
    task_description: str,
    trigger: Dict[str, Any],
    max_retries: int = 3,
) -> Optional[str]:
    """Call summarizer LLM to produce an insight string."""
    code = _extract_python(trigger["raw_response"])
    plan = _extract_goal(trigger["raw_response"])
    fmt_kwargs = dict(
        task_description=task_description,
        prev_best=trigger.get("prev_best"),
        score=trigger.get("score"),
        plan=plan,
        code=code,
        exec_output=trigger["exec_output"],
        notes=trigger["model_notes"],
    )
    if trigger["entry_type"] == "improvement":
        ref_code = _extract_python(trigger.get("ref_raw_response", ""))
        ref_plan = _extract_goal(trigger.get("ref_raw_response", ""))
        fmt_kwargs.update(
            ref_plan=ref_plan,
            ref_code=ref_code,
            ref_exec_output=trigger.get("ref_exec_output", ""),
            ref_notes=trigger.get("ref_notes", ""),
        )
    prompt = PROMPT_BY_TYPE[trigger["entry_type"]].format(**fmt_kwargs)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  [retry {attempt+1}/{max_retries}] LLM error: {e!s}; sleeping {wait}s")
            time.sleep(wait)
    return None


# -----------------------------
# Per-trajectory worker
# -----------------------------

def process_trajectory(
    traj_path: str,
    summarizer_model: str,
    base_url: str,
    api_key: str,
    draft_prob: float,
) -> List[Dict[str, Any]]:
    """Worker function: process one trajectory file, return list of memory entries."""
    try:
        with open(traj_path) as f:
            traj = json.load(f)
    except Exception as e:
        print(f"  [skip] failed to load {traj_path}: {e}")
        return []

    if not traj.get("success"):
        return []
    final_best = traj.get("final_best_score")
    if final_best is None:
        return []

    # Find best_turn_idx
    task_memory = traj.get("task_memory", []) or []
    best_turn_idx = None
    for i, tm in enumerate(task_memory):
        s = tm.get("score")
        if s is not None and abs(s - final_best) < 1e-9:
            best_turn_idx = i
            break
    if best_turn_idx is None or best_turn_idx <= 1:
        return []

    # Get task description (from first user message)
    conv = traj.get("conversation", [])
    task_description = ""
    for m in conv:
        if m["role"] == "user":
            task_description = m["content"]
            break

    challenge_name = traj.get("challenge_name", "")
    teacher_model = traj.get("model", "")
    rng = random.Random(hash((traj_path, "rng")) & 0xFFFFFFFF)

    triggers = find_triggers(traj, draft_prob, rng)
    if not triggers:
        return []

    client = OpenAI(api_key=api_key, base_url=base_url)
    entries = []
    for trig in triggers:
        insight = call_summarizer(
            client, summarizer_model, task_description, trig
        )
        if not insight:
            continue
        plan_text = _extract_goal(trig["raw_response"])
        code_text = _extract_python(trig["raw_response"])
        entry = {
            "challenge_name": challenge_name,
            "task_description": task_description,
            "turn": trig["turn_idx"] + 1,
            "action": trig["action"],
            "plan": plan_text,
            "model_type": detect_model_type(code_text or plan_text or trig["tm_model"]),
            "score": trig["score"],
            "score_improved": trig["entry_type"] == "improvement",
            "buggy": False,
            "insight": insight,
            "entry_type": trig["entry_type"],
            "timestamp": datetime.now().isoformat(),
            "teacher_model": teacher_model,
            "source_traj": os.path.basename(traj_path),
        }
        entries.append(entry)
    return entries


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dirs", nargs="+", required=True,
                   help="Glob patterns for input dirs (e.g. evaluation_results/distill_*)")
    p.add_argument("--exclude-glob", default="*swap*",
                   help="Filename/dirname substring to skip (default: '*swap*')")
    p.add_argument("--summarizer-model", default="openai/claude-sonnet-4.6")
    p.add_argument("--base-url", default="https://litellm.nbdevenv.xiaoaojianghu.fun")
    p.add_argument("--api-key", default=os.environ.get("LITELLM_API_KEY", ""),
                   help="API key (default: $LITELLM_API_KEY)")
    p.add_argument("--output", required=True, help="Output JSON file path")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--draft-prob", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N trajectories (for testing)")
    args = p.parse_args()

    if not args.api_key:
        print("ERROR: --api-key or $LITELLM_API_KEY required")
        sys.exit(1)

    # Discover all trajectory files
    all_traj_paths: List[str] = []
    exclude = args.exclude_glob.strip("*")
    for input_pat in args.input_dirs:
        for d in glob.glob(input_pat):
            if exclude and exclude in d:
                continue
            for f in sorted(glob.glob(os.path.join(d, "*_trajectory.json"))):
                if exclude and exclude in os.path.basename(f):
                    continue
                all_traj_paths.append(f)

    print(f"Discovered {len(all_traj_paths)} trajectory files")
    if args.limit:
        all_traj_paths = all_traj_paths[: args.limit]
        print(f"Limited to first {args.limit}")

    if not all_traj_paths:
        print("No trajectories found, exiting")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Process in parallel
    all_entries: List[Dict[str, Any]] = []
    n_done = 0
    n_total = len(all_traj_paths)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(
                process_trajectory,
                tp,
                args.summarizer_model,
                args.base_url,
                args.api_key,
                args.draft_prob,
            ): tp
            for tp in all_traj_paths
        }
        for fut in as_completed(futures):
            tp = futures[fut]
            try:
                entries = fut.result()
                all_entries.extend(entries)
                n_done += 1
                elapsed = time.time() - t0
                eta = elapsed / n_done * (n_total - n_done) if n_done else 0
                print(
                    f"[{n_done}/{n_total}] {os.path.basename(tp):<70} "
                    f"+{len(entries)} entries (total {len(all_entries)}, "
                    f"elapsed {elapsed:.0f}s, eta {eta:.0f}s)"
                )
            except Exception as e:
                n_done += 1
                print(f"[{n_done}/{n_total}] FAILED {tp}: {e!s}")

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    print(
        f"\nDone. Wrote {len(all_entries)} memory entries to {args.output} "
        f"(elapsed {time.time()-t0:.1f}s)"
    )

    # Per-type breakdown
    from collections import Counter
    by_type = Counter(e["entry_type"] for e in all_entries)
    by_teacher = Counter(e["teacher_model"] for e in all_entries)
    by_challenge = Counter(e["challenge_name"] for e in all_entries)
    print("\nBy entry_type:", dict(by_type))
    print("By teacher (top 6):", dict(by_teacher.most_common(6)))
    print(f"By challenge (top 8 of {len(by_challenge)}):", dict(by_challenge.most_common(8)))


if __name__ == "__main__":
    main()
