#!/usr/bin/env python3
"""Extract the actual memory entries V2's agent saw from V2 trajectories.

For each trajectory file under V2's output dirs, parse each rendered
CROSS-TASK MEMORY block and match the insight snippets back to entry
indices in the v5 memory JSON. The resulting lookup can be replayed
from SmartRetriever to pin cross-run retrieval to V2's exact samples.

Output JSON structure:
  {
    "<challenge_name>": {
      "<action>": [
        [idx, idx, idx],       # turn 0 entries (in pool-rank order)
        [idx, idx, idx],       # turn 1
        ...
      ],
      ...
    },
    ...
  }

Usage:
  python scripts/extract_v2_retrieval_samples.py \
      --memory data/memory/cross_task_memory_teacher_v5_enriched.json \
      --trajectories evaluation_results/claude_sonnet_smartmem_hard_test_v2 \
                     evaluation_results/claude_sonnet_smartmem_easy_test_v2 \
      --output data/memory/v2_retrieval_replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


MEM_BLOCK_RE = re.compile(
    r"=== CROSS-TASK EXPERIENCE MEMORY ===(.*?)=== END CROSS-TASK MEMORY ===",
    re.DOTALL,
)
TASK_HEADER_RE = re.compile(r"^--- Task: (.+?) ---\s*$")
ENTRY_LINE_RE = re.compile(
    r"^\s+\[(?P<tag>[A-Z_]+)\] Model: (?P<model>.*?) \| Score: (?P<score>\S+) \| (?P<insight>.+?)\s*$"
)
TAG_TO_TYPE = {
    "SUMMARY": "task_summary",
    "IMPROVED": "improvement",
    "DEBUG_FIX": "debug_fix",
    "DRAFT": "draft_success",
    "INFO": "turn",
}


def build_insight_index(memory_entries: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], int]:
    """Index memory entries by (challenge_name, entry_type, insight-prefix) -> list-index.

    The insight prefix (~first 200 chars) is unique enough within a (challenge, entry_type)
    triple to disambiguate entries.
    """
    idx: Dict[Tuple[str, str, str], int] = {}
    for i, e in enumerate(memory_entries):
        cn = e.get("challenge_name", "")
        et = e.get("entry_type", "turn")
        ins = (e.get("insight") or "").strip()
        key = (cn, et, ins[:200])
        # if collision (same prefix in same (cn, et)), prefer lower index (first occurrence)
        idx.setdefault(key, i)
    return idx


def parse_memory_block(block: str) -> List[Tuple[str, str, str]]:
    """Parse a rendered CROSS-TASK MEMORY block. Returns list of (challenge, entry_type, insight)."""
    out: List[Tuple[str, str, str]] = []
    current_task: str = ""
    lines = block.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = TASK_HEADER_RE.match(line)
        if m:
            current_task = m.group(1).strip()
            i += 1
            continue
        m2 = ENTRY_LINE_RE.match(line)
        if m2:
            tag = m2.group("tag")
            insight = m2.group("insight").strip()
            # an insight may continue onto next lines before the next `  [TAG]` or `--- Task:` or blank
            # but our format_for_prompt doesn't wrap, so just take the single-line insight.
            out.append((current_task, TAG_TO_TYPE.get(tag, "turn"), insight))
        i += 1
    return out


def _extract_action_from_prompt(prompt_text: str) -> str:
    """Heuristic: figure out which action (draft/improve/debug/final_submission) triggered this retrieval."""
    # The DSGym step prompt starts with "[Step N/M]\n[ACTION: X ..."
    m = re.search(r"\[ACTION:\s*([A-Z_ ]+?)[\s\-—]", prompt_text)
    if m:
        name = m.group(1).strip().lower()
        # map to action keyword
        if name.startswith("draft") or "draft" in name:
            return "draft"
        if name.startswith("improve") or name.startswith("exploit"):
            return "improve"
        if name.startswith("debug") or "debug" in name:
            return "debug"
        if "final" in name or "submission" in name:
            return "final_submission"
    # Fallback: look for common tokens
    for kw, canon in [("DRAFT", "draft"), ("IMPROVE", "improve"), ("DEBUG", "debug"), ("FINAL", "final_submission")]:
        if kw in prompt_text[:500].upper():
            return canon
    return "unknown"


def extract_from_trajectory(
    traj_path: str,
    insight_index: Dict[Tuple[str, str, str], int],
) -> Dict[str, Dict[str, List[List[int]]]]:
    """Walk a single trajectory; return {challenge: {action: [[idx,...], ...]}}."""
    data = json.load(open(traj_path))
    conv = data.get("conversation", [])
    extra = data.get("sample_extra_info", {}) or {}
    challenge = extra.get("challenge_name", "")
    if not challenge:
        # guess from filename
        base = os.path.basename(traj_path)
        # strip timestamp suffix
        m = re.match(r"(.+?)_\d{8}_\d{6}_trajectory\.json$", base)
        if m:
            challenge = m.group(1)
    if not challenge:
        return {}

    result: Dict[str, Dict[str, List[List[int]]]] = {challenge: defaultdict(list)}

    # Walk messages: when a user message contains a memory block, the most recent prior
    # step prompt tells us the action. Step prompt messages are role=user.
    # Simplest: scan all user messages, for each that has memory, parse + infer action.
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "") or ""
        m = MEM_BLOCK_RE.search(content)
        if not m:
            continue
        block = m.group(1)
        entries = parse_memory_block(block)
        # Match each (challenge, entry_type, insight) back to memory JSON index
        indices: List[int] = []
        for (cn, et, ins) in entries:
            key = (cn, et, ins[:200])
            i = insight_index.get(key)
            if i is None:
                # try without entry_type (some old entries may have type mismatch)
                # fallback: search any entry_type
                for (cn2, et2, prefix2), i2 in insight_index.items():
                    if cn2 == cn and prefix2 == ins[:200]:
                        i = i2
                        break
            if i is not None:
                indices.append(i)
            else:
                # mark unmatched so we don't silently lose alignment
                indices.append(-1)
        action = _extract_action_from_prompt(content)
        if action == "unknown":
            continue
        result[challenge][action].append(indices)

    return {k: dict(v) for k, v in result.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--trajectories", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    mem = json.load(open(args.memory))
    print(f"Loaded {len(mem)} memory entries from {args.memory}")
    insight_index = build_insight_index(mem)

    combined: Dict[str, Dict[str, List[List[int]]]] = {}
    total_files = 0
    unmatched_count = 0
    matched_count = 0
    for d in args.trajectories:
        if not os.path.isdir(d):
            print(f"  skip (not a dir): {d}")
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith("_trajectory.json"):
                continue
            path = os.path.join(d, fname)
            total_files += 1
            one = extract_from_trajectory(path, insight_index)
            for cn, actions in one.items():
                combined.setdefault(cn, {})
                for ac, seqs in actions.items():
                    combined[cn].setdefault(ac, []).extend(seqs)
                    for seq in seqs:
                        for i in seq:
                            if i == -1:
                                unmatched_count += 1
                            else:
                                matched_count += 1

    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    total_entries = matched_count + unmatched_count
    unmatched_pct = (unmatched_count / total_entries * 100) if total_entries else 0.0
    print(f"Scanned {total_files} trajectories.")
    print(f"Matched {matched_count}/{total_entries} entries "
          f"({unmatched_pct:.1f}% unmatched).")
    print(f"Replay lookup covers {len(combined)} challenges:")
    for cn in sorted(combined.keys()):
        acts = combined[cn]
        counts = {a: len(v) for a, v in acts.items()}
        print(f"  {cn}: {counts}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
