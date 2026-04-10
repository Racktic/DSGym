"""
Convert AIDE V5 trajectory JSONs to LLaMA-Factory SFT format.

For each successful trajectory (has submission):
1. Extract conversation (system/user/assistant messages)
2. Merge consecutive user messages into one
3. Remove memory-related content from user messages (task_memory, cross-task memory, reference approach)
4. Attach per-turn meta info (turn, phase, score, best_score, baseline_score) from task_memory

Output format (LLaMA-Factory sharegpt):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
  ],
  "meta": {
    "challenge_name": "...",
    "split": "swap",
    "run": "run1",
    "success": true,
    "final_best_score": 0.85,
    "baseline_score": 0.92,
    "num_turns": 20,
    "turns": [
      {"turn": 1, "phase": "draft", "score": null, "best_score": null, "buggy": false},
      ...
    ]
  }
}
"""

import json
import os
import re
import glob
import argparse


def remove_memory_sections(text):
    """Remove task_memory, cross-task memory, and reference approach from user messages."""
    # Remove task memory block
    text = re.sub(
        r"=== TASK MEMORY \(Previous Attempts\) ===.*?=== END TASK MEMORY ===\n*",
        "", text, flags=re.DOTALL
    )
    # Remove cross-task memory block
    text = re.sub(
        r"=== CROSS-TASK EXPERIENCE MEMORY ===.*?=== END CROSS-TASK MEMORY ===\n*",
        "", text, flags=re.DOTALL
    )
    # Remove reference approach block
    text = re.sub(
        r"--- Reference approach \(.*?\) ---.*?--- End reference ---\n*",
        "", text, flags=re.DOTALL
    )
    # Remove "No previous attempts yet." placeholder
    text = text.replace("No previous attempts yet.\n\n", "")
    text = text.replace("No previous attempts yet.", "")
    # Remove model frequency stats line
    text = re.sub(
        r"Model usage frequency across tasks:.*?Consider exploring under-represented approaches for diversity\.\n*",
        "", text, flags=re.DOTALL
    )
    # Clean up excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def merge_consecutive_users(messages):
    """Merge consecutive user messages into one."""
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == "user" and msg["role"] == "user":
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + msg["content"]
        else:
            merged.append({"role": msg["role"], "content": msg["content"]})
    return merged


def build_turn_meta(task_memory, turns_data):
    """Build per-turn meta info from task_memory and turn records."""
    turn_meta = []
    for i, tm in enumerate(task_memory):
        meta = {
            "turn": tm.get("step", i + 1),
            "model": tm.get("model", "unknown"),
            "score": tm.get("score"),
            "best_score": tm.get("best_score"),
        }
        # Try to get phase from turns_data
        if i < len(turns_data):
            meta["phase"] = turns_data[i].get("phase", "unknown")
            meta["buggy"] = "Traceback" in (turns_data[i].get("execution_output", "") or "")
        turn_meta.append(meta)
    return turn_meta


def convert_trajectory(traj_path, split, run=""):
    """Convert a single trajectory JSON to SFT format."""
    with open(traj_path) as f:
        traj = json.load(f)

    # Skip unsuccessful trajectories
    if not traj.get("success", False):
        return None

    conversation = traj.get("conversation", [])
    if not conversation:
        return None

    task_memory = traj.get("task_memory", [])
    turns_data = traj.get("turns", [])

    # Clean user messages: remove memory sections
    cleaned_conv = []
    for msg in conversation:
        if msg["role"] == "user":
            cleaned_content = remove_memory_sections(msg["content"])
            if cleaned_content:
                cleaned_conv.append({"role": "user", "content": cleaned_content})
        else:
            cleaned_conv.append({"role": msg["role"], "content": msg["content"]})

    # Merge consecutive user messages
    merged_conv = merge_consecutive_users(cleaned_conv)

    # Verify it starts with system
    if not merged_conv or merged_conv[0]["role"] != "system":
        return None

    # Build meta
    meta = {
        "challenge_name": traj.get("challenge_name", ""),
        "split": split,
        "run": run,
        "success": traj.get("success", False),
        "final_best_score": traj.get("final_best_score"),
        "baseline_score": traj.get("baseline_score"),
        "num_turns": traj.get("num_turns", len(task_memory)),
        "best_node_strategy": traj.get("best_node_strategy", ""),
        "turns": build_turn_meta(task_memory, turns_data),
    }

    return {
        "messages": merged_conv,
        "meta": meta,
    }


def process_directory(traj_dir, split, run=""):
    """Process all trajectory files in a directory."""
    traj_files = glob.glob(os.path.join(traj_dir, "*_trajectory.json"))
    results = []
    skipped = 0
    for tf in sorted(traj_files):
        result = convert_trajectory(tf, split, run)
        if result:
            results.append(result)
        else:
            skipped += 1
    return results, skipped


def main():
    parser = argparse.ArgumentParser(description="Convert trajectories to SFT format")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                       help="Input directories containing trajectory JSONs")
    parser.add_argument("--splits", nargs="+", required=True,
                       help="Split names corresponding to each input dir")
    parser.add_argument("--runs", nargs="+", default=None,
                       help="Run names corresponding to each input dir")
    parser.add_argument("--output", required=True,
                       help="Output JSON file path")
    args = parser.parse_args()

    if args.runs is None:
        args.runs = [""] * len(args.input_dirs)

    assert len(args.input_dirs) == len(args.splits) == len(args.runs)

    all_results = []
    for input_dir, split, run in zip(args.input_dirs, args.splits, args.runs):
        results, skipped = process_directory(input_dir, split, run)
        print(f"{input_dir}: {len(results)} converted, {skipped} skipped")
        all_results.extend(results)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_results)} samples saved to {args.output}")


if __name__ == "__main__":
    main()
