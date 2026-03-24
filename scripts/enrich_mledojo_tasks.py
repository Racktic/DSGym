#!/usr/bin/env python3
"""
Enrich mle_dojo.json task descriptions using actual downloaded data files.

- Adds file listing to dataset_description
- Adds CSV column info where possible
- Removes tasks with no downloaded data
- Optionally uses LLM to improve sparse descriptions
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = REPO_ROOT / "data/task/dspredict/mle_dojo.json"
OUTPUT_JSON = REPO_ROOT / "data/task/dspredict/mle_dojo.json"  # overwrite
DATA_DIR = REPO_ROOT / "data/data/dspredict-mledojo"

TOGETHER_API_KEY = "tgp_v1_CZUEMzg7WnjCfNkZ6ActxCwcGgDWXbbuaQbdqpCzZfk"
LLM_MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"


def get_file_info(data_path: Path) -> str:
    """Get a description of files in a data directory."""
    if not data_path.exists():
        return ""

    lines = []
    files = sorted(os.listdir(data_path))

    # Separate files and dirs
    file_list = []
    dir_list = []
    for f in files:
        fp = data_path / f
        if fp.is_dir():
            n_items = len(list(fp.iterdir())) if fp.exists() else 0
            dir_list.append(f"  {f}/ ({n_items} items)")
        else:
            size = fp.stat().st_size
            if size > 1_000_000_000:
                size_str = f"{size / 1_000_000_000:.1f} GB"
            elif size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1000:
                size_str = f"{size / 1000:.1f} KB"
            else:
                size_str = f"{size} B"
            file_list.append(f"  {f} ({size_str})")

    lines.append("\nAvailable files:")
    for f in file_list[:30]:
        lines.append(f)
    if len(file_list) > 30:
        lines.append(f"  ... and {len(file_list) - 30} more files")
    for d in dir_list[:10]:
        lines.append(d)
    if len(dir_list) > 10:
        lines.append(f"  ... and {len(dir_list) - 10} more directories")

    return "\n".join(lines)


def get_csv_columns(data_path: Path) -> str:
    """Read CSV headers and first few rows to describe data schema."""
    info_parts = []

    for csv_name in ["train.csv", "training.csv", "train_transaction.csv",
                     "application_train.csv", "X_train.csv"]:
        csv_path = data_path / csv_name
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, nrows=3)
                cols = list(df.columns)
                n_rows_approx = sum(1 for _ in open(csv_path)) - 1
                info_parts.append(f"\n{csv_name}: ~{n_rows_approx:,} rows, {len(cols)} columns")
                if len(cols) <= 30:
                    info_parts.append(f"Columns: {', '.join(cols)}")
                else:
                    info_parts.append(f"Columns (first 20): {', '.join(cols[:20])}, ... ({len(cols)-20} more)")
            except Exception:
                pass

    for csv_name in ["test.csv", "test_transaction.csv", "application_test.csv",
                     "X_test.csv"]:
        csv_path = data_path / csv_name
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, nrows=1)
                n_rows_approx = sum(1 for _ in open(csv_path)) - 1
                info_parts.append(f"\n{csv_name}: ~{n_rows_approx:,} rows, {len(df.columns)} columns")
            except Exception:
                pass

    for csv_name in ["sample_submission.csv", "example_entry.csv",
                     "sampleSubmission.csv"]:
        csv_path = data_path / csv_name
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, nrows=3)
                cols = list(df.columns)
                info_parts.append(f"\n{csv_name} columns: {', '.join(cols)}")
                # Show sample
                info_parts.append(f"Sample:\n{df.head(2).to_csv(index=False).strip()}")
            except Exception:
                pass

    return "\n".join(info_parts)


def enrich_with_llm(task: dict, file_info: str, csv_info: str) -> Optional[dict]:
    """Use LLM to improve sparse descriptions."""
    try:
        import litellm
    except ImportError:
        return None

    os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

    current_desc = task.get("dataset_description", "")
    current_comp = task.get("competition_description", "")
    name = task["challenge_name"]

    # Only call LLM if description is sparse
    if len(current_desc) > 1000 and len(current_comp) > 500:
        return None  # Already good enough

    prompt = f"""Competition: "{name}"

Current description:
{current_comp[:800]}

Current dataset description:
{current_desc[:500]}

Actual files in data directory:
{file_info[:500]}

CSV schema info:
{csv_info[:800]}

Improve the descriptions. Output a JSON object:
{{
  "dataset_description_extra": "<2-4 sentences describing the data files, their structure, and how they relate to the task>",
  "competition_description_extra": "<1-2 sentences to add context about what makes this task interesting, only if the current description is very short>"
}}

Only add information that is factually supported by the file/schema info above. If the current descriptions are adequate, return empty strings.
Respond with valid JSON only. Think step by step inside <think></think> tags first."""

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        raw = resp.choices[0].message.content or ""
        raw = raw.strip()
        if "<think>" in raw:
            parts = raw.split("</think>")
            if len(parts) > 1:
                raw = parts[-1].strip()
        if not raw.startswith("{"):
            start = raw.find("{")
            if start >= 0:
                end = raw.rfind("}") + 1
                if end > start:
                    raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        print(f"  LLM failed for {name}: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = json.load(open(INPUT_JSON))
    print(f"Input: {len(tasks)} tasks")

    enriched_tasks = []
    skipped = 0
    llm_enriched = 0

    for i, task in enumerate(tasks):
        name = task["challenge_name"]
        data_path = DATA_DIR / name

        if not data_path.exists() or not os.listdir(data_path):
            print(f"[{i+1}] SKIP {name} (no data)")
            skipped += 1
            continue

        print(f"[{i+1}/{len(tasks)}] {name}", end="", flush=True)

        # Get file info
        file_info = get_file_info(data_path)
        csv_info = get_csv_columns(data_path)

        # Enrich dataset_description with actual file info
        current_ds_desc = task.get("dataset_description", "")
        if file_info and file_info not in current_ds_desc:
            task["dataset_description"] = current_ds_desc + "\n" + file_info
        if csv_info and csv_info not in current_ds_desc:
            task["dataset_description"] += "\n" + csv_info

        # LLM enrichment for sparse descriptions
        if not args.no_llm:
            llm_result = enrich_with_llm(task, file_info, csv_info)
            if llm_result:
                extra_ds = llm_result.get("dataset_description_extra", "")
                extra_comp = llm_result.get("competition_description_extra", "")
                if extra_ds:
                    task["dataset_description"] += "\n\n" + extra_ds
                if extra_comp and len(task.get("competition_description", "")) < 500:
                    task["competition_description"] += "\n" + extra_comp
                llm_enriched += 1
                print(f" +LLM", end="")

        # Rebuild full description to include enriched parts
        task["description"] = (
            f"Challenge:\n# {name.replace('-', ' ').replace('_', ' ').title()}\n\n"
            f"{task['competition_description']}\n\n"
            f"## Evaluation\n{task['evaluation_metric']}\n\n"
            f"{task['dataset_description']}"
        )

        enriched_tasks.append(task)
        print(f" OK")

    print(f"\n=== Done ===")
    print(f"Enriched: {len(enriched_tasks)}, Skipped (no data): {skipped}, LLM enriched: {llm_enriched}")

    if not args.dry_run:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(enriched_tasks, f, indent=2, ensure_ascii=False)
        print(f"Saved to {OUTPUT_JSON}")
    else:
        print("Dry run, not saving.")


if __name__ == "__main__":
    main()
