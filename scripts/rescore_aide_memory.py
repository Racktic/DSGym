#!/usr/bin/env python3
"""
Re-score AIDE+memory experiment by submitting existing submission.csv files to Kaggle.
"""

import os
import sys
import json
import glob
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dsgym.eval.metrics.dspredict.dspredict_metric import KaggleSubmissionMetric


def find_submission_files():
    """Find all AIDE+memory submission files (timestamped 20260321_02*)."""
    pattern = "submissions/container_*/*_20260321_02*_submission.csv"
    files = glob.glob(pattern)

    result = {}
    for f in files:
        # Extract challenge name from filename
        # Format: <challenge_name>_<container_id>_<timestamp>_submission.csv
        basename = os.path.basename(f)
        # Remove _submission.csv suffix
        parts = basename.replace("_submission.csv", "")
        # Split from right: timestamp, container_id, then rest is challenge_name
        tokens = parts.rsplit("_", 2)  # [challenge_name, container_id, timestamp]
        if len(tokens) >= 3:
            challenge_name = tokens[0]
            # Handle challenge names with underscores (e.g., house-prices-advanced-regression-techniques)
            # The container_id is a single digit, timestamp is YYYYMMDD_HHMMSS
            # Actually the format is: name_containerid_YYYYMMDD_HHMMSS
            # So we need to split differently
            pass

        # Better approach: extract challenge name from the path
        # The naming convention is: {challenge_name}_{container_id}_{timestamp}_submission.csv
        # where timestamp is YYYYMMDD_HHMMSS (contains underscore)
        # So split from right: _submission.csv, then HHMMSS, then YYYYMMDD, then container_id
        name_no_suffix = basename.replace("_submission.csv", "")
        # Split: last part is HHMMSS, second to last is YYYYMMDD, third is container_id
        parts = name_no_suffix.rsplit("_", 3)
        if len(parts) >= 4:
            challenge_name = parts[0]
        else:
            # Fallback
            parts = name_no_suffix.rsplit("_", 2)
            challenge_name = parts[0]

        result[challenge_name] = f

    return result


def main():
    # Find submission files
    submissions = find_submission_files()
    print(f"Found {len(submissions)} submission files:")
    for name, path in sorted(submissions.items()):
        print(f"  {name}: {path}")

    # Initialize metric
    metric = KaggleSubmissionMetric(timeout_minutes=10, online=False)

    # Score each submission
    results = []
    for i, (challenge_name, submission_path) in enumerate(sorted(submissions.items())):
        print(f"\n[{i+1}/{len(submissions)}] Scoring {challenge_name}...")
        print(f"  File: {submission_path}")

        extra_info = {"challenge_name": challenge_name}

        try:
            result = metric.evaluate(
                prediction=submission_path,
                extra_info=extra_info,
            )

            details = result.details or {}
            pub_score = details.get("public_score")
            pub_pct = details.get("public_percentile")
            pub_medal = details.get("public_medal")
            above_median = details.get("public_above_median")

            print(f"  Score: {pub_score}, Percentile: {pub_pct}, Medal: {pub_medal}, Above Median: {above_median}")

            results.append({
                "challenge_name": challenge_name,
                "submission_path": submission_path,
                "public_score": pub_score,
                "public_percentile": pub_pct,
                "public_medal": pub_medal,
                "public_above_median": above_median,
                "details": details,
                "error": result.error,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "challenge_name": challenge_name,
                "submission_path": submission_path,
                "error": str(e),
            })

    # Save results
    output_dir = "evaluation_results/aide_memory_qwen3_235b_easy_v1"
    output_path = os.path.join(output_dir, "rescore_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    scored = [r for r in results if r.get("public_percentile") is not None]
    if scored:
        percentiles = [r["public_percentile"] for r in scored]
        above_median_count = sum(1 for r in scored if r.get("public_above_median"))
        print(f"\n=== SUMMARY ===")
        print(f"Scored: {len(scored)}/{len(results)}")
        print(f"Mean Percentile: {sum(percentiles)/len(percentiles):.2f}")
        print(f"Median Percentile: {sorted(percentiles)[len(percentiles)//2]:.2f}")
        print(f"Above Median: {above_median_count}/{len(scored)} ({100*above_median_count/len(scored):.1f}%)")


if __name__ == "__main__":
    main()
