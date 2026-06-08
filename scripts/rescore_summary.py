#!/usr/bin/env python3
"""Rescore a finished eval directory: walk *_summary.json, re-submit each saved
submission.csv to Kaggle, and patch the kaggle_submission metric entry in place.

Usage:
    KAGGLE_API_TOKEN=KGAT_xxx python scripts/rescore_summary.py <results_dir>
"""
import glob
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsgym.eval.metrics.dspredict.dspredict_metric import KaggleSubmissionMetric


def challenge_from_sample_id(sid: str) -> str:
    if not sid.startswith("dspredict_"):
        return ""
    body = sid[len("dspredict_"):]
    half = body.rsplit("_", 1)[0]
    return half


def main(results_dir: str) -> int:
    summaries = glob.glob(os.path.join(results_dir, "*_summary.json"))
    if not summaries:
        print(f"No *_summary.json in {results_dir}")
        return 1
    summary_path = summaries[0]
    with open(summary_path) as f:
        data = json.load(f)

    metric = KaggleSubmissionMetric(timeout_minutes=10, online=False)

    for i, r in enumerate(data):
        sid = r.get("sample_id", "")
        challenge = challenge_from_sample_id(sid)
        sub_path = (r.get("prediction") or "").strip()
        existing = r.get("metrics", {}).get("kaggle_submission", {}) or {}
        existing_status = str(existing.get("details", {}).get("status", ""))
        existing_pub = existing.get("details", {}).get("public_percentile")
        if existing_pub is not None and "ERROR" not in existing_status:
            print(f"[{i+1}/{len(data)}] skip (already scored): {challenge}")
            continue
        if not sub_path or not os.path.exists(sub_path):
            print(f"[{i+1}/{len(data)}] skip (no submission file): {challenge}")
            continue
        print(f"[{i+1}/{len(data)}] rescoring {challenge} -> {os.path.basename(sub_path)}")
        result = metric.evaluate(prediction=sub_path, extra_info={"challenge_name": challenge})
        ks_entry = {
            "score": result.score,
            "details": result.details,
            "error": result.error,
            "evaluation_time": result.evaluation_time,
        }
        r.setdefault("metrics", {})["kaggle_submission"] = ks_entry

    with open(summary_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/rescore_summary.py <results_dir>")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
