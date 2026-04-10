#!/usr/bin/env python3
"""Analyze evaluation results from summary.json files."""
import json, sys, glob, os
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python analyze_results.py <results_dir>")
    sys.exit(1)

results_dir = sys.argv[1]
summary_files = glob.glob(os.path.join(results_dir, "*_summary.json"))
if not summary_files:
    print(f"No *_summary.json found in {results_dir}")
    sys.exit(1)

with open(summary_files[0]) as f:
    data = json.load(f)

pub, priv = [], []
no_sub, kaggle_err = 0, 0

for r in data:
    d = r.get("metrics", {}).get("kaggle_submission", {}).get("details", {})
    pp = d.get("public_percentile")
    prp = d.get("private_percentile")
    reason = d.get("reason", "")
    status = str(d.get("status", ""))

    if pp is not None:
        pub.append(pp)
    elif "submission file not found" in reason:
        no_sub += 1
    elif "ERROR" in status:
        kaggle_err += 1

    if prp is not None:
        priv.append(prp)

total = len(data)
print(f"Total: {total}")
print(f"Valid Kaggle scores: {len(pub)}/{total}")
print(f"No submission: {no_sub}, Kaggle rejected: {kaggle_err}")
print()
if pub:
    print(f"Public:  avg={np.mean(pub):.1f}%  median={np.median(pub):.1f}%  above50={sum(1 for p in pub if p>50)}/{len(pub)}")
if priv:
    print(f"Private: avg={np.mean(priv):.1f}%  median={np.median(priv):.1f}%  above50={sum(1 for p in priv if p>50)}/{len(priv)}")
