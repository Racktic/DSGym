#!/usr/bin/env python3
"""
Generate DSPredict target-swap tasks from HARD split.

Reuses logic from generate_swap_tasks.py but with hard-specific paths.
Uses LiteLLM proxy (Claude/GPT/Gemini) for description polishing.

Usage:
  python scripts/generate_hard_swap_tasks.py --analyze          # Step 1: show candidates
  python scripts/generate_hard_swap_tasks.py --generate         # Step 2: generate all
  python scripts/generate_hard_swap_tasks.py --generate --no-llm  # without LLM polish
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (HARD-specific)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_JSON = REPO_ROOT / "data/task/dspredict/hard.json"
DATA_DIR = REPO_ROOT / "data/data/dspredict-hard"
SWAP_DATA_DIR = REPO_ROOT / "data/data/dspredict-hard-swap"
SWAP_GT_DIR = REPO_ROOT / "data/data/dspredict-hard-swap-ground-truth"
OUTPUT_JSON = REPO_ROOT / "data/task/dspredict/hard_swap.json"

# LLM config (LiteLLM proxy)
LITELLM_BASE_URL = "https://litellm.nbdevenv.xiaoaojianghu.fun"
LITELLM_API_KEY = "$LITELLM_API_KEY"
LLM_MODEL = "openai/claude-sonnet-4.6"

# Import shared functions from generate_swap_tasks
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from generate_swap_tasks import (
    is_id_column,
    analyze_dataset as _analyze_dataset_orig,
    select_metric,
    check_learnability as _check_learnability_orig,
)


# ---------------------------------------------------------------------------
# Override functions to use hard paths
# ---------------------------------------------------------------------------

def analyze_dataset(dataset_name: str) -> Dict[str, Any]:
    """Analyze a hard dataset and return candidate swap targets."""
    dataset_dir = DATA_DIR / dataset_name

    # Try multiple train file locations (hard datasets have varied structures)
    train_path = None
    test_path = None
    for tp in [dataset_dir / "train.csv", dataset_dir / "train" / "train.csv"]:
        if tp.exists():
            train_path = tp
            break
    for tp in [dataset_dir / "test.csv", dataset_dir / "test" / "test.csv"]:
        if tp.exists():
            test_path = tp
            break

    if not train_path or not test_path:
        return {"name": dataset_name, "skip": True, "reason": "no train.csv or test.csv"}

    # Check file size - skip very large files (> 500MB for hard)
    try:
        fsize = train_path.stat().st_size
        if fsize > 500_000_000:
            return {"name": dataset_name, "skip": True, "reason": f"too large ({fsize // 1_000_000}MB)"}
    except:
        pass

    try:
        header = pd.read_csv(train_path, nrows=0)
        if len(header.columns) > 300:
            return {"name": dataset_name, "skip": True, "reason": f"too many columns ({len(header.columns)})"}
        train_df = pd.read_csv(train_path, nrows=10000)
        test_df = pd.read_csv(test_path, nrows=100)
    except Exception as e:
        return {"name": dataset_name, "skip": True, "reason": f"read error: {e}"}

    if len(train_df) < 200:
        return {"name": dataset_name, "skip": True, "reason": f"too few rows ({len(train_df)})"}

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    original_targets = train_cols - test_cols
    common_cols = train_cols & test_cols

    if not original_targets:
        return {"name": dataset_name, "skip": True, "reason": "no original target found"}

    candidates = []
    for col in sorted(common_cols):
        series = train_df[col]

        if is_id_column(series, col):
            continue
        null_pct = series.isnull().mean()
        if null_pct > 0.2:
            continue
        nunique = series.nunique()
        if nunique <= 1:
            continue
        if series.dtype == object:
            if nunique > len(train_df) * 0.5 or nunique > 15:
                continue

        if pd.api.types.is_numeric_dtype(series):
            task_type = "classification" if nunique <= 15 else "regression"
            n_classes = nunique if nunique <= 15 else None
        elif series.dtype == object and nunique <= 15:
            task_type = "classification"
            n_classes = nunique
        else:
            continue

        score = 0
        score += (1 - null_pct) * 30
        if task_type == "classification":
            vc = series.value_counts(normalize=True)
            entropy = -(vc * np.log2(vc + 1e-10)).sum()
            max_entropy = np.log2(nunique) if nunique > 1 else 1
            balance = entropy / max_entropy if max_entropy > 0 else 0
            score += balance * 30
        else:
            std = series.std()
            mean_val = series.mean()
            if mean_val != 0:
                cv = abs(std / mean_val)
                score += min(cv, 2) * 15
            score += 15

        if not col.startswith(("var_", "V", "f_", "feature_")):
            score += 20

        has_leakage = False
        if task_type == "regression":
            try:
                numeric_df = train_df.select_dtypes(include=["number"])
                if col in numeric_df.columns:
                    corrs = numeric_df.corr()[col].abs()
                    corrs = corrs.drop(col, errors="ignore")
                    if (corrs > 0.95).any():
                        has_leakage = True
            except:
                pass

        candidates.append({
            "column": col,
            "task_type": task_type,
            "n_classes": n_classes,
            "nunique": nunique,
            "null_pct": null_pct,
            "score": score,
            "has_leakage": has_leakage,
            "dtype": str(series.dtype),
        })

    candidates = [c for c in candidates if not c["has_leakage"]]
    candidates.sort(key=lambda x: -x["score"])

    final = []
    used_corr_groups = set()
    for cand in candidates:
        if len(final) >= 3:
            break
        col = cand["column"]
        if col in used_corr_groups:
            continue
        final.append(cand)
        if cand["task_type"] == "regression":
            try:
                numeric_df = train_df.select_dtypes(include=["number"])
                if col in numeric_df.columns:
                    corrs = numeric_df.corr()[col].abs()
                    for c2, v in corrs.items():
                        if v > 0.99 and c2 != col:
                            used_corr_groups.add(c2)
            except:
                pass

    return {
        "name": dataset_name,
        "skip": False,
        "n_train_rows": len(train_df),
        "n_cols": len(train_df.columns),
        "original_targets": list(original_targets),
        "candidates": final,
        "all_candidates_count": len(candidates),
    }


def transform_dataset(dataset_name: str, target_col: str, task_type: str) -> Optional[Dict]:
    """Create new train/test split with swapped target."""
    dataset_dir = DATA_DIR / dataset_name

    train_path = None
    for tp in [dataset_dir / "train.csv", dataset_dir / "train" / "train.csv"]:
        if tp.exists():
            train_path = tp
            break
    test_path = None
    for tp in [dataset_dir / "test.csv", dataset_dir / "test" / "test.csv"]:
        if tp.exists():
            test_path = tp
            break

    train_df = pd.read_csv(train_path, nrows=100000)
    test_df = pd.read_csv(test_path, nrows=100)

    original_targets = set(train_df.columns) - set(test_df.columns)

    id_col = None
    for col in train_df.columns:
        if is_id_column(train_df[col], col):
            id_col = col
            break

    df = train_df.copy()
    for ot in original_targets:
        if ot in df.columns:
            df = df.drop(columns=[ot])

    df = df.dropna(subset=[target_col])
    if len(df) < 250:
        return None

    if task_type == "classification":
        try:
            new_train, new_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
        except ValueError:
            new_train, new_test = train_test_split(df, test_size=0.2, random_state=42)
    else:
        new_train, new_test = train_test_split(df, test_size=0.2, random_state=42)

    if id_col is None:
        id_col = "id"
        new_train = new_train.reset_index(drop=True)
        new_train.insert(0, "id", range(len(new_train)))
        new_test = new_test.reset_index(drop=True)
        new_test.insert(0, "id", range(len(new_train), len(new_train) + len(new_test)))

    gt = new_test[[id_col, target_col]].copy()
    test_out = new_test.drop(columns=[target_col])
    sample_sub = new_test[[id_col]].copy()
    if task_type == "regression":
        sample_sub[target_col] = new_train[target_col].mean()
    else:
        sample_sub[target_col] = new_train[target_col].mode().iloc[0]

    swap_name = f"{dataset_name}__predict_{target_col}"
    out_dir = SWAP_DATA_DIR / swap_name
    gt_dir = SWAP_GT_DIR / swap_name
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    new_train.to_csv(out_dir / "train.csv", index=False)
    test_out.to_csv(out_dir / "test.csv", index=False)
    sample_sub.to_csv(out_dir / "sample_submission.csv", index=False)
    gt.to_csv(gt_dir / "ground_truth.csv", index=False)

    return {
        "swap_name": swap_name,
        "id_col": id_col,
        "n_train": len(new_train),
        "n_test": len(new_test),
        "n_features": len(new_train.columns) - 1,
    }


def check_learnability(swap_name: str, target_col: str, task_type: str) -> bool:
    """Check if a simple model can beat dummy baseline."""
    out_dir = SWAP_DATA_DIR / swap_name
    df = pd.read_csv(out_dir / "train.csv", nrows=10000)

    y = df[target_col]
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=["number"]).fillna(0)

    if len(X.columns) == 0:
        return False

    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_tr, y_tr)
        return accuracy_score(y_val, tree.predict(X_val)) > accuracy_score(y_val, dummy.predict(X_val)) + 0.02
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
        tree = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_tr, y_tr)
        dummy_rmse = np.sqrt(mean_squared_error(y_val, dummy.predict(X_val)))
        tree_rmse = np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))
        return tree_rmse < dummy_rmse * 0.9


def generate_description_llm(
    dataset_name: str, target_col: str, task_type: str,
    metric_name: str, original_desc: str, col_stats: str,
) -> Optional[Dict]:
    """Use LLM via LiteLLM proxy to generate polished task description."""
    try:
        import litellm
    except ImportError:
        print("  litellm not installed, skipping LLM")
        return None

    prompt = f"""You are helping create a Kaggle-style data science competition task.

Original dataset: {dataset_name}
New prediction target: {target_col}
Task type: {task_type}
Evaluation metric: {metric_name}

Original competition description (for context):
{original_desc[:2000]}

Column statistics for the new target:
{col_stats}

Generate a concise competition description for this new task. Include:
1. A brief title
2. Competition description (2-3 paragraphs)
3. Evaluation metric description
4. Submission file format

Return ONLY a JSON object with keys: "title", "competition_description", "evaluation_metric", "submission_format"
Do not include any other text."""

    try:
        response = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            api_base=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            max_tokens=2000,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        # Try to parse JSON
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"  LLM error: {e}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate hard-swap tasks")
    parser.add_argument("--analyze", action="store_true", help="Analyze candidates")
    parser.add_argument("--generate", action="store_true", help="Generate tasks")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM polish")
    parser.add_argument("--dataset", type=str, help="Process single dataset")
    args = parser.parse_args()

    with open(SOURCE_JSON) as f:
        source_tasks = json.load(f)

    dataset_names = [t["challenge_name"] for t in source_tasks]
    if args.dataset:
        dataset_names = [args.dataset]

    # Load original descriptions for LLM context
    orig_descs = {t["challenge_name"]: t.get("competition_description", "") for t in source_tasks}

    if args.analyze:
        print(f"Analyzing {len(dataset_names)} hard datasets...\n")
        total_candidates = 0
        skipped = 0
        for name in sorted(dataset_names):
            result = analyze_dataset(name)
            if result.get("skip"):
                print(f"  SKIP {name}: {result.get('reason')}")
                skipped += 1
            else:
                n_cand = len(result.get("candidates", []))
                total_candidates += n_cand
                print(f"  {name}: {n_cand} candidates (from {result.get('all_candidates_count', 0)} total)")
                for c in result.get("candidates", []):
                    print(f"    - {c['column']} ({c['task_type']}, nunique={c['nunique']}, score={c['score']:.1f})")
        print(f"\nTotal: {total_candidates} candidates from {len(dataset_names) - skipped} datasets ({skipped} skipped)")

    elif args.generate:
        print(f"Generating hard-swap tasks from {len(dataset_names)} datasets...\n")
        SWAP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        SWAP_GT_DIR.mkdir(parents=True, exist_ok=True)

        all_tasks = []
        for name in sorted(dataset_names):
            result = analyze_dataset(name)
            if result.get("skip"):
                continue

            for cand in result.get("candidates", []):
                col = cand["column"]
                task_type = cand["task_type"]
                n_classes = cand.get("n_classes")

                print(f"  {name} -> predict {col} ({task_type})...", end=" ", flush=True)

                # Transform
                stats = transform_dataset(name, col, task_type)
                if stats is None:
                    print("SKIP (too few rows after filtering)")
                    continue

                # Learnability check
                if not check_learnability(stats["swap_name"], col, task_type):
                    print("SKIP (not learnable)")
                    # Clean up
                    import shutil
                    shutil.rmtree(SWAP_DATA_DIR / stats["swap_name"], ignore_errors=True)
                    shutil.rmtree(SWAP_GT_DIR / stats["swap_name"], ignore_errors=True)
                    continue

                # Select metric
                train_df = pd.read_csv(SWAP_DATA_DIR / stats["swap_name"] / "train.csv", nrows=1000)
                metric_name, metric_text = select_metric(task_type, n_classes, train_df[col])

                # LLM description
                desc_data = None
                if not args.no_llm:
                    col_stats = str(train_df[col].describe())
                    desc_data = generate_description_llm(
                        name, col, task_type, metric_name,
                        orig_descs.get(name, ""), col_stats,
                    )

                # Build task JSON
                if desc_data:
                    comp_desc = desc_data.get("competition_description", "")
                    eval_desc = desc_data.get("evaluation_metric", metric_text)
                    title = desc_data.get("title", f"Predict {col} from {name}")
                else:
                    comp_desc = f"Predict the '{col}' column using the features from the {name} dataset."
                    eval_desc = metric_text
                    title = f"Predict {col} from {name}"

                task = {
                    "challenge_name": stats["swap_name"],
                    "docker_challenge_path": f"/data/dspredict-hard-swap/{stats['swap_name']}",
                    "competition_description": f"{title}\n\n{comp_desc}\n\n{eval_desc}",
                    "dataset_description": f"Files: train.csv, test.csv, sample_submission.csv\n"
                                           f"Target: {col}\nID column: {stats['id_col']}\n"
                                           f"Train rows: {stats['n_train']}, Test rows: {stats['n_test']}, Features: {stats['n_features']}",
                    "evaluation_metric": eval_desc,
                    "submission_format": f"{stats['id_col']},{col}\n...",
                    "metadata": {
                        "domain": "machine_learning",
                        "keywords": [task_type, metric_name, "tabular", "feature_engineering"],
                        "source": "target_swap_hard",
                        "original_dataset": name,
                        "swap_target": col,
                        "task_type": task_type,
                        "metric": metric_name,
                    },
                }
                all_tasks.append(task)
                print(f"OK ({metric_name}, {stats['n_train']} train)")

        # Save
        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_tasks, f, indent=2, ensure_ascii=False)
        print(f"\nGenerated {len(all_tasks)} tasks -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
