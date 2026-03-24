#!/usr/bin/env python3
"""
Generate DSPredict target-swap tasks.

Reuses existing tabular CSV data but swaps prediction targets to create new tasks.
Uses LLM (Together AI / Qwen 235B) to polish task descriptions.

Usage:
  python scripts/generate_swap_tasks.py --analyze          # Step 1: show candidates
  python scripts/generate_swap_tasks.py --generate         # Step 2: generate all
  python scripts/generate_swap_tasks.py --generate --no-llm  # without LLM polish
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
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
EASY_JSON = REPO_ROOT / "data/task/dspredict/easy.json"
DATA_DIR = REPO_ROOT / "data/data/dspredict-easy"
SWAP_DATA_DIR = REPO_ROOT / "data/data/dspredict-swap"
SWAP_GT_DIR = REPO_ROOT / "data/data/dspredict-swap-ground-truth"
OUTPUT_JSON = REPO_ROOT / "data/task/dspredict/swap.json"

# LLM config
TOGETHER_API_KEY = os.environ.get(
    "TOGETHER_API_KEY",
    "tgp_v1_CZUEMzg7WnjCfNkZ6ActxCwcGgDWXbbuaQbdqpCzZfk"
)
LLM_MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

# ---------------------------------------------------------------------------
# ID column detection
# ---------------------------------------------------------------------------
ID_PATTERNS = {"id", "index", "row_id", "passengerid", "customerid",
               "transactionid", "id_code", "sk_id_curr", "building_id",
               "image_id", "seg_id", "card_id"}


def is_id_column(series: pd.Series, col_name: str) -> bool:
    name_lower = col_name.strip().lower().replace(" ", "_")
    if name_lower in ID_PATTERNS:
        return True
    if name_lower.endswith("_id") or name_lower.endswith("id"):
        if name_lower not in {"valid", "morbid", "android", "humanoid"}:
            return True
    # Monotonically increasing integers
    if pd.api.types.is_integer_dtype(series):
        if series.is_monotonic_increasing and series.nunique() == len(series):
            return True
    return False


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def analyze_dataset(dataset_name: str) -> Dict[str, Any]:
    """Analyze a dataset and return candidate swap targets."""
    dataset_dir = DATA_DIR / dataset_name
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        return {"name": dataset_name, "skip": True, "reason": "no train.csv or test.csv"}

    # Check file size - skip very large files (> 200MB)
    try:
        fsize = train_path.stat().st_size
        if fsize > 200_000_000:
            return {"name": dataset_name, "skip": True, "reason": f"too large ({fsize // 1_000_000}MB)"}
    except:
        pass

    try:
        # Quick column count check
        header = pd.read_csv(train_path, nrows=0)
        if len(header.columns) > 200:
            return {"name": dataset_name, "skip": True, "reason": f"too many columns ({len(header.columns)})"}
        train_df = pd.read_csv(train_path, nrows=10000)
        test_df = pd.read_csv(test_path, nrows=100)
    except Exception as e:
        return {"name": dataset_name, "skip": True, "reason": f"read error: {e}"}

    if len(train_df) < 200:
        return {"name": dataset_name, "skip": True, "reason": f"too few rows ({len(train_df)})"}

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    # Original target: in train but not in test
    original_targets = train_cols - test_cols
    # Common columns: in both
    common_cols = train_cols & test_cols

    if not original_targets:
        return {"name": dataset_name, "skip": True, "reason": "no original target found"}

    candidates = []
    for col in sorted(common_cols):
        series = train_df[col]

        # Skip ID columns
        if is_id_column(series, col):
            continue

        # Skip high null
        null_pct = series.isnull().mean()
        if null_pct > 0.2:
            continue

        nunique = series.nunique()

        # Skip constant
        if nunique <= 1:
            continue

        # Skip high-cardinality strings
        if series.dtype == object:
            if nunique > len(train_df) * 0.5:
                continue
            if nunique > 15:
                continue  # Too many categories for classification

        # Determine task type
        if pd.api.types.is_numeric_dtype(series):
            if nunique <= 15:
                task_type = "classification"
                n_classes = nunique
            else:
                task_type = "regression"
                n_classes = None
        elif series.dtype == object:
            if nunique <= 15:
                task_type = "classification"
                n_classes = nunique
            else:
                continue
        else:
            continue

        # Quality score
        score = 0
        score += (1 - null_pct) * 30  # Less nulls = better
        if task_type == "classification":
            # Prefer balanced classes
            vc = series.value_counts(normalize=True)
            entropy = -(vc * np.log2(vc + 1e-10)).sum()
            max_entropy = np.log2(nunique) if nunique > 1 else 1
            balance = entropy / max_entropy if max_entropy > 0 else 0
            score += balance * 30
        else:
            # For regression, prefer non-skewed
            std = series.std()
            mean_val = series.mean()
            if mean_val != 0:
                cv = abs(std / mean_val)
                score += min(cv, 2) * 15  # Moderate variance
            score += 15

        # Prefer meaningful column names
        if not col.startswith(("var_", "V", "f_", "feature_")):
            score += 20

        # Check feature leakage: skip if any other column has corr > 0.95
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

    # Filter out leakage, sort by score, take top 3
    candidates = [c for c in candidates if not c["has_leakage"]]
    candidates.sort(key=lambda x: -x["score"])

    # Deduplicate: remove columns correlated > 0.99 with each other
    final = []
    used_corr_groups = set()
    for cand in candidates:
        if len(final) >= 3:
            break
        col = cand["column"]
        if col in used_corr_groups:
            continue
        final.append(cand)
        # Mark correlated columns
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


# ---------------------------------------------------------------------------
# Metric selection
# ---------------------------------------------------------------------------

def select_metric(task_type: str, n_classes: Optional[int],
                  series: pd.Series) -> Tuple[str, str]:
    """Return (metric_name, metric_description_text)."""
    if task_type == "classification":
        if n_classes == 2:
            return "auc", (
                "### Metric\n"
                "Submissions are evaluated on area under the ROC curve "
                "between the predicted probability and the observed target."
            )
        elif n_classes <= 5:
            return "log_loss", (
                "### Metric\n"
                "Submissions are evaluated using multi-class logarithmic loss. "
                "Each row should contain predicted probabilities for each class."
            )
        else:
            return "accuracy", (
                "### Metric\n"
                "Submissions are evaluated using classification accuracy — "
                "the percentage of correctly predicted labels."
            )
    else:  # regression
        non_null = series.dropna()
        if (non_null > 0).all():
            return "rmsle", (
                "### Metric\n"
                "Submissions are evaluated on Root Mean Squared Logarithmic Error (RMSLE) "
                "between the predicted and observed values."
            )
        else:
            return "rmse", (
                "### Metric\n"
                "Submissions are evaluated on Root Mean Squared Error (RMSE) "
                "between the predicted and observed values."
            )


# ---------------------------------------------------------------------------
# Data transformation
# ---------------------------------------------------------------------------

def transform_dataset(dataset_name: str, target_col: str,
                      task_type: str) -> Optional[Dict]:
    """Create new train/test split with swapped target. Returns stats or None."""
    dataset_dir = DATA_DIR / dataset_name
    # Read full data but cap at 100k rows for transform
    train_df = pd.read_csv(dataset_dir / "train.csv", nrows=100000)
    test_df = pd.read_csv(dataset_dir / "test.csv", nrows=100)

    # Identify original target and ID columns
    original_targets = set(train_df.columns) - set(test_df.columns)

    # Find ID column
    id_col = None
    for col in train_df.columns:
        if is_id_column(train_df[col], col):
            id_col = col
            break

    # Work with train.csv only (test.csv doesn't have ground truth for new target)
    df = train_df.copy()

    # Drop original target columns
    for ot in original_targets:
        if ot in df.columns:
            df = df.drop(columns=[ot])

    # Drop rows where new target is null
    df = df.dropna(subset=[target_col])

    if len(df) < 250:
        return None

    # Split 80/20
    if task_type == "classification":
        try:
            new_train, new_test = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df[target_col]
            )
        except ValueError:
            new_train, new_test = train_test_split(
                df, test_size=0.2, random_state=42
            )
    else:
        new_train, new_test = train_test_split(
            df, test_size=0.2, random_state=42
        )

    # Ensure ID column exists
    if id_col is None:
        id_col = "id"
        new_train = new_train.reset_index(drop=True)
        new_train.insert(0, "id", range(len(new_train)))
        new_test = new_test.reset_index(drop=True)
        new_test.insert(0, "id", range(len(new_train), len(new_train) + len(new_test)))

    # Ground truth
    gt = new_test[[id_col, target_col]].copy()

    # Test CSV: remove target column
    test_out = new_test.drop(columns=[target_col])

    # Sample submission
    sample_sub = new_test[[id_col]].copy()
    if task_type == "regression":
        sample_sub[target_col] = new_train[target_col].mean()
    else:
        sample_sub[target_col] = new_train[target_col].mode().iloc[0]

    # Save
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
        "n_features": len(new_train.columns) - 1,  # excluding target
    }


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def check_learnability(swap_name: str, target_col: str,
                       task_type: str) -> bool:
    """Check if a simple model can beat dummy baseline."""
    out_dir = SWAP_DATA_DIR / swap_name
    df = pd.read_csv(out_dir / "train.csv", nrows=10000)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep only numeric columns for simple check
    X = X.select_dtypes(include=["number"]).fillna(0)

    if len(X.columns) == 0:
        return False

    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_tr, y_tr)
        dummy_score = accuracy_score(y_val, dummy.predict(X_val))
        tree_score = accuracy_score(y_val, tree.predict(X_val))
        return tree_score > dummy_score + 0.02
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        dummy = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
        tree = DecisionTreeRegressor(max_depth=5, random_state=42).fit(X_tr, y_tr)
        dummy_rmse = np.sqrt(mean_squared_error(y_val, dummy.predict(X_val)))
        tree_rmse = np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))
        return tree_rmse < dummy_rmse * 0.9


# ---------------------------------------------------------------------------
# LLM description generation
# ---------------------------------------------------------------------------

def generate_description_llm(
    dataset_name: str, target_col: str, task_type: str,
    metric_name: str, original_desc: str, col_stats: str,
) -> Optional[Dict]:
    """Use LLM to generate polished task description."""
    try:
        import litellm
    except ImportError:
        print("  litellm not installed, skipping LLM")
        return None

    os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

    prompt = f"""You are creating a Kaggle-style machine learning competition description.

Original dataset: "{dataset_name}"
Original context (first 800 chars):
{original_desc[:800]}

New prediction target: "{target_col}" ({task_type})
Evaluation metric: {metric_name}

Column statistics for the new target:
{col_stats}

Generate a JSON object with these fields:
{{
  "competition_description": "<2-4 sentences describing what the competition is about and why predicting {target_col} is valuable. Write as if this is a real Kaggle competition.>",
  "dataset_description_extra": "<1-2 sentences about what {target_col} represents and its characteristics.>",
  "keywords": ["<kw1>", "<kw2>", "<kw3>", "<kw4>"]
}}

keywords should be 3-5 items chosen from: regression, binary_classification, multiclass_classification, tabular, feature_engineering, tree_models, rmse, rmsle, auc, log_loss, accuracy, mae, housing, healthcare, insurance, retail, energy, transportation, biology, material_science, sports, finance

Respond with valid JSON only. No markdown fences. No extra text. Think step by step inside <think></think> tags before responding with JSON."""

    try:
        resp = litellm.completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content
        if raw is None:
            raw = ""
        raw = raw.strip()
        # Remove think tags if present
        if "<think>" in raw:
            parts = raw.split("</think>")
            if len(parts) > 1:
                raw = parts[-1].strip()
        # Remove markdown fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        # Try to find JSON object in the text
        if not raw.startswith("{"):
            start = raw.find("{")
            if start >= 0:
                end = raw.rfind("}") + 1
                if end > start:
                    raw = raw[start:end]
        return json.loads(raw)
    except Exception as e:
        print(f"  LLM failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Task JSON builder
# ---------------------------------------------------------------------------

def build_task_entry(
    dataset_name: str, target_col: str, task_type: str,
    n_classes: Optional[int], transform_info: Dict,
    original_task: Optional[Dict], use_llm: bool = True,
) -> Dict:
    """Build a DSPredict-format task entry."""
    swap_name = transform_info["swap_name"]
    id_col = transform_info["id_col"]

    # Read target stats
    train_path = SWAP_DATA_DIR / swap_name / "train.csv"
    df = pd.read_csv(train_path, nrows=5000)
    series = df[target_col]

    metric_name, metric_text = select_metric(task_type, n_classes, series)

    # Build column stats string for LLM
    col_stats = f"dtype={series.dtype}, nunique={series.nunique()}, "
    if task_type == "classification":
        vc = series.value_counts().head(10).to_dict()
        col_stats += f"value_counts={vc}"
    else:
        col_stats += f"mean={series.mean():.4f}, std={series.std():.4f}, min={series.min():.4f}, max={series.max():.4f}"

    # Get original description for context
    orig_desc = ""
    if original_task:
        orig_desc = original_task.get("description", "")

    # LLM polish
    llm_result = None
    if use_llm:
        llm_result = generate_description_llm(
            dataset_name, target_col, task_type,
            metric_name, orig_desc, col_stats
        )

    # Build title
    pretty_target = target_col.replace("_", " ").title()
    pretty_dataset = dataset_name.replace("-", " ").replace("_", " ").title()

    # Competition description
    if llm_result and "competition_description" in llm_result:
        comp_desc = f"## Competition Description\n{llm_result['competition_description']}"
    else:
        comp_desc = (
            f"## Competition Description\n"
            f"In this competition, your task is to predict the `{target_col}` "
            f"for each observation in the test set. This is a {task_type} task "
            f"based on the {pretty_dataset} dataset."
        )

    # Dataset description
    features = [c for c in df.columns if c != target_col]
    ds_desc_base = (
        f"Data description:\n{pretty_dataset} - Predict {pretty_target}\n\n"
        f"Dataset Description\n"
        f"File descriptions\n"
        f"train.csv - the training set ({transform_info['n_train']} rows)\n"
        f"test.csv - the test set ({transform_info['n_test']} rows)\n"
        f"sample_submission.csv - a sample submission file in the correct format\n\n"
        f"Target variable: `{target_col}`"
    )
    if llm_result and "dataset_description_extra" in llm_result:
        ds_desc_base += f" - {llm_result['dataset_description_extra']}"
    ds_desc_base += f"\n\nFeatures ({len(features)}): {', '.join(features[:20])}"
    if len(features) > 20:
        ds_desc_base += f", ... ({len(features) - 20} more)"

    # Submission format
    if task_type == "classification" and n_classes == 2:
        sub_example = f"{id_col},{target_col}\n0,0.5\n1,0.8\n2,0.1"
        sub_note = f"For each {id_col} in the test set, predict the probability of {target_col}."
    elif task_type == "classification":
        unique_vals = series.unique()[:5]
        sub_example = f"{id_col},{target_col}\n0,{unique_vals[0]}\n1,{unique_vals[1] if len(unique_vals) > 1 else unique_vals[0]}"
        sub_note = f"For each {id_col} in the test set, predict the value of {target_col}."
    else:
        mean_val = f"{series.mean():.2f}"
        sub_example = f"{id_col},{target_col}\n0,{mean_val}\n1,{mean_val}"
        sub_note = f"For each {id_col} in the test set, predict the value of {target_col}."

    # Full description
    full_desc = (
        f"Challenge:\n# {pretty_dataset} - Predict {pretty_target}\n\n"
        f"{comp_desc}\n\n"
        f"## Evaluation\n### Goal\n{sub_note}\n\n"
        f"{metric_text}\n\n"
        f"### Submission File Format\n"
        f"The file should contain a header and have the following format:\n"
        f"```\n{sub_example}\netc.\n```\n\n"
        f"{ds_desc_base}"
    )

    # Keywords
    if llm_result and "keywords" in llm_result:
        keywords = llm_result["keywords"]
    else:
        keywords = [task_type, "tabular", metric_name]

    return {
        "challenge_name": swap_name,
        "description": full_desc,
        "docker_challenge_path": f"/data/{swap_name}",
        "competition_description": comp_desc,
        "evaluation_metric": metric_text,
        "dataset_description": ds_desc_base,
        "metadata": {
            "domain": "machine_learning",
            "keywords": keywords,
            "source": "target_swap",
            "original_dataset": dataset_name,
            "original_target": list(set(pd.read_csv(DATA_DIR / dataset_name / "train.csv", nrows=1).columns) - set(pd.read_csv(DATA_DIR / dataset_name / "test.csv", nrows=1).columns)),
            "swap_target": target_col,
            "task_type": task_type,
            "metric": metric_name,
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_original_tasks() -> Dict[str, Dict]:
    """Load original easy.json tasks as a lookup dict."""
    if EASY_JSON.exists():
        tasks = json.load(open(EASY_JSON))
        return {t["challenge_name"]: t for t in tasks}
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Analyze datasets and show candidates")
    parser.add_argument("--generate", action="store_true", help="Generate swap tasks")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM description polishing")
    parser.add_argument("--dataset", type=str, help="Process only this dataset")
    args = parser.parse_args()

    if not args.analyze and not args.generate:
        args.analyze = True

    # Find all tabular datasets
    datasets = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "train.csv").exists() and (d / "test.csv").exists():
            if args.dataset and d.name != args.dataset:
                continue
            datasets.append(d.name)

    print(f"Found {len(datasets)} tabular datasets in {DATA_DIR}")

    # Step 1: Analyze
    all_analyses = []
    for i, ds in enumerate(datasets):
        print(f"Analyzing [{i+1}/{len(datasets)}] {ds}...", end=" ", flush=True)
        result = analyze_dataset(ds)
        if result.get("skip"):
            print(f"SKIP: {result.get('reason', 'unknown')}")
        else:
            n = len(result.get("candidates", []))
            print(f"OK ({n} candidates)")
        all_analyses.append(result)

    if args.analyze:
        total_candidates = 0
        for a in all_analyses:
            if a.get("skip"):
                continue
            cands = a.get("candidates", [])
            if not cands:
                continue
            print(f"\n=== {a['name']} ({a['n_train_rows']} rows, {a['n_cols']} cols) ===")
            print(f"  Original target: {a['original_targets']}")
            print(f"  Candidates ({len(cands)}/{a['all_candidates_count']} after filtering):")
            for c in cands:
                print(f"    {c['column']:30s} {c['task_type']:15s} "
                      f"nunique={c['nunique']:6d}  null={c['null_pct']:.1%}  "
                      f"score={c['score']:.1f}")
                total_candidates += 1
        print(f"\n=== Total: {total_candidates} swap candidates ===")

    if not args.generate:
        return

    # Step 2: Generate
    print("\n" + "=" * 60)
    print("GENERATING SWAP TASKS")
    print("=" * 60)

    original_tasks = get_original_tasks()
    all_tasks = []
    skipped = 0

    for a in all_analyses:
        if a.get("skip") or not a.get("candidates"):
            continue

        ds_name = a["name"]
        orig_task = original_tasks.get(ds_name)

        for cand in a["candidates"]:
            col = cand["column"]
            task_type = cand["task_type"]
            n_classes = cand.get("n_classes")
            swap_name = f"{ds_name}__predict_{col}"

            print(f"\n[{len(all_tasks) + 1}] {swap_name} ({task_type})")

            # Transform data
            info = transform_dataset(ds_name, col, task_type)
            if info is None:
                print(f"  SKIP: transform failed (too few rows after dropna)")
                skipped += 1
                continue

            # Quality check
            if not check_learnability(swap_name, col, task_type):
                print(f"  SKIP: not learnable (tree can't beat dummy)")
                skipped += 1
                # Clean up
                import shutil
                shutil.rmtree(SWAP_DATA_DIR / swap_name, ignore_errors=True)
                shutil.rmtree(SWAP_GT_DIR / swap_name, ignore_errors=True)
                continue

            # Build task entry
            entry = build_task_entry(
                ds_name, col, task_type, n_classes, info,
                orig_task, use_llm=not args.no_llm
            )
            all_tasks.append(entry)
            print(f"  OK: train={info['n_train']}, test={info['n_test']}, "
                  f"features={info['n_features']}")

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_tasks, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(all_tasks)} swap tasks, skipped {skipped}")
    print(f"Saved to {OUTPUT_JSON}")
    print(f"Data at {SWAP_DATA_DIR}")
    print(f"Ground truth at {SWAP_GT_DIR}")


if __name__ == "__main__":
    main()
