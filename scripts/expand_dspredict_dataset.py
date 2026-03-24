#!/usr/bin/env python3
"""
Expand DSPredict easy split with more Kaggle competitions.

Steps per competition:
  1. Download via Kaggle REST API  (/api/v1/competitions/data/download-all/)
  2. Generate metadata via LLM     (reads data_description.txt + infers keywords)
  3. Build a JSON entry            (matching easy.json schema)
  4. Write expanded easy.json

Download status per competition:
  200 → downloadable immediately
  403 → need to accept rules at https://www.kaggle.com/competitions/{name}/rules
  404 → competition does not exist

Slug notes:
  - Season 3-5 Playground: "playground-series-s3e2", "playground-series-s4e1", etc.
  - Season 1/2 (TPS era):  "tabular-playground-series-jan-2021", ...-nov-2022
    These require rule acceptance on Kaggle before downloading.

Usage:
  python scripts/expand_dspredict_dataset.py --check-status
  python scripts/expand_dspredict_dataset.py --print-rules-urls
  python scripts/expand_dspredict_dataset.py --target s3_gaps --dry-run
  python scripts/expand_dspredict_dataset.py --target s3_gaps
  python scripts/expand_dspredict_dataset.py --target hard_tabular
  python scripts/expand_dspredict_dataset.py --target tps          # after accepting rules
  python scripts/expand_dspredict_dataset.py --target all
  python scripts/expand_dspredict_dataset.py --competitions "spaceship-titanic,ieee-fraud-detection"
"""

import argparse
import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
EASY_JSON = REPO_ROOT / "data/task/dspredict/easy.json"
DATA_DIR = REPO_ROOT / "data/data/dspredict-easy"
OUTPUT_JSON = REPO_ROOT / "data/task/dspredict/easy_expanded.json"

_KAGGLE_KEY: Optional[str] = None

def _kaggle_headers() -> dict:
    global _KAGGLE_KEY
    if _KAGGLE_KEY is None:
        cfg = Path.home() / ".kaggle" / "kaggle.json"
        _KAGGLE_KEY = json.loads(cfg.read_text())["key"]
    return {"Authorization": f"Bearer {_KAGGLE_KEY}"}

# ---------------------------------------------------------------------------
# Competition lists
# ---------------------------------------------------------------------------

# Season 3 gaps — downloadable immediately (200)
S3_GAPS = [
    "playground-series-s3e2",
    "playground-series-s3e4",
    "playground-series-s3e6",
    "playground-series-s3e8",
    "playground-series-s3e10",
    "playground-series-s3e12",
    "playground-series-s3e17",
    "playground-series-s3e18",
    "playground-series-s3e20",
    "playground-series-s3e23",
]

# TPS Season 1 & 2 — need rule acceptance first (403)
TPS_COMPETITIONS = [
    "tabular-playground-series-jan-2021",
    "tabular-playground-series-feb-2021",
    "tabular-playground-series-mar-2021",
    "tabular-playground-series-apr-2021",
    "tabular-playground-series-may-2021",
    "tabular-playground-series-jun-2021",
    "tabular-playground-series-jul-2021",
    "tabular-playground-series-aug-2021",
    "tabular-playground-series-sep-2021",
    "tabular-playground-series-oct-2021",
    "tabular-playground-series-nov-2021",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-jan-2022",
    "tabular-playground-series-feb-2022",
    "tabular-playground-series-mar-2022",
    "tabular-playground-series-apr-2022",
    "tabular-playground-series-may-2022",
    "tabular-playground-series-jun-2022",
    "tabular-playground-series-jul-2022",
    "tabular-playground-series-aug-2022",
    "tabular-playground-series-sep-2022",
    "tabular-playground-series-oct-2022",
    "tabular-playground-series-nov-2022",
]

# Hard split — tabular/TS only, downloadable immediately (200)
HARD_TABULAR = [
    "spaceship-titanic",
    "home-data-for-ml-course",
    "ieee-fraud-detection",
    "home-credit-default-risk",
    "santander-customer-transaction-prediction",
    "santander-value-prediction-challenge",
    "talkingdata-adtracking-fraud-detection",
    "microsoft-malware-prediction",
    "elo-merchant-category-recommendation",
    "otto-recommender-system",
    "predict-ai-model-runtime",
    "planttraits2024",
    "career-con-2019",
    "ventilator-pressure-prediction",
    "liverpool-ion-switching",
    "store-sales-time-series-forecasting",
    "recruit-restaurant-visitor-forecasting",
    "ashrae-energy-prediction",
    "m5-forecasting-accuracy",
    "LANL-Earthquake-Prediction",
    "vsb-power-line-fault-detection",
    "march-machine-learning-mania-2023",
    "mens-march-mania-2022",
    "womens-machine-learning-competition-2019",
    "mens-machine-learning-competition-2019",
]

TARGET_MAP = {
    "s3_gaps":      S3_GAPS,
    "tps":          TPS_COMPETITIONS,
    "playground":   S3_GAPS + TPS_COMPETITIONS,
    "hard_tabular": HARD_TABULAR,
    "all":          S3_GAPS + TPS_COMPETITIONS + HARD_TABULAR,
}

# ---------------------------------------------------------------------------
# Metric keyword extraction rules (first match wins)
# ---------------------------------------------------------------------------
METRIC_KEYWORDS = [
    ("rmse",     ["regression", "rmse"]),
    ("rmsle",    ["regression", "rmsle"]),
    ("mae",      ["regression", "mae"]),
    ("r2",       ["regression", "r2"]),
    ("log loss", ["classification", "log_loss"]),
    ("logloss",  ["classification", "log_loss"]),
    ("auc",      ["classification", "auc"]),
    ("accuracy", ["classification", "accuracy"]),
    ("f1",       ["classification", "f1"]),
    ("smape",    ["time_series", "smape"]),
    ("wrmsse",   ["time_series", "wrmsse"]),
    ("pinball",  ["time_series", "pinball"]),
    ("ndcg",     ["ranking", "ndcg"]),
    ("map",      ["ranking", "map"]),
]

# ---------------------------------------------------------------------------
# Status check helpers
# ---------------------------------------------------------------------------

def check_download_status(comp_name: str) -> int:
    """Return HTTP status code for download endpoint."""
    try:
        r = requests.get(
            f"https://www.kaggle.com/api/v1/competitions/data/download-all/{comp_name}",
            headers=_kaggle_headers(), stream=True, timeout=10,
        )
        return r.status_code
    except Exception:
        return -1


def print_status_table(competitions: list):
    print(f"\n{'Competition':<50} {'Status'}")
    print("-" * 60)
    ready, need_rules, not_found = [], [], []
    for comp in competitions:
        status = check_download_status(comp)
        symbol = {200: "✓ ready", 403: "✗ need rules", 404: "✗ not found"}.get(status, f"? {status}")
        print(f"  {comp:<48} {symbol}")
        if status == 200:
            ready.append(comp)
        elif status == 403:
            need_rules.append(comp)
        else:
            not_found.append(comp)
    print(f"\nSummary: {len(ready)} ready, {len(need_rules)} need rules, {len(not_found)} not found")
    return ready, need_rules, not_found


def print_rules_urls(competitions: list):
    """Print Kaggle rules URLs for competitions that need acceptance."""
    print("\nOpen these URLs in your browser and click 'I Understand and Accept':")
    print("=" * 70)
    for comp in competitions:
        print(f"  https://www.kaggle.com/competitions/{comp}/rules")
    print(f"\nTotal: {len(competitions)} URLs")


# ---------------------------------------------------------------------------
# Kaggle download (REST API, not kaggle CLI)
# ---------------------------------------------------------------------------

def download_competition(comp_name: str, dest_dir: Path, force: bool = False) -> bool:
    """Download competition data via Kaggle REST API. Returns True on success."""
    comp_dir = dest_dir / comp_name
    if comp_dir.exists() and any(comp_dir.iterdir()) and not force:
        print(f"  ✓ Already downloaded: {comp_name}")
        return True

    comp_dir.mkdir(parents=True, exist_ok=True)

    print(f"  ↓ Downloading {comp_name} ...")
    try:
        r = requests.get(
            f"https://www.kaggle.com/api/v1/competitions/data/download-all/{comp_name}",
            headers=_kaggle_headers(), stream=True, timeout=120,
        )
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        return False

    if r.status_code == 403:
        print(f"  ✗ 403 — accept rules first: https://www.kaggle.com/competitions/{comp_name}/rules")
        return False
    if r.status_code == 404:
        print(f"  ✗ 404 — competition not found: {comp_name}")
        return False
    if r.status_code != 200:
        print(f"  ✗ HTTP {r.status_code} — {r.text[:100]}")
        return False

    # Write to zip then extract
    zip_path = comp_dir / f"{comp_name}.zip"
    zip_path.write_bytes(r.content)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(comp_dir)
        zip_path.unlink()
        files = [f.name for f in comp_dir.iterdir()]
        print(f"  ✓ Extracted {len(files)} file(s): {', '.join(files[:5])}")
        return True
    except zipfile.BadZipFile:
        # Some competitions serve the file directly (not zipped)
        zip_path.rename(comp_dir / "data.bin")
        print(f"  ⚠ Not a zip; saved raw content to data.bin")
        return True


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

def read_file_safe(path: Path, max_chars: int = 6000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def collect_competition_text(comp_dir: Path) -> dict:
    result = {"data_description": "", "readme": "", "sample_submission_header": "", "file_list": []}
    for f in sorted(comp_dir.iterdir()):
        result["file_list"].append(f.name)
        nl = f.name.lower()
        if nl in ("data_description.txt", "description.txt"):
            result["data_description"] = read_file_safe(f)
        elif nl in ("readme.md", "readme.txt"):
            result["readme"] = read_file_safe(f)
        elif nl == "sample_submission.csv":
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").split("\n")[:2]
                result["sample_submission_header"] = "\n".join(lines)
            except Exception:
                pass
    return result


def extract_keywords_from_text(text: str) -> list:
    text_lower = text.lower()
    keywords = ["tabular"]
    for phrase, kws in METRIC_KEYWORDS:
        if phrase in text_lower:
            keywords.extend(kws)
            break
    for hint, kw in [
        ("fraud", "fraud_detection"), ("credit", "credit_risk"), ("energy", "energy"),
        ("sales", "retail"), ("medical", "healthcare"), ("health", "healthcare"),
        ("earthquake", "geophysics"), ("time series", "time_series"),
        ("forecasting", "time_series"), ("gradient boost", "tree_models"),
        ("lightgbm", "tree_models"), ("xgboost", "tree_models"),
    ]:
        if hint in text_lower and kw not in keywords:
            keywords.append(kw)
    return keywords[:6]


def generate_metadata_with_llm(
    comp_name: str, comp_texts: dict,
    model: str = "gpt-4o-mini", api_key: Optional[str] = None,
) -> dict:
    try:
        import litellm
    except ImportError:
        return {}

    if api_key:
        # Set the right env var for the provider
        if model.startswith("together_ai/"):
            os.environ["TOGETHER_API_KEY"] = api_key
        elif model.startswith("anthropic/"):
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key

    source = (comp_texts.get("data_description") or comp_texts.get("readme", ""))[:3000]
    sample_hdr = comp_texts.get("sample_submission_header", "")
    files = ", ".join(comp_texts.get("file_list", []))

    prompt = f"""Competition slug: "{comp_name}"

Data description (first 3000 chars):
{source}

sample_submission.csv header:
{sample_hdr}

Files: {files}

Output a JSON object with exactly these fields:
{{
  "competition_description": "<2-3 sentence summary>",
  "evaluation_metric": "<metric name + brief explanation>",
  "dataset_description": "<1-2 sentence dataset structure summary>",
  "keywords": ["<kw1>", "<kw2>", "<kw3>"]
}}

keywords must be 3-5 items from: regression, classification, time_series, tabular, \
feature_engineering, tree_models, rmse, rmsle, auc, log_loss, accuracy, mae, \
fraud_detection, housing, healthcare, credit_risk, retail, energy, geophysics, time_series

Respond with valid JSON only."""

    try:
        resp = litellm.completion(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"  ⚠ LLM metadata failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Entry builder
# ---------------------------------------------------------------------------

def build_entry(
    comp_name: str, comp_dir: Path,
    use_llm: bool = True, llm_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Optional[dict]:
    if not comp_dir.exists():
        print(f"  ✗ Data directory missing: {comp_dir}")
        return None

    texts = collect_competition_text(comp_dir)
    source = texts.get("data_description") or texts.get("readme", "")

    llm_meta = {}
    if use_llm and source:
        llm_meta = generate_metadata_with_llm(comp_name, texts, llm_model, api_key)

    comp_desc = llm_meta.get("competition_description") or (
        f"Kaggle competition: {comp_name}. " + source[:200].replace("\n", " ")
    )
    eval_metric = llm_meta.get("evaluation_metric") or _extract_metric_from_text(source)
    data_desc = llm_meta.get("dataset_description") or source[:300].replace("\n", " ")
    keywords = llm_meta.get("keywords") or extract_keywords_from_text(source)

    full_desc = (
        f"Challenge description:\n{comp_desc}\n\n"
        f"Evaluation metric:\n{eval_metric}\n\n"
        f"Data description:\n{data_desc}\n\n"
        f"Files: {', '.join(texts['file_list'][:10])}"
    )

    return {
        "challenge_name": comp_name,
        "description": full_desc,
        "docker_challenge_path": f"/data/{comp_name}",
        "competition_description": comp_desc,
        "evaluation_metric": eval_metric,
        "dataset_description": data_desc,
        "metadata": {"domain": "machine_learning", "keywords": keywords},
    }


def _extract_metric_from_text(text: str) -> str:
    text_lower = text.lower()
    for phrase, _ in METRIC_KEYWORDS:
        if phrase in text_lower:
            return phrase.upper()
    return "See competition description"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def get_existing_names() -> set:
    with open(EASY_JSON, encoding="utf-8") as f:
        return {e["challenge_name"] for e in json.load(f)}


def expand_dataset(
    competitions: list,
    dry_run: bool = False,
    force_download: bool = False,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    output_path: Path = OUTPUT_JSON,
) -> list:
    existing_names = get_existing_names()
    with open(EASY_JSON, encoding="utf-8") as f:
        existing_entries = json.load(f)

    new_competitions = [c for c in competitions if c not in existing_names]
    skipped = len(competitions) - len(new_competitions)
    if skipped:
        print(f"Skipping {skipped} already-present competitions")

    if not new_competitions:
        print("Nothing new to add.")
        return existing_entries

    print(f"Target: {len(new_competitions)} new competitions\n" + "-" * 60)

    new_entries, failed = [], []

    for i, comp_name in enumerate(new_competitions, 1):
        print(f"\n[{i}/{len(new_competitions)}] {comp_name}")

        if dry_run:
            print(f"  [DRY RUN] would download + generate entry")
            continue

        if not download_competition(comp_name, DATA_DIR, force=force_download):
            failed.append(comp_name)
            continue

        print(f"  ✎ Generating metadata ...")
        entry = build_entry(comp_name, DATA_DIR / comp_name, use_llm, llm_model, api_key)
        if entry is None:
            failed.append(comp_name)
            continue

        new_entries.append(entry)
        print(f"  ✓ keywords: {entry['metadata']['keywords']}")

    if dry_run:
        print(f"\n[DRY RUN] would add {len(new_competitions)} entries → {output_path}")
        return existing_entries

    merged = existing_entries + new_entries
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Original: {len(existing_entries)}  Added: {len(new_entries)}  Failed: {len(failed)}")
    print(f"Output:   {output_path}")

    if failed:
        print(f"\nFailed (accept rules first):")
        for c in failed:
            print(f"  https://www.kaggle.com/competitions/{c}/rules")

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Expand DSPredict easy split")
    parser.add_argument("--target", choices=list(TARGET_MAP), default="s3_gaps",
                        help="Preset competition list (default: s3_gaps)")
    parser.add_argument("--competitions", type=str, default=None,
                        help="Comma-separated competition slugs (overrides --target)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_JSON))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--check-status", action="store_true",
                        help="Check download status for all competitions in --target, then exit")
    parser.add_argument("--print-rules-urls", action="store_true",
                        help="Print rules URLs for TPS competitions that need acceptance, then exit")
    args = parser.parse_args()

    # Resolve competition list
    if args.competitions:
        competitions = [c.strip() for c in args.competitions.split(",")]
    else:
        competitions = TARGET_MAP.get(args.target, S3_GAPS)

    # Deduplicate preserving order
    seen: set = set()
    competitions = [c for c in competitions if not (c in seen or seen.add(c))]

    # Special actions
    if args.print_rules_urls:
        # Show only TPS (the ones likely to need acceptance)
        tps_in_list = [c for c in competitions if "tabular-playground" in c]
        if not tps_in_list:
            tps_in_list = TPS_COMPETITIONS
        print_rules_urls(tps_in_list)
        return

    if args.check_status:
        print_status_table(competitions)
        return

    print(f"DSPredict Dataset Expansion")
    print(f"Target: {args.target if not args.competitions else 'custom'}")
    print(f"Competitions: {len(competitions)}  |  Output: {args.output}")
    if args.dry_run:
        print("DRY RUN — no files will be written")

    expand_dataset(
        competitions=competitions,
        dry_run=args.dry_run,
        force_download=args.force_download,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
        api_key=args.api_key,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
