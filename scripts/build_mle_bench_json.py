#!/usr/bin/env python3
"""Build data/task/dspredict/mle_bench.json from MLE-Bench competition metadata.

Reads the 47 unique task names from data/task/dspredict/mle_bench_unique_tasks.txt
and emits a hard.json/mle_dojo.json-style task list.
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MLEB = ROOT / "examples/MLE_Bench_Eval/mle-bench/mlebench/competitions"
NAMES_FILE = ROOT / "data/task/dspredict/mle_bench_unique_tasks.txt"
OUT = ROOT / "data/task/dspredict/mle_bench.json"


def section(md: str, header: str) -> str:
    """Extract a markdown section by H1/H2 header (case-insensitive)."""
    pat = re.compile(rf"(^|\n)#{{1,3}}\s*{re.escape(header)}\s*\n(.*?)(?=\n#{{1,3}}\s|\Z)", re.S | re.I)
    m = pat.search(md)
    return m.group(2).strip() if m else ""


def first_paragraph(md_section: str) -> str:
    for para in md_section.split("\n\n"):
        para = para.strip()
        if para:
            return para
    return md_section.strip()


def build_entry(name: str) -> dict:
    comp_dir = MLEB / name
    desc_md = (comp_dir / "description.md").read_text(encoding="utf-8")
    cfg = yaml.safe_load((comp_dir / "config.yaml").read_text(encoding="utf-8"))

    overview = section(desc_md, "Description")
    if not overview:
        overview = section(desc_md, "Overview")
    eval_section = section(desc_md, "Evaluation")
    data_section = section(desc_md, "Dataset Description")
    if not data_section:
        data_section = section(desc_md, "Data")

    competition_description = "## Competition Description\n" + (overview or first_paragraph(desc_md))
    evaluation_metric = "### Metric\n" + (eval_section or "See competition page.")
    dataset_description = "Data description:\n" + (data_section or "See competition page.")

    pretty = name.replace("-", " ").title()
    description = (
        f"Challenge:\n# {pretty}\n\n"
        f"{competition_description}\n\n"
        f"## Evaluation\n{evaluation_metric}\n\n"
        f"{dataset_description}"
    )

    grader = (cfg.get("grader") or {}).get("name", "")

    return {
        "challenge_name": name,
        "description": description,
        "docker_challenge_path": f"/data/{name}",
        "competition_description": competition_description,
        "evaluation_metric": evaluation_metric,
        "dataset_description": dataset_description,
        "metadata": {
            "domain": "machine_learning",
            "keywords": [grader] if grader else [],
            "source": "mle-bench",
            "competition_type": cfg.get("competition_type", ""),
            "awards_medals": cfg.get("awards_medals", False),
        },
    }


def main():
    names = [l.strip() for l in NAMES_FILE.read_text().splitlines() if l.strip()]
    entries = [build_entry(n) for n in names]
    OUT.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"Wrote {len(entries)} entries to {OUT}")


if __name__ == "__main__":
    main()
