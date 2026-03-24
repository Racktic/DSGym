#!/usr/bin/env python3
"""
Student Rollout Script for Route B: On-Policy Distillation

This script uses the current student model to generate trajectories in DSGym,
automatically evaluating correctness. It serves as the core "on-policy" data
collection step — the student explores the environment with its own policy,
producing both correct and incorrect trajectories.

Key features:
  - Uses vLLM/multi-vLLM backend for fast local inference with student model
  - Auto-evaluates trajectory correctness via DSGym metrics
  - Separates correct vs incorrect trajectories for downstream use (RFT/DPO)
  - Supports resume from partial runs
  - Outputs trajectory statistics for curriculum learning

Usage:
    # Basic student rollout with local vLLM
    python student_rollout.py \
        --model /path/to/student_checkpoint \
        --dataset qrdata \
        --backend vllm \
        --k 8 \
        --output-dir ./route_b_outputs/iter_0/student_rollout

    # Multi-GPU rollout
    python student_rollout.py \
        --model /path/to/student_checkpoint \
        --dataset qrdata \
        --backend multi-vllm \
        --k 8 \
        --max-workers 4 \
        --output-dir ./route_b_outputs/iter_1/student_rollout

    # Teacher rollout for backfill (same script, different model)
    python student_rollout.py \
        --model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
        --dataset qrdata \
        --backend litellm \
        --k 3 \
        --output-dir ./route_b_outputs/iter_0/teacher_backfill \
        --failed-queries ./route_b_outputs/iter_0/student_rollout/failed_queries.json
"""

import json
import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add DSGym to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dsgym.synth.generators.trajectory_generator import (
    TrajectoryGenerator,
    TrajectoryConfig,
    create_trajectory_generator,
)
from dsgym.datasets import DatasetRegistry


# ============================================================================
# Correctness Evaluation
# ============================================================================

CORRECTNESS_METRICS = [
    "exact_match", "fuzzy_exact_match", "list_match",
    "equivalence_by_llm", "fast_equivalence_by_llm",
]


def evaluate_trajectory_correctness(trajectory: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Evaluate whether a trajectory produced a correct answer.

    Returns:
        (is_correct, best_score)
    """
    metrics = trajectory.get("metrics", {})
    best_score = 0.0
    is_correct = False

    for metric_name in CORRECTNESS_METRICS:
        if metric_name in metrics:
            metric_data = metrics[metric_name]
            if isinstance(metric_data, dict):
                score = metric_data.get("score", 0.0)
            elif isinstance(metric_data, (int, float)):
                score = float(metric_data)
            else:
                continue
            best_score = max(best_score, score)
            if score >= 0.99:
                is_correct = True

    return is_correct, best_score


# ============================================================================
# Post-processing: separate correct / incorrect
# ============================================================================

@dataclass
class RolloutStats:
    """Statistics for a rollout run."""
    total_samples: int = 0
    total_trajectories: int = 0
    correct_trajectories: int = 0
    incorrect_trajectories: int = 0
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0
    avg_turns: float = 0.0
    avg_execution_time: float = 0.0
    failed_query_count: int = 0
    succeeded_query_count: int = 0


def classify_trajectories(
    output_dir: str,
    k: int = 8,
) -> Tuple[RolloutStats, List[str], List[str]]:
    """
    Walk through all trajectory files produced by TrajectoryGenerator,
    classify them as correct/incorrect, and emit summary statistics.

    Returns:
        (stats, failed_sample_ids, succeeded_sample_ids)
    """
    predictions_dir = Path(output_dir) / "predictions"
    if not predictions_dir.exists():
        # TrajectoryGenerator may put files directly in output_dir
        predictions_dir = Path(output_dir)

    # Group trajectory files by sample_id
    sample_trajs: Dict[str, List[Dict]] = defaultdict(list)
    all_files = sorted(predictions_dir.glob("*.json"))

    for fp in all_files:
        if fp.name.startswith("metrics") or fp.name.startswith("config"):
            continue
        try:
            with open(fp, "r") as f:
                traj = json.load(f)
            sample_id = traj.get("sample_id", fp.stem)
            traj["_file_path"] = str(fp)
            sample_trajs[sample_id].append(traj)
        except (json.JSONDecodeError, KeyError):
            continue

    stats = RolloutStats()
    stats.total_samples = len(sample_trajs)

    all_turns = []
    all_exec_times = []
    failed_sample_ids: List[str] = []
    succeeded_sample_ids: List[str] = []

    correct_dir = Path(output_dir) / "correct"
    incorrect_dir = Path(output_dir) / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    pass_at_1_hits = 0
    pass_at_k_hits = 0

    for sample_id, trajs in sample_trajs.items():
        sample_has_correct = False
        first_correct = False

        for i, traj in enumerate(trajs):
            stats.total_trajectories += 1
            is_correct, score = evaluate_trajectory_correctness(traj)

            turns = traj.get("turns", traj.get("total_turns", 0))
            exec_time = traj.get("execution_time", 0.0)
            all_turns.append(turns)
            all_exec_times.append(exec_time)

            src_path = traj["_file_path"]
            dst_name = Path(src_path).name

            if is_correct:
                stats.correct_trajectories += 1
                sample_has_correct = True
                if i == 0:
                    first_correct = True
                shutil.copy2(src_path, correct_dir / dst_name)
            else:
                stats.incorrect_trajectories += 1
                shutil.copy2(src_path, incorrect_dir / dst_name)

        if first_correct:
            pass_at_1_hits += 1
        if sample_has_correct:
            pass_at_k_hits += 1
            succeeded_sample_ids.append(sample_id)
        else:
            failed_sample_ids.append(sample_id)

    stats.pass_at_1 = pass_at_1_hits / max(stats.total_samples, 1)
    stats.pass_at_k = pass_at_k_hits / max(stats.total_samples, 1)
    stats.avg_turns = sum(all_turns) / max(len(all_turns), 1)
    stats.avg_execution_time = sum(all_exec_times) / max(len(all_exec_times), 1)
    stats.failed_query_count = len(failed_sample_ids)
    stats.succeeded_query_count = len(succeeded_sample_ids)

    return stats, failed_sample_ids, succeeded_sample_ids


# ============================================================================
# Main
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Student/Teacher Rollout for Route B On-Policy Distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model / backend
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or API model name")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["litellm", "vllm", "multi-vllm", "sglang"],
                        help="Inference backend (default: vllm)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for litellm backend")

    # Dataset
    parser.add_argument("--dataset", type=str, default="qrdata",
                        choices=["daeval", "qrdata", "dabstep", "dsbio",
                                 "discoverybench", "dspredict-easy", "dspredict-hard"],
                        help="Dataset to rollout on")
    parser.add_argument("--dataset-type", type=str, default="original",
                        choices=["original", "synthetic"],
                        help="Dataset type (for qrdata)")
    parser.add_argument("--synthetic-path", type=str, default=None,
                        help="Path to synthetic dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of samples")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start index in dataset (0-based)")

    # Generation
    parser.add_argument("--k", type=int, default=8,
                        help="Number of trajectories per sample (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--max-turns", type=int, default=15,
                        help="Max turns per trajectory")
    parser.add_argument("--max-workers", type=int, default=24,
                        help="Parallel workers for trajectory generation")

    # Environment
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000",
                        help="Docker execution manager URL")

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for trajectories")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (for logging)")

    # Teacher back-fill mode
    parser.add_argument("--failed-queries", type=str, default=None,
                        help="JSON file listing sample_ids to re-attempt "
                             "(teacher backfill mode)")

    # Flags
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip post-rollout classification step")
    parser.add_argument("--classify-only", action="store_true",
                        help="Only run classification on existing output dir")

    return parser


def load_failed_query_ids(path: str) -> Set[str]:
    """Load the set of sample_ids that failed in a previous student rollout."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    elif isinstance(data, dict) and "failed_sample_ids" in data:
        return set(data["failed_sample_ids"])
    raise ValueError(f"Unrecognised failed-queries format in {path}")


def main():
    parser = create_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # classify-only shortcut
    # ------------------------------------------------------------------
    if args.classify_only:
        print("📊 Running classification only …")
        stats, failed_ids, succ_ids = classify_trajectories(
            args.output_dir, k=args.k
        )
        _save_stats_and_failed(args.output_dir, stats, failed_ids, succ_ids)
        _print_stats(stats)
        return 0

    # ------------------------------------------------------------------
    # Build trajectory config
    # ------------------------------------------------------------------
    run_name = args.run_name or f"{args.dataset}_student_k{args.k}"
    config = TrajectoryConfig(
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        k=args.k,
        max_workers=args.max_workers,
        max_turns=args.max_turns,
        manager_url=args.manager_url,
        api_key=args.api_key,
        dataset_name=args.dataset if "dspredict" not in args.dataset else "dspredict",
        synthetic_path=args.synthetic_path,
        compute_metrics=True,
        output_dir=args.output_dir,
        run_name=run_name,
    )

    generator = TrajectoryGenerator(config)

    # ------------------------------------------------------------------
    # Load samples (optionally filtered to failed queries only)
    # ------------------------------------------------------------------
    print(f"📂 Loading dataset: {args.dataset} …")
    generator._initialize_components()

    load_kwargs: Dict[str, Any] = {}
    if args.limit:
        load_kwargs["limit"] = args.limit
    if args.start_index:
        load_kwargs["start_index"] = args.start_index

    samples = generator.dataset.load(**load_kwargs)
    print(f"   Loaded {len(samples)} samples")

    # Filter to failed queries if in teacher-backfill mode
    if args.failed_queries:
        failed_ids = load_failed_query_ids(args.failed_queries)
        samples = [s for s in samples
                   if s.get("sample_id", s.get("id", "")) in failed_ids]
        print(f"   Filtered to {len(samples)} failed queries for teacher backfill")

    if not samples:
        print("⚠️  No samples to process. Exiting.")
        return 0

    # ------------------------------------------------------------------
    # Generate trajectories
    # ------------------------------------------------------------------
    print(f"🚀 Generating {args.k} trajectories per sample with {args.model} …")
    start_time = time.time()
    generator.generate(samples)
    elapsed = time.time() - start_time
    print(f"✅ Generation complete in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Post-process: classify & save stats
    # ------------------------------------------------------------------
    if not args.skip_classify:
        print("📊 Classifying trajectories …")
        stats, failed_ids, succ_ids = classify_trajectories(
            args.output_dir, k=args.k
        )
        _save_stats_and_failed(args.output_dir, stats, failed_ids, succ_ids)
        _print_stats(stats)

    return 0


# ============================================================================
# Helpers
# ============================================================================

def _save_stats_and_failed(
    output_dir: str,
    stats: RolloutStats,
    failed_ids: List[str],
    succ_ids: List[str],
):
    base = Path(output_dir)
    with open(base / "rollout_stats.json", "w") as f:
        json.dump(asdict(stats), f, indent=2)
    with open(base / "failed_queries.json", "w") as f:
        json.dump({"failed_sample_ids": failed_ids}, f, indent=2)
    with open(base / "succeeded_queries.json", "w") as f:
        json.dump({"succeeded_sample_ids": succ_ids}, f, indent=2)
    print(f"   Saved stats → {base / 'rollout_stats.json'}")
    print(f"   Saved failed queries → {base / 'failed_queries.json'}")


def _print_stats(stats: RolloutStats):
    print()
    print("=" * 60)
    print("  ROLLOUT STATISTICS")
    print("=" * 60)
    print(f"  Total samples:            {stats.total_samples}")
    print(f"  Total trajectories:       {stats.total_trajectories}")
    print(f"  Correct trajectories:     {stats.correct_trajectories}")
    print(f"  Incorrect trajectories:   {stats.incorrect_trajectories}")
    print(f"  Pass@1:                   {stats.pass_at_1:.3f}")
    print(f"  Pass@k:                   {stats.pass_at_k:.3f}")
    print(f"  Avg turns:                {stats.avg_turns:.1f}")
    print(f"  Avg execution time (s):   {stats.avg_execution_time:.1f}")
    print(f"  Queries with ≥1 correct:  {stats.succeeded_query_count}")
    print(f"  Queries with 0 correct:   {stats.failed_query_count}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
