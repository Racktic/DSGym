#!/usr/bin/env python3
"""
Batch trajectory generation for DSPredict (easy expanded split).

Wraps TrajectoryGenerator to run teacher rollout on all expanded competitions.
Supports resuming from a checkpoint (skip already-done samples).

Usage:
  python scripts/generate_dspredict_trajectories.py \\
      --model "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput" \\
      --backend litellm \\
      --k 8 \\
      --output-dir ./trajectory_outputs/dspredict_expanded \\
      --dataset-json data/task/dspredict/easy_expanded.json

  # Resume from index 20
  python scripts/generate_dspredict_trajectories.py \\
      --model "..." --start-index 20 --output-dir ./trajectory_outputs/dspredict_expanded

  # Dry run (print plan)
  python scripts/generate_dspredict_trajectories.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_JSON = REPO_ROOT / "data/task/dspredict/easy_expanded.json"
FALLBACK_JSON = REPO_ROOT / "data/task/dspredict/easy.json"


def count_existing_trajectories(output_dir: Path, n_samples: int, k: int) -> list[int]:
    """Return list of sample indices that already have all k trajectories."""
    predictions_dir = output_dir / "predictions"
    if not predictions_dir.exists():
        return []

    done = []
    for i in range(n_samples):
        traj_files = list(predictions_dir.glob(f"prediction_{i}_traj_*.json"))
        if len(traj_files) >= k:
            done.append(i)
    return done


def load_samples_from_dspredict_json(json_path: Path, virtual_data_root: str = "/data") -> list:
    """
    Load DSPredict competition entries and convert to the standard sample format
    expected by TrajectoryGenerator (same as what DSPredictDataset.load() produces).
    """
    from dsgym.datasets import DatasetRegistry

    # DSPredictDataset reads from a file path via the 'split' arg, but it's
    # hard-coded to easy/hard/lite. We work around by using the registry
    # with a custom split that points to our expanded json.
    # Simplest: just load the dataset directly, temporarily overriding the path.

    # Patch: register a temporary dataset pointing at expanded json
    from dsgym.datasets.loaders.dspredict import DSPredictDataset
    from dsgym.datasets.config import RAW_DATA_DIR, get_task_path

    class ExpandedDSPredictDataset(DSPredictDataset):
        def load(self, limit=None, split="expanded", start_index=0, **kwargs):
            import json as _json
            from dsgym.datasets.utils import (
                apply_limit_and_start,
                validate_file_exists,
                create_standard_task,
                construct_data_paths,
            )
            from dsgym.datasets.prompts import SYSTEM_PROMPT_DSPREDICT
            from dsgym.datasets.loaders.dspredict import create_dspredict_prompt
            from dsgym.datasets.loaders.kaggle_downloader import KaggleChallengeDownloader

            dataset_path = str(json_path)
            validate_file_exists(dataset_path, "Expanded DSPredict dataset")

            with open(dataset_path, "r", encoding="utf-8") as f:
                items = _json.load(f)

            items = apply_limit_and_start(
                items, limit, start_index, random_sample=False,
                random_seed=self.config.get("random_seed", 42)
            )

            samples = []
            kaggle_data_dir = RAW_DATA_DIR / "dspredict-easy"
            downloader = None

            for idx, item in enumerate(items):
                docker_path = item["docker_challenge_path"]
                challenge_dir = docker_path.replace("/data/", "") if docker_path.startswith("/data/") else docker_path.split("/")[-1]

                competition_path = kaggle_data_dir / challenge_dir
                if not competition_path.exists():
                    print(f"  ⚠ Data not found locally: {challenge_dir} — skipping")
                    continue

                data_paths = construct_data_paths(
                    relative_paths=[challenge_dir],
                    dataset_name="dspredict-easy",
                    data_root=RAW_DATA_DIR,
                    virtual_data_root=self.virtual_data_root,
                )

                user_content = create_dspredict_prompt(
                    challenge_name=item["challenge_name"],
                    description=item["description"],
                    data_paths=data_paths,
                )

                extra_info = {
                    "challenge_name": item["challenge_name"],
                    "docker_challenge_path": item["docker_challenge_path"],
                    "data_files": data_paths,
                    "question": user_content,
                    "index": start_index + idx,
                    "source": "dspredict_expanded",
                    "metadata_id": item["challenge_name"],
                    "query_id": item["challenge_name"],
                    "id": item["challenge_name"],
                }

                standard_sample = create_standard_task(
                    prompt_content=user_content,
                    ground_truth="",
                    extra_info=extra_info,
                    system_prompt=SYSTEM_PROMPT_DSPREDICT,
                )
                samples.append(standard_sample)

            self._samples = samples
            return samples

    return ExpandedDSPredictDataset(virtual_data_root=virtual_data_root)


def main():
    parser = argparse.ArgumentParser(description="Generate DSPredict trajectories (expanded dataset)")

    # Model / backend
    parser.add_argument("--model", type=str,
                        default="together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
                        help="Teacher model (litellm name)")
    parser.add_argument("--backend", type=str, default="litellm",
                        choices=["litellm", "vllm", "sglang", "multi-vllm"])
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (or set env var for the provider)")

    # Dataset
    parser.add_argument("--dataset-json", type=str, default=str(DEFAULT_JSON),
                        help="Path to expanded easy.json")

    # Generation settings
    parser.add_argument("--k", type=int, default=8,
                        help="Trajectories per competition (default: 8)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000")

    # Range / resume
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start from this sample index (for resuming)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N samples")

    # Output
    parser.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "trajectory_outputs/dspredict_expanded"))

    # Misc
    parser.add_argument("--compute-metrics", action="store_true", default=False,
                        help="Compute metrics (DSPredict has no fixed answer, so disabled by default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running generation")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Automatically detect start-index from existing predictions")

    args = parser.parse_args()

    # Resolve dataset json
    dataset_json = Path(args.dataset_json)
    if not dataset_json.exists():
        print(f"⚠ {dataset_json} not found, falling back to {FALLBACK_JSON}")
        dataset_json = FALLBACK_JSON

    # Load competition list to show summary
    with open(dataset_json, encoding="utf-8") as f:
        all_entries = json.load(f)

    print(f"DSPredict Trajectory Generation")
    print(f"  Dataset JSON:  {dataset_json}")
    print(f"  Competitions:  {len(all_entries)}")
    print(f"  Model:         {args.model}")
    print(f"  Backend:       {args.backend}")
    print(f"  k:             {args.k}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Max turns:     {args.max_turns}")
    print(f"  Max workers:   {args.max_workers}")
    print(f"  Output dir:    {args.output_dir}")

    output_dir = Path(args.output_dir)

    # Auto-resume: find first index that doesn't have all k trajectories
    start_index = args.start_index
    if args.auto_resume:
        done = count_existing_trajectories(output_dir, len(all_entries), args.k)
        if done:
            # Find first incomplete index
            done_set = set(done)
            start_index = next((i for i in range(len(all_entries)) if i not in done_set), len(all_entries))
            print(f"  Auto-resume:   starting from index {start_index} ({len(done)} already complete)")
        else:
            print(f"  Auto-resume:   no existing trajectories found, starting from 0")

    n_to_process = min(args.limit or len(all_entries), len(all_entries) - start_index)
    print(f"  Start index:   {start_index}")
    print(f"  To process:    {n_to_process} competitions")
    print(f"  Est. total:    {n_to_process * args.k} trajectories")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without running generation.")
        return

    # Confirm
    print()

    # Import DSGym components
    from dsgym.synth.generators.trajectory_generator import TrajectoryConfig, TrajectoryGenerator

    # Build config for DSPredict (no metrics, since there's no fixed ground truth)
    config = TrajectoryConfig(
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        k=args.k,
        max_workers=args.max_workers,
        max_turns=args.max_turns,
        manager_url=args.manager_url,
        api_key=args.api_key,
        dataset_name="dspredict",
        compute_metrics=args.compute_metrics,
        output_dir=args.output_dir,
        run_name=f"dspredict_expanded_{args.model.replace('/', '_').replace(':', '_')}_k{args.k}",
    )

    generator = TrajectoryGenerator(config)

    # Load samples via our custom dataset wrapper
    dataset_wrapper = load_samples_from_dspredict_json(dataset_json)
    all_samples = dataset_wrapper.load()

    print(f"Loaded {len(all_samples)} samples with local data available")

    if not all_samples:
        print("ERROR: No samples loaded. Check that competition data is downloaded.")
        sys.exit(1)

    # Run generation
    results = generator.generate(
        samples=all_samples,
        limit=args.limit,
        start_index=start_index,
        show_progress=True,
    )

    print(f"\nGeneration complete!")
    print(f"  Total trajectories: {results['total_trajectories']}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
