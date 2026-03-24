#!/usr/bin/env python3
"""
DSGym trajectory generation command.

Supports two modes:
- --agent teacher: Generate structured VGS trajectories for SFT training (TeacherAgent)
- --agent standard: Standard trajectory generation (original, future implementation)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def add_generate_parser(subparsers):
    """Add generate command parser."""
    parser = subparsers.add_parser(
        "generate",
        help="Generate trajectories for synthetic data creation",
        description="Generate multiple trajectories per sample for pass@k evaluation or VGS SFT training",
    )

    # Agent type
    parser.add_argument(
        "--agent",
        type=str,
        default="standard",
        choices=["standard", "teacher", "eet", "aide"],
        help="Agent type: 'standard' (original), 'teacher' (VGS structured trajectories), 'eet' (explore-exploit-terminate), or 'aide' (draft-improve-debug)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'gpt-4o', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="litellm",
        choices=["litellm", "vllm", "sglang"],
        help="Backend to use for model inference",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "daeval",
            "discoverybench",
            "qrdata",
            "dabstep",
            "dspredict",
            "dspredict-easy",
            "dspredict-hard",
            "bio",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "--synthetic-path",
        type=str,
        default=None,
        help="Path to synthetic dataset (optional)",
    )

    # Trajectory generation
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for trajectory generation",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Number of trajectories to generate per sample (standard mode)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to process",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trajectory_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name for output files",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip metric computation (faster generation)",
    )

    # Infrastructure
    parser.add_argument(
        "--manager-url",
        type=str,
        default="http://localhost:5000",
        help="Code sandbox manager URL",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Maximum turns per sample",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=24,
        help="Maximum number of parallel workers",
    )

    # API keys
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (uses environment variable if not provided)",
    )

    return parser


def run_generate(args) -> int:
    """Run trajectory generation command."""
    if args.agent == "teacher":
        return _run_teacher_generate(args)
    elif args.agent == "eet":
        return _run_eet_generate(args)
    elif args.agent == "aide":
        return _run_aide_generate(args)
    else:
        return _run_standard_generate(args)


def _run_standard_generate(args) -> int:
    """Original standard trajectory generation (not yet implemented)."""
    print("🎯 DSGym Trajectory Generation")
    print("⚠️ Generate functionality is not yet implemented.")
    print("Please use examples/generate_trajectories.py for now.")

    # TODO: Implement standard trajectory generation CLI
    # This will use the TrajectoryGenerator from dsgym.synth

    return 0


def _run_eet_generate(args) -> int:
    """Generate EET (Explore-Exploit-Terminate) trajectories."""
    from dsgym.datasets import DatasetRegistry
    from dsgym.agents.vgs import EETAgent
    from dsgym.eval import Evaluator
    from dsgym.eval.utils import EvaluationConfig

    print("Starting EET Trajectory Generation")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Dataset: {args.dataset}")
    print(f"Temperature: {args.temperature}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output: {args.output_dir}")
    print("-" * 50)

    os.makedirs(args.output_dir, exist_ok=True)

    agent_config = {
        "manager_url": args.manager_url,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "output_dir": args.output_dir,
        "trajectory_output_dir": args.output_dir,
        "submission_dir": "./submissions",
    }

    if args.backend == "litellm" and args.api_key:
        agent_config["api_key"] = args.api_key

    try:
        agent = EETAgent(
            backend=args.backend,
            model=args.model,
            **agent_config,
        )
        print("EETAgent initialized")
    except Exception as e:
        print(f"Failed to initialize EETAgent: {e}")
        return 1

    print(f"Loading {args.dataset} dataset...")
    try:
        dataset_config = {}
        load_config = {"limit": args.limit}
        dataset_name = args.dataset

        if "dspredict" in dataset_name:
            dataset_config["split"] = dataset_name.split("-")[-1]
            dataset_name = dataset_name.split("-")[0]
            dataset_config["virtual_data_root"] = "/data"
            load_config["split"] = dataset_config["split"]

        dataset = DatasetRegistry.load(dataset_name, **dataset_config)
        samples = dataset.load(**load_config)
        print(f"Loaded {len(samples)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return 1

    evaluator = Evaluator(
        protocol="multi_turn",
        dataset=dataset,
        parallel_workers=min(args.max_workers, 1),
    )

    run_name = args.run_name or (
        f"eet_{args.dataset}_{args.backend}_{args.model.replace('/', '_')}"
    )
    config = EvaluationConfig(
        model_name=args.model,
        backend_type=args.backend,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        run_name=run_name,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )

    print("Starting EET trajectory generation...")
    try:
        results = evaluator.evaluate(
            agent=agent,
            tasks=samples,
            config=config,
            save_results=True,
        )
        print("EET trajectory generation completed!")
        print(f"Results saved to: {args.output_dir}")

        if "metrics" in results:
            metrics = results["metrics"]
            for key in [
                "success_rate",
                "total_samples",
                "successful_samples",
                "average_execution_time",
            ]:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")

        if "dspredict" in args.dataset:
            dataset.print_dspredict_results_overview(results["results"])

        return 0

    except Exception as e:
        print(f"EET trajectory generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def _run_aide_generate(args) -> int:
    """Generate AIDE (Draft/Improve/Debug) trajectories."""
    from dsgym.datasets import DatasetRegistry
    from dsgym.agents.vgs import AIDEAgent
    from dsgym.eval import Evaluator
    from dsgym.eval.utils import EvaluationConfig

    print("Starting AIDE Trajectory Generation")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Dataset: {args.dataset}")
    print(f"Temperature: {args.temperature}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output: {args.output_dir}")
    print("-" * 50)

    os.makedirs(args.output_dir, exist_ok=True)

    agent_config = {
        "manager_url": args.manager_url,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "output_dir": args.output_dir,
        "trajectory_output_dir": args.output_dir,
        "submission_dir": "./submissions",
    }

    if args.backend == "litellm" and args.api_key:
        agent_config["api_key"] = args.api_key

    try:
        agent = AIDEAgent(
            backend=args.backend,
            model=args.model,
            **agent_config,
        )
        print("AIDEAgent initialized")
    except Exception as e:
        print(f"Failed to initialize AIDEAgent: {e}")
        return 1

    print(f"Loading {args.dataset} dataset...")
    try:
        dataset_config = {}
        load_config = {"limit": args.limit}
        dataset_name = args.dataset

        if "dspredict" in dataset_name:
            dataset_config["split"] = dataset_name.split("-")[-1]
            dataset_name = dataset_name.split("-")[0]
            dataset_config["virtual_data_root"] = "/data"
            load_config["split"] = dataset_config["split"]

        dataset = DatasetRegistry.load(dataset_name, **dataset_config)
        samples = dataset.load(**load_config)
        print(f"Loaded {len(samples)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return 1

    evaluator = Evaluator(
        protocol="multi_turn",
        dataset=dataset,
        parallel_workers=min(args.max_workers, 1),
    )

    run_name = args.run_name or (
        f"aide_{args.dataset}_{args.backend}_{args.model.replace('/', '_')}"
    )
    config = EvaluationConfig(
        model_name=args.model,
        backend_type=args.backend,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        run_name=run_name,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )

    print("Starting AIDE trajectory generation...")
    try:
        results = evaluator.evaluate(
            agent=agent,
            tasks=samples,
            config=config,
            save_results=True,
        )
        print("AIDE trajectory generation completed!")
        print(f"Results saved to: {args.output_dir}")

        if "metrics" in results:
            metrics = results["metrics"]
            for key in [
                "success_rate",
                "total_samples",
                "successful_samples",
                "average_execution_time",
            ]:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")

        if "dspredict" in args.dataset:
            dataset.print_dspredict_results_overview(results["results"])

        return 0

    except Exception as e:
        print(f"AIDE trajectory generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def _run_teacher_generate(args) -> int:
    """Generate structured VGS trajectories with TeacherAgent."""
    from dsgym.datasets import DatasetRegistry
    from dsgym.agents.vgs import TeacherAgent
    from dsgym.eval import Evaluator
    from dsgym.eval.utils import EvaluationConfig

    print("Starting VGS Teacher Trajectory Generation")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Dataset: {args.dataset}")
    print(f"Temperature: {args.temperature}")
    print(f"Max turns: {args.max_turns}")
    print(f"Output: {args.output_dir}")
    print("-" * 50)

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize TeacherAgent
    print("Initializing TeacherAgent...")
    agent_config = {
        "manager_url": args.manager_url,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "output_dir": args.output_dir,
        "trajectory_output_dir": args.output_dir,
        "submission_dir": "./submissions",
    }

    if args.backend == "litellm" and args.api_key:
        agent_config["api_key"] = args.api_key

    try:
        agent = TeacherAgent(
            backend=args.backend,
            model=args.model,
            **agent_config,
        )
        print("TeacherAgent initialized")
    except Exception as e:
        print(f"Failed to initialize TeacherAgent: {e}")
        return 1

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    try:
        dataset_config = {}
        load_config = {"limit": args.limit}
        dataset_name = args.dataset

        if "dspredict" in dataset_name:
            dataset_config["split"] = dataset_name.split("-")[-1]
            dataset_name = dataset_name.split("-")[0]
            dataset_config["virtual_data_root"] = "/data"
            load_config["split"] = dataset_config["split"]

        dataset = DatasetRegistry.load(dataset_name, **dataset_config)
        samples = dataset.load(**load_config)
        print(f"Loaded {len(samples)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return 1

    # Create evaluator (reuse existing pipeline for running tasks)
    evaluator = Evaluator(
        protocol="multi_turn",
        dataset=dataset,
        parallel_workers=min(args.max_workers, 1),  # teacher default single-worker
    )

    run_name = args.run_name or (
        f"teacher_{args.dataset}_{args.backend}_{args.model.replace('/', '_')}"
    )
    config = EvaluationConfig(
        model_name=args.model,
        backend_type=args.backend,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        run_name=run_name,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )

    # Run generation
    print("Starting trajectory generation...")
    try:
        results = evaluator.evaluate(
            agent=agent,
            tasks=samples,
            config=config,
            save_results=True,
        )
        print("Trajectory generation completed!")
        print(f"Results saved to: {args.output_dir}")

        # Print summary
        if "metrics" in results:
            metrics = results["metrics"]
            for key in [
                "success_rate",
                "total_samples",
                "successful_samples",
                "average_execution_time",
            ]:
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")

        if "dspredict" in args.dataset:
            dataset.print_dspredict_results_overview(results["results"])

        return 0

    except Exception as e:
        print(f"Trajectory generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSGym trajectory generation")
    add_generate_parser(
        parser._subparsers_action.add_parser
        if hasattr(parser, "_subparsers_action")
        else parser.add_subparsers()
    )
    parsed_args = parser.parse_args()
    sys.exit(run_generate(parsed_args))
