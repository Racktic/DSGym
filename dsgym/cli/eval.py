#!/usr/bin/env python3
"""
DSGym evaluation command.

This command provides a unified interface for evaluating agents on datasets,
similar to the example scripts but with a standardized CLI.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add DSGym to path
DSGYM_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(DSGYM_ROOT))

from dsgym.datasets import DatasetRegistry
from dsgym.agents import ReActDSAgent, DSPredictReActAgent
from dsgym.eval import Evaluator
from dsgym.eval.utils import EvaluationConfig


def add_eval_parser(subparsers):
    """Add eval command parser."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate agent on dataset",
        description="Evaluate LLM agents on data science datasets"
    )
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'gpt-4', 'together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput')")
    parser.add_argument("--backend", type=str, default="litellm",
                       choices=["litellm", "vllm", "sglang", "multi-vllm"],
                       help="Backend to use for model inference")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["daeval", "discoverybench", "qrdata", "dabstep", "dspredict-easy", "dspredict-hard", "dspredict-swap", "dspredict-hard-swap", "dspredict-mledojo", "dspredict-hard-rejected", "dspredict-easy-claude-retry", "dspredict-mledojo-remaining", "dspredict-mle-bench", "dspredict-easy-train", "dspredict-easy-test", "dspredict-easy-test-retry", "dspredict-hard-train", "dspredict-hard-test", "dspredict-hard-test-variance2", "dspredict-hard-test-variance3", "dspredict-hard-train-gpt-remaining", "bio"],
                       help="Dataset to evaluate on")
    parser.add_argument("--limit", type=int, default=None,
                       help="Number of samples to evaluate")
    parser.add_argument("--synthetic-path", type=str, default=None,
                       help="Path to synthetic dataset (optional)")
    
    # Evaluation configuration
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name for output files")
    parser.add_argument("--max-turns", type=int, default=15,
                       help="Maximum turns per sample")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Maximum tokens per generation (default: 1524 for litellm)")

    # Infrastructure
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000",
                       help="Code sandbox manager URL")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (auto-set based on backend if not provided)")
    
    # Agent type
    parser.add_argument("--agent", type=str, default="react",
                       choices=["react", "aide"],
                       help="Agent type: 'react' (default) or 'aide' (draft-improve-debug)")

    # AIDE-specific
    parser.add_argument("--num-drafts", type=int, default=None,
                       help="AIDE: number of initial draft turns (default: 5)")
    parser.add_argument("--no-draft-memory", action="store_true", default=False,
                       help="AIDE: skip cross-task memory injection during draft phase")
    parser.add_argument("--best-node-strategy", type=str, default="latest",
                       choices=["latest", "best"],
                       help="AIDE: how to select node for Improve ('latest'=most recent good node, 'best'=highest scoring node)")

    parser.add_argument("--memory-version", type=str, default="v4",
                       choices=["v4", "v5", "v6"],
                       help="AIDE: memory version ('v4'=LLM summary, 'v5'=V2-style cross-task, 'v6'=simplified output format + is_best tracking)")
    parser.add_argument("--no-task-memory", action="store_true", default=False,
                       help="AIDE: skip task-internal memory injection into prompts (summary LLM still runs)")
    parser.add_argument("--no-cross-memory", action="store_true", default=False,
                       help="AIDE: disable cross-task memory (no read/write)")
    parser.add_argument("--no-cross-memory-write", action="store_true", default=False,
                       help="AIDE: read cross-task memory but do NOT write new entries (use with offline-built enriched memory + .npy to avoid corrupting alignment)")

    parser.add_argument("--log-degradation", action="store_true", default=False,
                       help="AIDE V5: also log failed improve attempts (score degradation) to cross-task memory")
    parser.add_argument("--sticky-cross-memory", action="store_true", default=False,
                       help="AIDE ablation: retrieve cross-task memory once at turn 1 (mixing draft/improve/debug pools) and inject only into the first turn's instruction. Disables per-turn retrieval.")
    parser.add_argument("--sticky-top-k", type=int, default=15,
                       help="AIDE sticky ablation: number of cross-task entries to sample from the union of three action pools (default: 15)")

    parser.add_argument("--no-think", action="store_true", default=False,
                       help="Disable thinking mode for Qwen3 models (pass enable_thinking=False to chat template)")

    # Cross-task memory
    parser.add_argument("--memory-path", type=str, default=None,
                       help="Path to cross-task memory JSON file (enables cross-task experience sharing)")

    # API keys
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (uses environment variable if not provided)")
    parser.add_argument("--base-url", type=str, default=None,
                       help="Base URL for API endpoint (e.g., LiteLLM proxy or OpenRouter)")

    return parser


def run_eval(args) -> int:
    """Run evaluation command."""
    print("🚀 Starting DSGym Evaluation")
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.limit if args.limit is not None else 'all'}")
    
    # Set default max_workers based on backend
    if args.max_workers is None:
        if args.backend == "litellm":
            args.max_workers = 24
        elif args.backend == "multi-vllm":
            args.max_workers = 8  # one worker per GPU instance
        else:  # vllm or sglang
            args.max_workers = 1
    
    print(f"Max workers: {args.max_workers}")
    print("-" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize agent
    print("🤖 Initializing agent...")
    agent_config = {
        "manager_url": args.manager_url,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "output_dir": args.output_dir,
    }
    
    if args.no_think:
        agent_config["enable_thinking"] = False

    if args.backend == "litellm" and args.api_key:
        agent_config["api_key"] = args.api_key
    if args.backend == "litellm" and args.base_url:
        agent_config["base_url"] = args.base_url
    if args.max_tokens:
        agent_config["max_tokens"] = args.max_tokens
    
    try:
        if args.agent == "aide" and "dspredict" in args.dataset:
            from dsgym.agents import AIDEAgent
            aide_kwargs = {}
            if args.num_drafts is not None:
                aide_kwargs["num_drafts"] = args.num_drafts
            if args.no_draft_memory:
                aide_kwargs["no_draft_memory"] = True
            if args.best_node_strategy != "latest":
                aide_kwargs["best_node_strategy"] = args.best_node_strategy
            if args.memory_path is not None:
                aide_kwargs["memory_path"] = args.memory_path
            if args.memory_version != "v4":
                aide_kwargs["memory_version"] = args.memory_version
            if args.no_task_memory:
                aide_kwargs["no_task_memory"] = True
            if args.no_cross_memory:
                aide_kwargs["no_cross_memory"] = True
            if args.no_cross_memory_write:
                aide_kwargs["no_cross_memory_write"] = True
            if args.log_degradation:
                aide_kwargs["log_degradation"] = True
            if args.sticky_cross_memory:
                aide_kwargs["sticky_cross_memory"] = True
            if args.sticky_top_k != 15:
                aide_kwargs["sticky_top_k"] = args.sticky_top_k
            agent = AIDEAgent(
                backend=args.backend,
                model=args.model,
                submission_dir=str(DSGYM_ROOT / "submissions"),
                trajectory_output_dir=args.output_dir,
                **aide_kwargs,
                **agent_config
            )
        elif "dspredict" in args.dataset:
            agent = DSPredictReActAgent(
                backend=args.backend,
                model=args.model,
                submission_dir=str(DSGYM_ROOT / "submissions"),
                **agent_config
            )
        else:
            agent = ReActDSAgent(
            backend=args.backend,
            model=args.model,
            **agent_config
        )
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return 1
    
    # Load dataset
    print(f"📊 Loading {args.dataset} dataset...")
    try:
        dataset_config = {}
        if args.synthetic_path:
            dataset_config["synthetic_path"] = args.synthetic_path
            
        load_config = {
            "limit": args.limit
        }
        dataset_name = args.dataset
        if "dspredict" in dataset_name:
            # Map CLI name to split key: dspredict-mledojo -> mle_dojo
            _split_map = {"mledojo": "mle_dojo", "mledojo-remaining": "mle_dojo_remaining", "hard-swap": "hard_swap", "hard-rejected": "hard_rejected", "easy-claude-retry": "easy_claude_retry", "mle-bench": "mle_bench", "easy-train": "easy_train", "easy-test": "easy_test", "easy-test-retry": "easy_test_retry", "hard-train": "hard_train", "hard-test": "hard_test", "hard-test-variance2": "hard_test_variance2", "hard-test-variance3": "hard_test_variance3", "hard-train-gpt-remaining": "hard_train_gpt_remaining"}
            raw_split = dataset_name.split("-", 1)[-1]  # "easy", "hard", "swap", "mledojo"
            dataset_config["split"] = _split_map.get(raw_split, raw_split)
            dataset_name = dataset_name.split("-")[0]
            dataset_config["virtual_data_root"] = "/data"
            load_config["split"] = dataset_config["split"]
        dataset = DatasetRegistry.load(dataset_name, **dataset_config)
        samples = dataset.load(**load_config)
        print(f"✅ Loaded {len(samples)} samples from {args.dataset}")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Create evaluator with dataset-specific metrics
    print("⚖️ Setting up evaluator...")
    evaluator = Evaluator(
        protocol="multi_turn",
        dataset=dataset,  # This will automatically use dataset metrics and configs
        parallel_workers=args.max_workers
    )
    
    # Create evaluation config
    run_name = args.run_name or f"{args.dataset}_{args.backend}_{args.model.replace('/', '_')}"
    config = EvaluationConfig(
        model_name=args.model,
        backend_type=args.backend,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        run_name=run_name,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_workers=args.max_workers
    )
    
    # Run evaluation
    print("🏃 Starting evaluation...")
    try:
        results = evaluator.evaluate(
            agent=agent,
            tasks=samples,
            config=config,
            save_results=True
        )
        
        print("✅ Evaluation completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        
        # Print summary statistics
        if "metrics" in results:
            print("\n📈 Results Summary:")
            print("-" * 30)
            metrics = results["metrics"]
            
            # Show key metrics
            key_metrics = [
                "success_rate", "total_samples", "successful_samples", 
                "average_execution_time", "average_turns"
            ]
            
            for metric_name in key_metrics:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, float):
                        print(f"{metric_name}: {value:.3f}")
                    else:
                        print(f"{metric_name}: {value}")
            
            # Show dataset-specific metrics
            metric_scores = {}
            for key, value in metrics.items():
                if key.endswith("_mean") and not key.startswith("error"):
                    metric_name = key.replace("_mean", "")
                    metric_scores[metric_name] = value
            
            if metric_scores:
                print("\n📊 Dataset Metrics:")
                print("-" * 20)
                for metric_name, score in metric_scores.items():
                    print(f"{metric_name}: {score:.3f}")
        if "dspredict" in args.dataset:
            dataset.print_dspredict_results_overview(results["results"])
        
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # For standalone testing
    parser = argparse.ArgumentParser(description="DSGym evaluation")
    add_eval_parser(parser._subparsers_action.add_parser if hasattr(parser, '_subparsers_action') else parser.add_subparsers())
    args = parser.parse_args()
    sys.exit(run_eval(args))