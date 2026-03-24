#!/usr/bin/env python3
"""
DSGym Logit Distillation Integration

Bridges DSGym trajectory collection with the logit distillation trainer.

This script handles the DSGym-specific parts:
  1. Collect on-policy rollouts using DSGym TrajectoryGenerator
  2. Prepare trajectories with correct metadata (ground_truth, metrics, conversation)
  3. For SDPO: collect execution feedback annotations
  4. For OPSD: attach ground-truth answers for teacher conditioning
  5. Launch the logit distillation trainer

Usage:
    # Qwen3-style logit distillation
    python logit_distill_dsgym.py \
        --method qwen3 \
        --student-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --teacher-model Qwen/Qwen3-14B-Instruct \
        --dataset qrdata \
        --output-dir ./logit_qwen3_outputs

    # SDPO self-distillation
    python logit_distill_dsgym.py \
        --method sdpo \
        --student-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset qrdata \
        --output-dir ./logit_sdpo_outputs

    # OPSD self-distillation
    python logit_distill_dsgym.py \
        --method opsd \
        --student-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset qrdata \
        --output-dir ./logit_opsd_outputs
"""

import json
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# ============================================================================
# Trajectory Collection via DSGym
# ============================================================================


def collect_dsgym_trajectories(
    model: str,
    dataset: str,
    backend: str = "vllm",
    k: int = 4,
    temperature: float = 0.8,
    max_turns: int = 15,
    max_workers: int = 24,
    manager_url: str = "http://localhost:5000",
    output_dir: str = "./rollout",
    limit: Optional[int] = None,
    dataset_type: str = "original",
    run_name: str = "logit_distill",
) -> List[Dict]:
    """
    Collect trajectories using DSGym TrajectoryGenerator.

    Falls back to using student_rollout.py if DSGym is not directly importable.
    """
    try:
        return _collect_via_dsgym_api(
            model, dataset, backend, k, temperature, max_turns,
            max_workers, manager_url, output_dir, limit, dataset_type, run_name,
        )
    except ImportError:
        print("  DSGym not importable, falling back to student_rollout.py")
        return _collect_via_script(
            model, dataset, backend, k, temperature, max_turns,
            max_workers, manager_url, output_dir, limit, run_name,
        )


def _collect_via_dsgym_api(
    model, dataset, backend, k, temperature, max_turns,
    max_workers, manager_url, output_dir, limit, dataset_type, run_name,
) -> List[Dict]:
    """Direct DSGym API call for trajectory collection."""
    from dsgym.synth import TrajectoryConfig, TrajectoryGenerator

    config = TrajectoryConfig(
        model=model,
        backend=backend,
        temperature=temperature,
        k=k,
        max_workers=max_workers,
        max_turns=max_turns,
        manager_url=manager_url,
        dataset_name=dataset,
        output_dir=output_dir,
    )

    generator = TrajectoryGenerator(config)
    results = generator.generate()

    # Convert to trajectory dicts
    trajectories = []
    for result in results:
        traj = {
            "conversation": result.get("conversation", []),
            "prediction": result.get("prediction", ""),
            "ground_truth": result.get("ground_truth", ""),
            "metrics": result.get("metrics", {}),
            "query_id": result.get("query_id", ""),
            "dataset": dataset,
        }
        trajectories.append(traj)

    return trajectories


def _collect_via_script(
    model, dataset, backend, k, temperature, max_turns,
    max_workers, manager_url, output_dir, limit, run_name,
) -> List[Dict]:
    """Use student_rollout.py subprocess for trajectory collection."""
    script_path = Path(__file__).parent / "student_rollout.py"
    if not script_path.exists():
        raise FileNotFoundError(f"student_rollout.py not found at {script_path}")

    cmd = [
        sys.executable, str(script_path),
        "--model", model,
        "--backend", backend,
        "--dataset", dataset,
        "--k", str(k),
        "--temperature", str(temperature),
        "--max-turns", str(max_turns),
        "--max-workers", str(max_workers),
        "--manager-url", manager_url,
        "--output-dir", output_dir,
        "--run-name", run_name,
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    return _load_trajectories_from_dir(output_dir)


def _load_trajectories_from_dir(rollout_dir: str) -> List[Dict]:
    """Load trajectory JSON files from rollout output directory."""
    trajs = []
    d = Path(rollout_dir)

    # Try multiple subdirectory patterns
    for subdir in ["predictions", "correct", "incorrect", ""]:
        search_dir = d / subdir if subdir else d
        if not search_dir.exists():
            continue
        for fp in sorted(search_dir.glob("*.json")):
            if fp.name.startswith(("metrics", "config", "rollout", "failed", "succeeded")):
                continue
            try:
                with open(fp) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        trajs.extend(data)
                    else:
                        trajs.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
        if trajs:
            break

    # Also try loading from a single combined file
    combined_file = d / "trajectories.json"
    if not trajs and combined_file.exists():
        with open(combined_file) as f:
            data = json.load(f)
            trajs = data if isinstance(data, list) else [data]

    return trajs


# ============================================================================
# Trajectory Enrichment
# ============================================================================

def enrich_trajectories_for_sdpo(trajectories: List[Dict]) -> List[Dict]:
    """
    Enrich trajectories with execution feedback annotations for SDPO.

    SDPO requires:
      - Student input: standard conversation (no feedback)
      - Teacher input: conversation with execution feedback in context

    The execution feedback is already captured in the conversation
    (as user messages containing stdout/stderr). We annotate each
    trajectory with a feedback summary that will be injected into
    the teacher's system prompt.
    """
    for traj in trajectories:
        conv = traj.get("conversation", [])
        feedback = _extract_execution_feedback(conv)
        traj["execution_feedback"] = feedback
        traj["has_errors"] = any("error" in f.lower() for f in feedback)
        traj["n_execution_steps"] = len(feedback)

    return trajectories


def enrich_trajectories_for_opsd(trajectories: List[Dict]) -> List[Dict]:
    """
    Enrich trajectories with ground-truth answer for OPSD.

    OPSD requires ground_truth to condition the teacher: p_T(·|x, y*).
    Filter out trajectories without ground truth.
    """
    enriched = []
    for traj in trajectories:
        gt = traj.get("ground_truth", "")
        if not gt:
            # Try to extract from metrics
            metrics = traj.get("metrics", {})
            for metric_name in ["exact_match", "fuzzy_exact_match"]:
                m = metrics.get(metric_name, {})
                if isinstance(m, dict) and "expected" in m:
                    gt = str(m["expected"])
                    break

        if gt:
            traj["ground_truth"] = gt
            enriched.append(traj)

    if len(enriched) < len(trajectories):
        print(f"  OPSD: filtered {len(trajectories) - len(enriched)} trajectories "
              f"without ground truth ({len(enriched)} remaining)")

    return enriched


def _extract_execution_feedback(conversation: List[Dict]) -> List[str]:
    """Extract execution outputs from conversation user messages."""
    feedback = []
    for i, msg in enumerate(conversation):
        if msg.get("role") != "user" or i == 0:
            continue
        content = msg.get("content", "")
        # Detect execution output patterns
        if any(pattern in content for pattern in [
            "[Code execution output]", "Output:", "Traceback",
            "Error:", "Exception:", ">>> ", "stderr:",
        ]):
            # Truncate long outputs
            feedback.append(content[:500] if len(content) > 500 else content)
    return feedback


# ============================================================================
# Statistics & Reporting
# ============================================================================

def compute_rollout_stats(trajectories: List[Dict]) -> Dict[str, Any]:
    """Compute statistics on collected trajectories."""
    stats = {
        "total": len(trajectories),
        "correct": 0,
        "incorrect": 0,
        "avg_turns": 0,
        "avg_length": 0,
        "datasets": defaultdict(int),
    }

    total_turns = 0
    total_length = 0

    for traj in trajectories:
        # Check correctness
        metrics = traj.get("metrics", {})
        is_correct = False
        for m_name in ["exact_match", "fuzzy_exact_match"]:
            m = metrics.get(m_name, {})
            score = m.get("score", 0) if isinstance(m, dict) else float(m) if m else 0
            if score >= 0.99:
                is_correct = True
                break

        if is_correct:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1

        # Conversation stats
        conv = traj.get("conversation", [])
        n_turns = sum(1 for m in conv if m.get("role") == "assistant")
        total_turns += n_turns
        total_length += sum(len(m.get("content", "")) for m in conv)

        stats["datasets"][traj.get("dataset", "unknown")] += 1

    n = max(len(trajectories), 1)
    stats["avg_turns"] = total_turns / n
    stats["avg_length"] = total_length / n
    stats["pass_rate"] = stats["correct"] / n
    stats["datasets"] = dict(stats["datasets"])

    return stats


def print_stats(stats: Dict, label: str = ""):
    """Pretty-print rollout statistics."""
    print(f"\n  {'─' * 50}")
    print(f"  Rollout Statistics {label}")
    print(f"  {'─' * 50}")
    print(f"  Total trajectories: {stats['total']}")
    print(f"  Correct:            {stats['correct']} ({stats['pass_rate']:.1%})")
    print(f"  Incorrect:          {stats['incorrect']}")
    print(f"  Avg turns:          {stats['avg_turns']:.1f}")
    print(f"  Avg length (chars): {stats['avg_length']:.0f}")
    if stats.get("datasets"):
        print(f"  Datasets: {stats['datasets']}")
    print(f"  {'─' * 50}\n")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_logit_distillation_pipeline(args):
    """
    Full pipeline: collect rollouts → enrich → train with logit distillation.

    Each iteration:
      1. Collect on-policy rollouts from DSGym
      2. Enrich trajectories (SDPO feedback / OPSD ground truth)
      3. Save enriched trajectories
      4. Launch logit distillation trainer
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # Save pipeline config
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(args.output_dir, "pipeline_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  LOGIT DISTILLATION PIPELINE")
    print(f"  Method:  {args.method.upper()}")
    print(f"  Student: {args.student_model}")
    print(f"  Teacher: {args.teacher_model or '(self-distillation)'}")
    print(f"  Dataset: {args.dataset}")
    print(f"{'=' * 70}\n")

    all_results = []

    for iteration in range(args.num_iterations):
        print(f"\n{'#' * 70}")
        print(f"#  PIPELINE ITERATION {iteration}/{args.num_iterations}")
        print(f"{'#' * 70}\n")

        iter_dir = os.path.join(args.output_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        # Determine current student model
        if iteration == 0:
            current_model = args.student_model
        else:
            prev_ckpt = os.path.join(
                args.output_dir, f"iter_{iteration-1}", "checkpoint",
            )
            current_model = prev_ckpt if os.path.isdir(prev_ckpt) else args.student_model

        # ── Step 1: Collect on-policy rollouts ──
        print(f"\n  Step 1: Collecting on-policy rollouts with {current_model}")
        t0 = time.time()
        rollout_dir = os.path.join(iter_dir, "rollout")

        trajectories = collect_dsgym_trajectories(
            model=current_model,
            dataset=args.dataset,
            backend=args.student_backend,
            k=args.k,
            temperature=args.temperature,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            manager_url=args.manager_url,
            output_dir=rollout_dir,
            limit=args.dataset_limit,
            run_name=f"logit_distill_iter{iteration}",
        )
        rollout_time = time.time() - t0

        if not trajectories:
            print("  ⚠️  No trajectories collected; skipping iteration.")
            continue

        stats = compute_rollout_stats(trajectories)
        print_stats(stats, f"(iter {iteration})")

        # ── Step 2: Enrich trajectories ──
        print(f"  Step 2: Enriching trajectories for {args.method}")

        if args.method == "sdpo":
            trajectories = enrich_trajectories_for_sdpo(trajectories)
        elif args.method == "opsd":
            trajectories = enrich_trajectories_for_opsd(trajectories)

        # Save enriched trajectories
        traj_path = os.path.join(iter_dir, "trajectories.json")
        with open(traj_path, "w") as f:
            json.dump(trajectories, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(trajectories)} trajectories → {traj_path}")

        # ── Step 3: Launch logit distillation trainer ──
        print(f"\n  Step 3: Running logit distillation (method={args.method})")

        trainer_cmd = [
            sys.executable,
            str(Path(__file__).parent / "logit_distillation.py"),
            "--method", args.method,
            "--student-model", current_model,
            "--num-iterations", "1",  # Trainer handles 1 iteration; we loop externally
            "--inner-epochs", str(args.inner_epochs),
            "--batch-size", str(args.batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--learning-rate", str(args.learning_rate),
            "--max-seq-length", str(args.max_seq_length),
            "--distill-temperature", str(args.distill_temperature),
            "--alpha-distill", str(args.alpha_distill),
            "--alpha-sft", str(args.alpha_sft),
            "--loss-type", args.loss_type,
            "--jsd-beta", str(args.jsd_beta),
            "--k", str(args.k),
            "--temperature", str(args.temperature),
            "--max-turns", str(args.max_turns),
            "--max-workers", str(args.max_workers),
            "--manager-url", args.manager_url,
            "--dataset", args.dataset,
            "--output-dir", iter_dir,
        ]

        if args.teacher_model:
            trainer_cmd += ["--teacher-model", args.teacher_model]
        if args.use_lora:
            trainer_cmd.append("--use-lora")
            trainer_cmd += ["--lora-rank", str(args.lora_rank)]
        if args.reward_weighted:
            trainer_cmd.append("--reward-weighted")
        if args.use_wandb:
            trainer_cmd.append("--use-wandb")
            trainer_cmd += ["--wandb-project", args.wandb_project]
            trainer_cmd += ["--run-name", f"{args.method}_iter{iteration}"]
        if args.use_deepspeed:
            trainer_cmd.append("--use-deepspeed")

        print(f"  Running: {' '.join(trainer_cmd[:8])} ...")

        t1 = time.time()
        result = subprocess.run(trainer_cmd, capture_output=False)
        train_time = time.time() - t1

        if result.returncode != 0:
            print(f"  ⚠️  Trainer exited with code {result.returncode}")

        # ── Step 4: Record results ──
        iter_result = {
            "iteration": iteration,
            "model": current_model,
            "n_trajectories": len(trajectories),
            "stats": stats,
            "rollout_time_sec": rollout_time,
            "train_time_sec": train_time,
        }
        all_results.append(iter_result)

        # Save progress
        with open(os.path.join(args.output_dir, "pipeline_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # ── Final Summary ──
    print(f"\n\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — {args.method.upper()}")
    print(f"{'=' * 70}")

    for r in all_results:
        s = r["stats"]
        print(
            f"  iter {r['iteration']}: "
            f"{s['total']} trajs, "
            f"pass={s['pass_rate']:.3f}, "
            f"rollout={r['rollout_time_sec']:.0f}s, "
            f"train={r['train_time_sec']:.0f}s"
        )

    print(f"\n  Results: {args.output_dir}/pipeline_results.json")
    print(f"{'=' * 70}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="DSGym Logit Distillation Integration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Method
    p.add_argument("--method", required=True, choices=["qwen3", "sdpo", "opsd"])

    # Models
    p.add_argument("--student-model", required=True)
    p.add_argument("--teacher-model", default="")
    p.add_argument("--student-backend", default="vllm",
                   choices=["vllm", "multi-vllm", "sglang"])

    # Dataset
    p.add_argument("--dataset", default="qrdata")
    p.add_argument("--dataset-limit", type=int, default=None)

    # Rollout
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--max-workers", type=int, default=24)
    p.add_argument("--manager-url", default="http://localhost:5000")

    # Training
    p.add_argument("--num-iterations", type=int, default=3)
    p.add_argument("--inner-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-seq-length", type=int, default=8192)

    # Distillation
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--alpha-distill", type=float, default=0.8)
    p.add_argument("--alpha-sft", type=float, default=0.2)
    p.add_argument("--loss-type", default="kl",
                   choices=["kl", "reverse_kl", "jsd", "full_vocab"])
    p.add_argument("--jsd-beta", type=float, default=0.5)
    p.add_argument("--reward-weighted", action="store_true")

    # LoRA
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.add_argument("--lora-rank", type=int, default=64)

    # Infrastructure
    p.add_argument("--use-deepspeed", action="store_true")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", default="dsgym-logit-distill")

    # Output
    p.add_argument("--output-dir", default="./logit_distill_outputs")

    args = p.parse_args()

    # Auto-set method defaults
    if args.method == "sdpo" and args.loss_type == "kl":
        args.loss_type = "jsd"
    if args.method == "opsd" and args.loss_type == "kl":
        args.loss_type = "full_vocab"

    run_logit_distillation_pipeline(args)


if __name__ == "__main__":
    main()
