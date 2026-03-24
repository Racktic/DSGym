#!/usr/bin/env python3
"""
Iterative Rejection Sampling Fine-Tuning (RFT) Orchestrator

This is the **core** Route B script. It manages the full iterative loop:

    for iteration k = 0, 1, 2, …:
        1. Student_k rollout on DSGym  →  trajectories
        2. Evaluate & separate correct / incorrect
        3. Teacher backfill for failed queries (optional, ratio decays)
        4. Merge correct student + teacher trajectories → SFT dataset
        5. (Optional) Build DPO pairs from correct vs incorrect
        6. Train Student_{k+1} via LLaMA-Factory SFT (+ DPO)
        7. Evaluate on held-out set → decide whether to continue

Implements three RFT strategies:
  - vanilla:     Standard rejection sampling (keep correct, discard incorrect)
  - star:        STaR (Self-Taught Reasoner) — retry incorrect with hint, then SFT
  - rest:        ReST (Reinforced Self-Training) — reward-weighted sampling

Usage:
    python iterative_rft.py \
        --student-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --teacher-model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
        --dataset qrdata \
        --iterations 5 \
        --strategy vanilla \
        --output-root ./route_b_outputs

    python iterative_rft.py \
        --student-model /data/fnie/saves/student_iter0 \
        --dataset qrdata \
        --iterations 3 \
        --strategy star \
        --teacher-backfill-ratio 0.3 \
        --enable-dpo \
        --output-root ./route_b_outputs
"""

import json
import os
import sys
import time
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RFTConfig:
    """Configuration for iterative RFT."""
    # Models
    student_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    teacher_model: str = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
    student_backend: str = "vllm"
    teacher_backend: str = "litellm"

    # Dataset
    dataset: str = "qrdata"
    dataset_type: str = "original"
    synthetic_path: Optional[str] = None
    eval_limit: Optional[int] = None  # Held-out eval subset size

    # Iteration
    iterations: int = 5
    strategy: str = "vanilla"  # vanilla | star | rest

    # Rollout
    k: int = 8  # trajectories per query
    temperature: float = 0.8
    max_turns: int = 15
    max_workers: int = 24

    # Teacher backfill
    teacher_backfill_ratio: float = 1.0   # Initial ratio; decays over iterations
    teacher_backfill_decay: float = 0.5   # Multiply ratio each iteration
    teacher_k: int = 3                    # K for teacher backfill rollout

    # Training
    train_epochs: int = 2
    learning_rate: str = "1e-5"
    batch_size: int = 1
    gradient_accum: int = 16
    finetuning_type: str = "lora"
    lora_rank: int = 64

    # DPO
    enable_dpo: bool = False
    dpo_beta: float = 0.1

    # Paths
    output_root: str = "./route_b_outputs"
    llama_factory_root: str = "/data/fnie/LLaMA-Factory"
    manager_url: str = "http://localhost:5000"

    # Early stopping
    patience: int = 2          # Stop if no improvement for N iterations
    min_improvement: float = 0.01  # Minimum pass@1 improvement to count

    # STaR-specific
    star_hint_template: str = (
        "The correct answer is: {ground_truth}\n"
        "Please re-attempt the analysis step by step to arrive at this answer."
    )

    # ReST-specific
    rest_reward_temperature: float = 1.0  # Temperature for reward-weighted sampling


@dataclass
class IterationResult:
    """Result of a single RFT iteration."""
    iteration: int
    student_model_path: str
    rollout_pass_at_1: float = 0.0
    rollout_pass_at_k: float = 0.0
    student_correct: int = 0
    teacher_backfill: int = 0
    total_sft_samples: int = 0
    total_dpo_pairs: int = 0
    eval_accuracy: float = 0.0
    train_time_sec: float = 0.0
    rollout_time_sec: float = 0.0


# ============================================================================
# Iteration Steps
# ============================================================================

SCRIPT_DIR = Path(__file__).parent


def run_command(cmd: List[str], description: str, cwd: Optional[str] = None):
    """Run a command and stream output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd[:6])} ...")
    print(f"{'='*60}\n")

    result = subprocess.run(
        cmd, cwd=cwd,
        capture_output=False,  # Stream to console
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd[:6])}"
        )
    return result


def step_student_rollout(config: RFTConfig, iteration: int, model_path: str) -> str:
    """
    Step 1: Student rollout. Returns the output directory.
    """
    out_dir = os.path.join(config.output_root, f"iter_{iteration}", "student_rollout")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "student_rollout.py"),
        "--model", model_path,
        "--backend", config.student_backend,
        "--dataset", config.dataset,
        "--k", str(config.k),
        "--temperature", str(config.temperature),
        "--max-turns", str(config.max_turns),
        "--max-workers", str(config.max_workers),
        "--manager-url", config.manager_url,
        "--output-dir", out_dir,
        "--run-name", f"iter{iteration}_student",
    ]
    if config.dataset_type == "synthetic" and config.synthetic_path:
        cmd += ["--dataset-type", "synthetic", "--synthetic-path", config.synthetic_path]

    run_command(cmd, f"Iteration {iteration} — Student Rollout")
    return out_dir


def step_teacher_backfill(
    config: RFTConfig, iteration: int, failed_queries_path: str
) -> Optional[str]:
    """
    Step 2: Teacher backfill for failed queries. Returns output dir or None.
    """
    # Compute decayed backfill ratio
    ratio = config.teacher_backfill_ratio * (config.teacher_backfill_decay ** iteration)
    if ratio < 0.05:
        print(f"  ⏭️  Teacher backfill ratio {ratio:.3f} < 0.05; skipping.")
        return None

    # Check if there are failed queries
    with open(failed_queries_path, "r") as f:
        data = json.load(f)
    failed_ids = data.get("failed_sample_ids", [])
    if not failed_ids:
        print("  ✅ No failed queries — teacher backfill not needed.")
        return None

    # Subsample failed queries by ratio
    import random
    random.seed(42 + iteration)
    n_backfill = max(1, int(len(failed_ids) * ratio))
    sampled_ids = random.sample(failed_ids, min(n_backfill, len(failed_ids)))

    # Write subsampled list
    iter_dir = os.path.join(config.output_root, f"iter_{iteration}")
    sampled_path = os.path.join(iter_dir, "backfill_sample_ids.json")
    with open(sampled_path, "w") as f:
        json.dump({"failed_sample_ids": sampled_ids}, f)

    out_dir = os.path.join(iter_dir, "teacher_backfill")
    cmd = [
        sys.executable, str(SCRIPT_DIR / "student_rollout.py"),
        "--model", config.teacher_model,
        "--backend", config.teacher_backend,
        "--dataset", config.dataset,
        "--k", str(config.teacher_k),
        "--temperature", str(0.7),
        "--max-turns", str(config.max_turns),
        "--max-workers", str(config.max_workers),
        "--manager-url", config.manager_url,
        "--output-dir", out_dir,
        "--run-name", f"iter{iteration}_teacher_backfill",
        "--failed-queries", sampled_path,
    ]

    run_command(cmd, f"Iteration {iteration} — Teacher Backfill ({len(sampled_ids)} queries)")
    return out_dir


def step_star_retry_with_hint(
    config: RFTConfig, iteration: int, student_dir: str, model_path: str
) -> Optional[str]:
    """
    STaR-specific: Re-attempt failed queries with the correct answer as hint.
    The model sees the ground truth and must generate a valid reasoning trace.
    """
    failed_path = os.path.join(student_dir, "failed_queries.json")
    with open(failed_path, "r") as f:
        failed_ids = json.load(f).get("failed_sample_ids", [])
    if not failed_ids:
        return None

    print(f"  ⭐ STaR: Re-attempting {len(failed_ids)} failed queries with hints …")

    # Load dataset to get ground truths
    from dsgym.datasets import DatasetRegistry
    dataset = DatasetRegistry.load(config.dataset)
    all_samples = dataset.load()

    # Create hint-augmented samples
    hint_dir = os.path.join(config.output_root, f"iter_{iteration}", "star_hints")
    os.makedirs(hint_dir, exist_ok=True)

    hint_samples = []
    failed_set = set(failed_ids)
    for sample in all_samples:
        sid = sample.get("sample_id", sample.get("id", ""))
        if sid in failed_set:
            gt = sample.get("ground_truth", sample.get("reward_spec", {}).get("ground_truth", ""))
            if gt:
                # Append hint to the user prompt
                augmented = dict(sample)
                prompt = augmented.get("prompt", [])
                if prompt:
                    hint_msg = config.star_hint_template.format(ground_truth=gt)
                    # Add hint as extra context in the last user message
                    last_user_idx = None
                    for i, msg in enumerate(prompt):
                        if msg.get("role") == "user":
                            last_user_idx = i
                    if last_user_idx is not None:
                        prompt[last_user_idx]["content"] += f"\n\n[HINT] {hint_msg}"
                    augmented["prompt"] = prompt
                hint_samples.append(augmented)

    if not hint_samples:
        return None

    # Save hint-augmented samples and run rollout
    hint_samples_path = os.path.join(hint_dir, "hint_samples.json")
    with open(hint_samples_path, "w") as f:
        json.dump([{
            "sample_id": s.get("sample_id", s.get("id", "")),
            "ground_truth": s.get("ground_truth", "")
        } for s in hint_samples], f, indent=2)

    # Use TrajectoryGenerator directly with the hint samples
    from dsgym.synth.generators.trajectory_generator import TrajectoryGenerator, TrajectoryConfig
    traj_config = TrajectoryConfig(
        model=model_path,
        backend=config.student_backend,
        temperature=config.temperature,
        k=2,  # Fewer attempts since we have hint
        max_workers=config.max_workers,
        max_turns=config.max_turns,
        manager_url=config.manager_url,
        dataset_name=config.dataset,
        compute_metrics=True,
        output_dir=hint_dir,
        run_name=f"iter{iteration}_star_hints",
    )
    generator = TrajectoryGenerator(traj_config)
    generator._initialize_components()
    generator.generate(hint_samples)

    return hint_dir


def step_merge_and_convert(
    config: RFTConfig,
    iteration: int,
    student_dir: str,
    teacher_dir: Optional[str],
    star_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 3: Merge correct student trajectories + teacher backfill → SFT data.
    Optionally build DPO pairs.

    Returns dict with counts and file paths.
    """
    iter_dir = os.path.join(config.output_root, f"iter_{iteration}")
    sft_out = os.path.join(iter_dir, "sft_data.json")
    dpo_out = os.path.join(iter_dir, "dpo_data.json")

    # Collect correct trajectories from student
    correct_trajs = _load_trajectories_from_dir(os.path.join(student_dir, "correct"))

    # Collect incorrect trajectories (for DPO)
    incorrect_trajs = _load_trajectories_from_dir(os.path.join(student_dir, "incorrect"))

    # Collect teacher backfill correct trajectories
    teacher_correct = []
    if teacher_dir:
        teacher_correct_dir = os.path.join(teacher_dir, "correct")
        if os.path.isdir(teacher_correct_dir):
            teacher_correct = _load_trajectories_from_dir(teacher_correct_dir)
        else:
            # If no classify was done, load all and filter
            all_teacher = _load_trajectories_from_dir(teacher_dir)
            from scripts.student_rollout import evaluate_trajectory_correctness
            teacher_correct = [t for t in all_teacher
                               if evaluate_trajectory_correctness(t)[0]]

    # STaR hint trajectories (only correct ones)
    star_correct = []
    if star_dir:
        star_correct_dir = os.path.join(star_dir, "correct")
        if os.path.isdir(star_correct_dir):
            star_correct = _load_trajectories_from_dir(star_correct_dir)

    print(f"  Correct student trajectories:  {len(correct_trajs)}")
    print(f"  Teacher backfill trajectories: {len(teacher_correct)}")
    print(f"  STaR hint trajectories:        {len(star_correct)}")
    print(f"  Incorrect trajectories (DPO):  {len(incorrect_trajs)}")

    # === Strategy-specific processing ===
    if config.strategy == "rest":
        # ReST: Reward-weighted sampling instead of binary keep/discard
        all_trajs = correct_trajs + incorrect_trajs
        sft_samples = _rest_reward_weighted_convert(all_trajs, config.rest_reward_temperature)
    else:
        # Vanilla / STaR: keep correct, discard incorrect
        all_correct = correct_trajs + teacher_correct + star_correct

        # Select best trajectory per sample (shortest correct)
        sft_samples = _select_best_per_sample_and_convert(all_correct)

    # Save SFT data
    with open(sft_out, "w") as f:
        json.dump(sft_samples, f, indent=2, ensure_ascii=False)
    print(f"  SFT data: {len(sft_samples)} samples → {sft_out}")

    # === DPO pairs ===
    dpo_pairs = []
    if config.enable_dpo and incorrect_trajs:
        dpo_pairs = _build_dpo_pairs(correct_trajs, incorrect_trajs)
        with open(dpo_out, "w") as f:
            json.dump(dpo_pairs, f, indent=2, ensure_ascii=False)
        print(f"  DPO data: {len(dpo_pairs)} pairs → {dpo_out}")

    return {
        "sft_path": sft_out,
        "dpo_path": dpo_out if dpo_pairs else None,
        "sft_count": len(sft_samples),
        "dpo_count": len(dpo_pairs),
        "student_correct": len(correct_trajs),
        "teacher_backfill": len(teacher_correct),
    }


def step_train_sft(config: RFTConfig, iteration: int, sft_data_path: str) -> str:
    """
    Step 4: SFT training via LLaMA-Factory. Returns path to new model.
    """
    iter_dir = os.path.join(config.output_root, f"iter_{iteration}")
    save_dir = os.path.join(iter_dir, "student_checkpoint")

    # Determine base model: iter 0 → original, iter N → previous checkpoint
    if iteration == 0:
        base_model = config.student_model
    else:
        prev_ckpt = os.path.join(
            config.output_root, f"iter_{iteration - 1}", "student_checkpoint"
        )
        if os.path.isdir(prev_ckpt):
            base_model = prev_ckpt
        else:
            base_model = config.student_model

    # Copy SFT data to LLaMA-Factory data dir
    lf_data_dir = os.path.join(config.llama_factory_root, "data")
    dataset_name = f"route_b_iter{iteration}_sft"
    lf_data_file = os.path.join(lf_data_dir, f"{dataset_name}.json")
    shutil.copy2(sft_data_path, lf_data_file)

    # Update dataset_info.json
    _update_llama_factory_dataset_info(lf_data_dir, dataset_name, lf_data_file)

    # Build training command
    cmd = [
        "llamafactory-cli", "train",
        "--model_name_or_path", base_model,
        "--stage", "sft",
        "--do_train", "true",
        "--dataset", dataset_name,
        "--template", "qwen",
        "--finetuning_type", config.finetuning_type,
        "--output_dir", save_dir,
        "--overwrite_output_dir", "true",
        "--per_device_train_batch_size", str(config.batch_size),
        "--gradient_accumulation_steps", str(config.gradient_accum),
        "--num_train_epochs", str(config.train_epochs),
        "--learning_rate", config.learning_rate,
        "--logging_steps", "10",
        "--save_strategy", "epoch",
        "--report_to", "wandb",
        "--run_name", f"route_b_sft_iter{iteration}",
        "--bf16", "true",
        "--warmup_ratio", "0.05",
    ]

    if config.finetuning_type == "lora":
        cmd += [
            "--lora_rank", str(config.lora_rank),
            "--lora_alpha", str(config.lora_rank * 2),
            "--lora_target", "all",
        ]

    run_command(cmd, f"Iteration {iteration} — SFT Training", cwd=config.llama_factory_root)
    return save_dir


def step_train_dpo(config: RFTConfig, iteration: int, dpo_data_path: str) -> str:
    """
    Step 4b: DPO training (optional). Returns updated model path.
    """
    iter_dir = os.path.join(config.output_root, f"iter_{iteration}")
    sft_ckpt = os.path.join(iter_dir, "student_checkpoint")
    dpo_save_dir = os.path.join(iter_dir, "student_checkpoint_dpo")

    lf_data_dir = os.path.join(config.llama_factory_root, "data")
    dataset_name = f"route_b_iter{iteration}_dpo"
    lf_data_file = os.path.join(lf_data_dir, f"{dataset_name}.json")
    shutil.copy2(dpo_data_path, lf_data_file)
    _update_llama_factory_dataset_info(lf_data_dir, dataset_name, lf_data_file,
                                       ranking=True)

    cmd = [
        "llamafactory-cli", "train",
        "--model_name_or_path", sft_ckpt,
        "--stage", "dpo",
        "--do_train", "true",
        "--dataset", dataset_name,
        "--template", "qwen",
        "--finetuning_type", config.finetuning_type,
        "--output_dir", dpo_save_dir,
        "--overwrite_output_dir", "true",
        "--per_device_train_batch_size", str(config.batch_size),
        "--gradient_accumulation_steps", str(config.gradient_accum),
        "--num_train_epochs", "1",
        "--learning_rate", "5e-6",
        "--logging_steps", "10",
        "--save_strategy", "epoch",
        "--report_to", "wandb",
        "--run_name", f"route_b_dpo_iter{iteration}",
        "--bf16", "true",
        "--dpo_beta", str(config.dpo_beta),
    ]

    if config.finetuning_type == "lora":
        cmd += [
            "--lora_rank", str(config.lora_rank),
            "--lora_alpha", str(config.lora_rank * 2),
            "--lora_target", "all",
        ]

    run_command(cmd, f"Iteration {iteration} — DPO Training", cwd=config.llama_factory_root)
    return dpo_save_dir


def step_evaluate(config: RFTConfig, iteration: int, model_path: str) -> float:
    """
    Step 5: Evaluate on held-out eval set. Returns accuracy.
    """
    eval_dir = os.path.join(config.output_root, f"iter_{iteration}", "eval_results")
    cmd = [
        sys.executable, str(Path(__file__).parent.parent / "examples" / "evaluate.py"),
        "--dataset", config.dataset,
        "--model", model_path,
        "--backend", config.student_backend,
        "--output-dir", eval_dir,
        "--max-turns", str(config.max_turns),
        "--manager-url", config.manager_url,
        "--temperature", "0.0",  # greedy for eval
    ]
    if config.eval_limit:
        cmd += ["--limit", str(config.eval_limit)]

    try:
        run_command(cmd, f"Iteration {iteration} — Held-out Evaluation")
    except RuntimeError:
        print("  ⚠️  Evaluation failed; returning 0.0")
        return 0.0

    # Read metrics from output
    metrics_file = os.path.join(eval_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        # Try to extract accuracy
        for key in ["fuzzy_exact_match", "exact_match", "accuracy"]:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, dict):
                    return val.get("mean_score", 0.0)
                return float(val)
    return 0.0


# ============================================================================
# Helper Functions
# ============================================================================

def _load_trajectories_from_dir(dir_path: str) -> List[Dict]:
    """Load all JSON trajectory files from a directory."""
    trajs = []
    d = Path(dir_path)
    if not d.exists():
        return trajs
    for fp in sorted(d.glob("*.json")):
        if fp.name.startswith("metrics") or fp.name.startswith("config") or \
           fp.name.startswith("rollout") or fp.name.startswith("failed") or \
           fp.name.startswith("succeeded"):
            continue
        try:
            with open(fp, "r") as f:
                trajs.append(json.load(f))
        except (json.JSONDecodeError, KeyError):
            continue
    return trajs


def _select_best_per_sample_and_convert(trajectories: List[Dict]) -> List[Dict]:
    """
    Group trajectories by sample_id, pick the best one per sample
    (shortest correct trajectory), and convert to ShareGPT format.
    """
    by_sample: Dict[str, List[Dict]] = defaultdict(list)
    for t in trajectories:
        sid = t.get("sample_id", "unknown")
        by_sample[sid].append(t)

    sft_data = []
    for sid, trajs in by_sample.items():
        # Pick shortest correct trajectory
        trajs.sort(key=lambda t: t.get("turns", t.get("total_turns", 999)))
        best = trajs[0]
        converted = _trajectory_to_sharegpt(best)
        if converted:
            sft_data.append(converted)

    return sft_data


def _rest_reward_weighted_convert(
    trajectories: List[Dict], temperature: float = 1.0
) -> List[Dict]:
    """
    ReST: Reward-weighted sampling. Higher reward → higher weight → more
    likely to be sampled. Uses softmax over rewards to produce a distribution,
    then samples proportionally.
    """
    import math
    import random
    random.seed(42)

    # Compute rewards
    rewarded = []
    for t in trajectories:
        from scripts.student_rollout import evaluate_trajectory_correctness
        is_correct, score = evaluate_trajectory_correctness(t)
        reward = score  # 0.0 to 1.0
        # Bonus for correct answer
        if is_correct:
            reward += 0.5
        # Penalty for very long trajectories
        turns = t.get("turns", t.get("total_turns", 0))
        if turns > 12:
            reward *= 0.8
        rewarded.append((t, reward))

    if not rewarded:
        return []

    # Softmax over rewards/temperature
    max_r = max(r for _, r in rewarded)
    exp_rewards = [math.exp((r - max_r) / max(temperature, 0.01)) for _, r in rewarded]
    total = sum(exp_rewards)
    probs = [e / total for e in exp_rewards]

    # Sample with replacement, proportional to reward
    n_samples = min(len(rewarded), len(set(t.get("sample_id", i) for i, (t, _) in enumerate(rewarded))))
    indices = list(range(len(rewarded)))
    sampled_indices = random.choices(indices, weights=probs, k=n_samples)

    # Deduplicate by sample_id (keep highest reward version)
    seen = {}
    for idx in sampled_indices:
        t, r = rewarded[idx]
        sid = t.get("sample_id", "unknown")
        if sid not in seen or r > seen[sid][1]:
            seen[sid] = (t, r)

    sft_data = []
    for sid, (t, r) in seen.items():
        converted = _trajectory_to_sharegpt(t)
        if converted:
            sft_data.append(converted)

    return sft_data


def _trajectory_to_sharegpt(traj: Dict) -> Optional[Dict]:
    """Convert a single trajectory to ShareGPT format."""
    conversation = traj.get("conversation", [])
    if not conversation:
        return None

    messages = []
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue

        if role == "system":
            messages.append({"from": "system", "value": content})
        elif role == "assistant":
            messages.append({"from": "gpt", "value": content})
        elif role == "user":
            messages.append({"from": "human", "value": content})

    if len(messages) < 2:
        return None

    return {
        "conversations": messages,
        "system": next(
            (m["value"] for m in messages if m["from"] == "system"), ""
        ),
        "dataset": traj.get("dataset_name", ""),
        "sample_id": traj.get("sample_id", ""),
    }


def _build_dpo_pairs(
    correct_trajs: List[Dict], incorrect_trajs: List[Dict]
) -> List[Dict]:
    """
    Build DPO preference pairs: chosen=correct, rejected=incorrect
    for the same sample_id.
    """
    # Group by sample_id
    correct_by_id: Dict[str, List[Dict]] = defaultdict(list)
    incorrect_by_id: Dict[str, List[Dict]] = defaultdict(list)

    for t in correct_trajs:
        sid = t.get("sample_id", "unknown")
        correct_by_id[sid].append(t)
    for t in incorrect_trajs:
        sid = t.get("sample_id", "unknown")
        incorrect_by_id[sid].append(t)

    pairs = []
    for sid in correct_by_id:
        if sid not in incorrect_by_id:
            continue
        # Pick best correct and worst incorrect
        chosen = min(correct_by_id[sid], key=lambda t: t.get("turns", 999))
        rejected = incorrect_by_id[sid][0]

        chosen_conv = _trajectory_to_sharegpt(chosen)
        rejected_conv = _trajectory_to_sharegpt(rejected)
        if not chosen_conv or not rejected_conv:
            continue

        pairs.append({
            "conversations": chosen_conv["conversations"][:1],  # Shared prompt
            "chosen": chosen_conv["conversations"][1:],
            "rejected": rejected_conv["conversations"][1:],
            "sample_id": sid,
        })

    return pairs


def _update_llama_factory_dataset_info(
    data_dir: str, dataset_name: str, data_file: str, ranking: bool = False
):
    """Update LLaMA-Factory's dataset_info.json to register a new dataset."""
    info_path = os.path.join(data_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
    else:
        info = {}

    relative_path = os.path.basename(data_file)
    entry = {
        "file_name": relative_path,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
    }
    if ranking:
        entry["ranking"] = True
        entry["columns"] = {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        }

    info[dataset_name] = entry

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  📝 Registered dataset '{dataset_name}' in {info_path}")


# ============================================================================
# Main Loop
# ============================================================================

def run_iterative_rft(config: RFTConfig):
    """Execute the full iterative RFT loop."""
    print()
    print("=" * 70)
    print("  ROUTE B: ITERATIVE REJECTION SAMPLING FINE-TUNING (RFT)")
    print(f"  Strategy: {config.strategy.upper()}")
    print(f"  Student:  {config.student_model}")
    print(f"  Teacher:  {config.teacher_model}")
    print(f"  Dataset:  {config.dataset}")
    print(f"  Iterations: {config.iterations}, K={config.k}")
    print("=" * 70)

    os.makedirs(config.output_root, exist_ok=True)

    # Save config
    with open(os.path.join(config.output_root, "rft_config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    current_model = config.student_model
    results: List[IterationResult] = []
    best_accuracy = 0.0
    patience_counter = 0

    for iteration in range(config.iterations):
        print(f"\n\n{'#' * 70}")
        print(f"#  ITERATION {iteration}")
        print(f"#  Model: {current_model}")
        print(f"{'#' * 70}\n")

        t0 = time.time()
        iter_result = IterationResult(iteration=iteration, student_model_path=current_model)

        # --- Step 1: Student Rollout ---
        rollout_start = time.time()
        student_dir = step_student_rollout(config, iteration, current_model)
        iter_result.rollout_time_sec = time.time() - rollout_start

        # Read rollout stats
        stats_path = os.path.join(student_dir, "rollout_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            iter_result.rollout_pass_at_1 = stats.get("pass_at_1", 0.0)
            iter_result.rollout_pass_at_k = stats.get("pass_at_k", 0.0)

        # --- Step 2: Teacher Backfill (if applicable) ---
        teacher_dir = None
        if config.teacher_model and config.strategy != "rest":
            failed_path = os.path.join(student_dir, "failed_queries.json")
            if os.path.exists(failed_path):
                teacher_dir = step_teacher_backfill(config, iteration, failed_path)

        # --- Step 2b: STaR retry with hint ---
        star_dir = None
        if config.strategy == "star":
            star_dir = step_star_retry_with_hint(config, iteration, student_dir, current_model)

        # --- Step 3: Merge & Convert ---
        merge_result = step_merge_and_convert(
            config, iteration, student_dir, teacher_dir, star_dir
        )
        iter_result.student_correct = merge_result["student_correct"]
        iter_result.teacher_backfill = merge_result["teacher_backfill"]
        iter_result.total_sft_samples = merge_result["sft_count"]
        iter_result.total_dpo_pairs = merge_result.get("dpo_count", 0)

        # --- Step 4: SFT Training ---
        if merge_result["sft_count"] < 5:
            print(f"  ⚠️  Only {merge_result['sft_count']} SFT samples; skipping training.")
            break

        train_start = time.time()
        new_model = step_train_sft(config, iteration, merge_result["sft_path"])
        iter_result.train_time_sec = time.time() - train_start

        # --- Step 4b: DPO Training (optional) ---
        if config.enable_dpo and merge_result.get("dpo_path"):
            dpo_model = step_train_dpo(config, iteration, merge_result["dpo_path"])
            new_model = dpo_model

        # --- Step 5: Evaluation ---
        eval_acc = step_evaluate(config, iteration, new_model)
        iter_result.eval_accuracy = eval_acc

        # --- Record results ---
        results.append(iter_result)
        current_model = new_model

        # Save iteration summary
        total_time = time.time() - t0
        _print_iteration_summary(iter_result, total_time)
        _save_iteration_results(config.output_root, results)

        # --- Early stopping ---
        improvement = eval_acc - best_accuracy
        if improvement >= config.min_improvement:
            best_accuracy = eval_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  ⏹️  Early stopping: no improvement for {config.patience} iterations.")
                break

    # Final summary
    _print_final_summary(results, config)
    return results


def _print_iteration_summary(result: IterationResult, total_time: float):
    print(f"\n{'─' * 60}")
    print(f"  ITERATION {result.iteration} SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Rollout pass@1:          {result.rollout_pass_at_1:.3f}")
    print(f"  Rollout pass@k:          {result.rollout_pass_at_k:.3f}")
    print(f"  Student correct trajs:   {result.student_correct}")
    print(f"  Teacher backfill trajs:  {result.teacher_backfill}")
    print(f"  Total SFT samples:       {result.total_sft_samples}")
    print(f"  DPO pairs:               {result.total_dpo_pairs}")
    print(f"  Eval accuracy:           {result.eval_accuracy:.3f}")
    print(f"  Rollout time:            {result.rollout_time_sec:.0f}s")
    print(f"  Training time:           {result.train_time_sec:.0f}s")
    print(f"  Total iteration time:    {total_time:.0f}s")
    print(f"{'─' * 60}")


def _save_iteration_results(output_root: str, results: List[IterationResult]):
    path = os.path.join(output_root, "iteration_results.json")
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def _print_final_summary(results: List[IterationResult], config: RFTConfig):
    print(f"\n\n{'=' * 70}")
    print("  ITERATIVE RFT — FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Strategy:    {config.strategy}")
    print(f"  Iterations:  {len(results)}")
    print()

    header = f"{'Iter':>4} {'Pass@1':>8} {'Pass@k':>8} {'SFT':>6} {'DPO':>5} {'Eval':>8} {'Model'}"
    print(f"  {header}")
    print(f"  {'─' * 70}")
    for r in results:
        model_short = Path(r.student_model_path).name[:30]
        line = (f"  {r.iteration:>4} {r.rollout_pass_at_1:>8.3f} "
                f"{r.rollout_pass_at_k:>8.3f} {r.total_sft_samples:>6} "
                f"{r.total_dpo_pairs:>5} {r.eval_accuracy:>8.3f} {model_short}")
        print(line)

    best = max(results, key=lambda r: r.eval_accuracy)
    print(f"\n  🏆 Best iteration: {best.iteration} (eval={best.eval_accuracy:.3f})")
    print(f"     Model: {best.student_model_path}")
    print(f"{'=' * 70}\n")


# ============================================================================
# CLI
# ============================================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description="Iterative Rejection Sampling Fine-Tuning (RFT) for DSGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Models
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--teacher-model", type=str,
                        default="together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput")
    parser.add_argument("--student-backend", type=str, default="vllm",
                        choices=["vllm", "multi-vllm", "sglang"])
    parser.add_argument("--teacher-backend", type=str, default="litellm",
                        choices=["litellm", "vllm", "multi-vllm", "sglang"])

    # Dataset
    parser.add_argument("--dataset", type=str, default="qrdata")
    parser.add_argument("--dataset-type", type=str, default="original")
    parser.add_argument("--synthetic-path", type=str, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)

    # Iteration
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--strategy", type=str, default="vanilla",
                        choices=["vanilla", "star", "rest"],
                        help="RFT strategy: vanilla, star (Self-Taught Reasoner), "
                             "rest (Reinforced Self-Training)")

    # Rollout
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=24)

    # Teacher
    parser.add_argument("--teacher-backfill-ratio", type=float, default=1.0)
    parser.add_argument("--teacher-backfill-decay", type=float, default=0.5)
    parser.add_argument("--teacher-k", type=int, default=3)

    # Training
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=str, default="1e-5")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accum", type=int, default=16)
    parser.add_argument("--finetuning-type", type=str, default="lora",
                        choices=["lora", "full"])
    parser.add_argument("--lora-rank", type=int, default=64)

    # DPO
    parser.add_argument("--enable-dpo", action="store_true")
    parser.add_argument("--dpo-beta", type=float, default=0.1)

    # Paths
    parser.add_argument("--output-root", type=str, default="./route_b_outputs")
    parser.add_argument("--llama-factory-root", type=str,
                        default="/data/fnie/LLaMA-Factory")
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000")

    # Early stopping
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-improvement", type=float, default=0.01)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = RFTConfig(
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        student_backend=args.student_backend,
        teacher_backend=args.teacher_backend,
        dataset=args.dataset,
        dataset_type=args.dataset_type,
        synthetic_path=args.synthetic_path,
        eval_limit=args.eval_limit,
        iterations=args.iterations,
        strategy=args.strategy,
        k=args.k,
        temperature=args.temperature,
        max_turns=args.max_turns,
        max_workers=args.max_workers,
        teacher_backfill_ratio=args.teacher_backfill_ratio,
        teacher_backfill_decay=args.teacher_backfill_decay,
        teacher_k=args.teacher_k,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accum=args.gradient_accum,
        finetuning_type=args.finetuning_type,
        lora_rank=args.lora_rank,
        enable_dpo=args.enable_dpo,
        dpo_beta=args.dpo_beta,
        output_root=args.output_root,
        llama_factory_root=args.llama_factory_root,
        manager_url=args.manager_url,
        patience=args.patience,
        min_improvement=args.min_improvement,
    )

    run_iterative_rft(config)


if __name__ == "__main__":
    main()
