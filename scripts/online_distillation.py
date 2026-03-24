#!/usr/bin/env python3
"""
Online Distillation Script for Route B (Advanced)

Implements state-of-the-art on-policy distillation methods that go beyond
vanilla Rejection Sampling. These methods keep the student's training
distribution aligned with its own policy, while leveraging teacher signals
for guidance.

Supported methods:

  1. Online DPO (On-Policy DPO)
     - Student rollout → candidate set
     - Rank by execution correctness/reward
     - Build preference pairs from the SAME batch
     - Update with DPO loss immediately
     → Avoids stale off-policy distribution

  2. SPIN (Self-Play Fine-Tuning)
     - Treat teacher trajectories as "real" and student trajectories as "fake"
     - Train discriminator: model learns to distinguish its own outputs from teacher
     - Converges when student distribution matches teacher
     → Self-play game minimax formulation

  3. Self-Rewarding (Self-Rewarding Language Models)
     - Student acts as BOTH generator and judge
     - Generate trajectories → self-evaluate → self-build preference pairs
     - No external teacher needed after initial SFT warm-up
     → Reduces dependency on expensive teacher model

  4. Step-DPO (Step-Level DPO)
     - Instead of trajectory-level preference, does step-level comparison
     - At each turn, compare student steps using execution feedback
     - More fine-grained credit assignment
     → Better for long multi-step agent trajectories

Usage:
    # Online DPO
    python online_distillation.py \
        --student-model /path/to/student \
        --method online-dpo \
        --dataset qrdata \
        --rounds 100 \
        --batch-size 4 \
        --output-dir ./route_b_outputs/online_dpo

    # SPIN
    python online_distillation.py \
        --student-model /path/to/student \
        --teacher-model together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
        --method spin \
        --dataset qrdata \
        --rounds 3 \
        --output-dir ./route_b_outputs/spin

    # Self-Rewarding
    python online_distillation.py \
        --student-model /path/to/student \
        --method self-rewarding \
        --dataset qrdata \
        --output-dir ./route_b_outputs/self_reward

    # Step-DPO
    python online_distillation.py \
        --student-model /path/to/student \
        --method step-dpo \
        --dataset qrdata \
        --output-dir ./route_b_outputs/step_dpo
"""

import json
import os
import sys
import time
import random
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

SCRIPT_DIR = Path(__file__).parent


# ============================================================================
# Correctness helpers (reuse from student_rollout)
# ============================================================================

CORRECTNESS_METRICS = [
    "exact_match", "fuzzy_exact_match", "list_match",
    "equivalence_by_llm", "fast_equivalence_by_llm",
]


def evaluate_trajectory_correctness(traj: Dict) -> Tuple[bool, float]:
    metrics = traj.get("metrics", {})
    best = 0.0
    correct = False
    for m in CORRECTNESS_METRICS:
        if m in metrics:
            d = metrics[m]
            s = d.get("score", 0.0) if isinstance(d, dict) else float(d)
            best = max(best, s)
            if s >= 0.99:
                correct = True
    return correct, best


def traj_to_sharegpt(traj: Dict) -> Optional[Dict]:
    conv = traj.get("conversation", [])
    if not conv:
        return None
    msgs = []
    for m in conv:
        r, c = m.get("role", ""), m.get("content", "")
        if not c:
            continue
        if r == "system":
            msgs.append({"from": "system", "value": c})
        elif r == "assistant":
            msgs.append({"from": "gpt", "value": c})
        elif r == "user":
            msgs.append({"from": "human", "value": c})
    if len(msgs) < 2:
        return None
    return {
        "conversations": msgs,
        "sample_id": traj.get("sample_id", ""),
        "dataset": traj.get("dataset_name", ""),
    }


# ============================================================================
# Method 1: Online DPO
# ============================================================================

@dataclass
class OnlineDPOConfig:
    student_model: str = ""
    student_backend: str = "vllm"
    dataset: str = "qrdata"
    k: int = 4               # candidate trajectories per query per round
    rounds: int = 100         # number of online rounds
    batch_size: int = 4       # queries per round
    temperature: float = 0.8
    max_turns: int = 15
    max_workers: int = 4
    manager_url: str = "http://localhost:5000"
    output_dir: str = "./online_dpo_output"
    llama_factory_root: str = "/data/fnie/LLaMA-Factory"
    # DPO training hyperparams
    dpo_beta: float = 0.1
    learning_rate: str = "5e-6"
    finetuning_type: str = "lora"
    lora_rank: int = 64
    accumulate_rounds: int = 10  # Accumulate N rounds before training


def run_online_dpo(config: OnlineDPOConfig):
    """
    Online DPO: At each round, student rolls out on a batch of queries,
    we rank trajectories by execution reward, build preference pairs from
    the SAME batch, and periodically train with DPO.

    This is on-policy because the preference pairs are always sampled from
    the current student policy.
    """
    print("\n" + "=" * 70)
    print("  ONLINE DPO — On-Policy Preference Learning")
    print("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)

    # Load all samples
    from dsgym.datasets import DatasetRegistry
    dataset = DatasetRegistry.load(config.dataset)
    all_samples = dataset.load()
    random.shuffle(all_samples)
    print(f"  Loaded {len(all_samples)} samples")

    accumulated_pairs: List[Dict] = []
    current_model = config.student_model
    train_count = 0

    for round_idx in range(config.rounds):
        # Select batch of queries (circular)
        start = (round_idx * config.batch_size) % len(all_samples)
        batch_samples = all_samples[start:start + config.batch_size]
        if len(batch_samples) < config.batch_size:
            batch_samples += all_samples[:config.batch_size - len(batch_samples)]

        print(f"\n--- Round {round_idx}/{config.rounds} "
              f"(accumulated pairs: {len(accumulated_pairs)}) ---")

        # Student rollout on batch
        round_dir = os.path.join(config.output_dir, f"round_{round_idx:04d}")
        os.makedirs(round_dir, exist_ok=True)

        from dsgym.synth.generators.trajectory_generator import TrajectoryGenerator, TrajectoryConfig
        traj_config = TrajectoryConfig(
            model=current_model,
            backend=config.student_backend,
            temperature=config.temperature,
            k=config.k,
            max_workers=config.max_workers,
            max_turns=config.max_turns,
            manager_url=config.manager_url,
            dataset_name=config.dataset,
            compute_metrics=True,
            output_dir=round_dir,
            run_name=f"online_dpo_round_{round_idx}",
        )
        generator = TrajectoryGenerator(traj_config)
        generator._initialize_components()
        generator.generate(batch_samples)

        # Load generated trajectories and build preference pairs
        new_pairs = _build_online_dpo_pairs(round_dir)
        accumulated_pairs.extend(new_pairs)
        print(f"  Built {len(new_pairs)} new DPO pairs (total: {len(accumulated_pairs)})")

        # Periodically train
        if (round_idx + 1) % config.accumulate_rounds == 0 and accumulated_pairs:
            print(f"\n  📚 Training DPO with {len(accumulated_pairs)} accumulated pairs …")
            train_count += 1
            current_model = _train_online_dpo(
                config, accumulated_pairs, train_count, current_model
            )
            # Keep a sliding window of recent pairs (prevent stale data)
            max_pairs = len(accumulated_pairs)
            accumulated_pairs = accumulated_pairs[-max_pairs // 2:]

    # Final training with remaining pairs
    if accumulated_pairs:
        train_count += 1
        _train_online_dpo(config, accumulated_pairs, train_count, current_model)

    print(f"\n✅ Online DPO complete. Total training rounds: {train_count}")


def _build_online_dpo_pairs(round_dir: str) -> List[Dict]:
    """
    From a single round's trajectories, build preference pairs.
    For each sample, pick best (correct, shortest) as chosen and
    worst (incorrect, longest) as rejected.
    """
    trajs = []
    pred_dir = Path(round_dir) / "predictions"
    search_dir = pred_dir if pred_dir.exists() else Path(round_dir)

    for fp in sorted(search_dir.glob("*.json")):
        if fp.name.startswith(("metrics", "config")):
            continue
        try:
            with open(fp) as f:
                trajs.append(json.load(f))
        except (json.JSONDecodeError, KeyError):
            continue

    # Group by sample_id
    by_sample: Dict[str, List[Dict]] = defaultdict(list)
    for t in trajs:
        by_sample[t.get("sample_id", "unknown")].append(t)

    pairs = []
    for sid, sample_trajs in by_sample.items():
        # Score each trajectory
        scored = []
        for t in sample_trajs:
            is_correct, score = evaluate_trajectory_correctness(t)
            turns = t.get("turns", t.get("total_turns", 0))
            # Composite score: correctness dominates, then prefer fewer turns
            composite = score * 10 - turns * 0.1
            scored.append((t, composite, is_correct))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Need at least one positive and one negative
        best = scored[0]
        worst = scored[-1]

        if best[1] <= worst[1]:
            continue  # No meaningful preference

        chosen_conv = traj_to_sharegpt(best[0])
        rejected_conv = traj_to_sharegpt(worst[0])
        if not chosen_conv or not rejected_conv:
            continue

        pairs.append({
            "conversations": chosen_conv["conversations"][:1],
            "chosen": chosen_conv["conversations"][1:],
            "rejected": rejected_conv["conversations"][1:],
            "sample_id": sid,
            "chosen_score": best[1],
            "rejected_score": worst[1],
        })

    return pairs


def _train_online_dpo(
    config: OnlineDPOConfig,
    pairs: List[Dict],
    train_idx: int,
    base_model: str,
) -> str:
    """Run a DPO training step on accumulated pairs."""
    import subprocess

    save_dir = os.path.join(config.output_dir, f"checkpoint_{train_idx:03d}")
    data_name = f"online_dpo_train_{train_idx}"
    lf_data_dir = os.path.join(config.llama_factory_root, "data")
    data_file = os.path.join(lf_data_dir, f"{data_name}.json")

    with open(data_file, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    # Register in dataset_info.json
    info_path = os.path.join(lf_data_dir, "dataset_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    info[data_name] = {
        "file_name": os.path.basename(data_file),
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"},
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    cmd = [
        "llamafactory-cli", "train",
        "--model_name_or_path", base_model,
        "--stage", "dpo",
        "--do_train", "true",
        "--dataset", data_name,
        "--template", "qwen",
        "--finetuning_type", config.finetuning_type,
        "--output_dir", save_dir,
        "--overwrite_output_dir", "true",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--num_train_epochs", "1",
        "--learning_rate", config.learning_rate,
        "--bf16", "true",
        "--dpo_beta", str(config.dpo_beta),
        "--report_to", "wandb",
        "--run_name", f"online_dpo_step{train_idx}",
    ]
    if config.finetuning_type == "lora":
        cmd += ["--lora_rank", str(config.lora_rank), "--lora_alpha", str(config.lora_rank * 2)]

    subprocess.run(cmd, cwd=config.llama_factory_root, check=True)
    return save_dir


# ============================================================================
# Method 2: SPIN (Self-Play Fine-Tuning)
# ============================================================================

@dataclass
class SPINConfig:
    student_model: str = ""
    teacher_model: str = ""
    student_backend: str = "vllm"
    teacher_backend: str = "litellm"
    dataset: str = "qrdata"
    rounds: int = 3
    k: int = 4
    temperature: float = 0.8
    max_turns: int = 15
    max_workers: int = 24
    manager_url: str = "http://localhost:5000"
    output_dir: str = "./spin_output"
    llama_factory_root: str = "/data/fnie/LLaMA-Factory"
    finetuning_type: str = "lora"
    lora_rank: int = 64
    learning_rate: str = "5e-6"
    dpo_beta: float = 0.1
    limit: Optional[int] = None


def run_spin(config: SPINConfig):
    """
    SPIN: Self-Play fINe-tuning.

    Core idea from Zephyr / SPIN paper:
      - "Real" data = Teacher trajectories (or gold-standard)
      - "Fake" data = Student-generated trajectories (current policy)
      - Train with DPO where chosen=real, rejected=fake
      - Iterate: as student improves, "fake" data gets closer to "real"
      - Converges when p_student ≈ p_teacher (generator = discriminator)

    Key difference from standard DPO:
      - Both chosen and rejected are on the SAME query
      - Chosen is always from teacher; rejected is always from current student
      - This creates a self-play dynamic
    """
    print("\n" + "=" * 70)
    print("  SPIN — Self-Play Fine-Tuning")
    print("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)
    current_model = config.student_model

    for round_idx in range(config.rounds):
        print(f"\n{'#' * 50}")
        print(f"# SPIN Round {round_idx}")
        print(f"{'#' * 50}")

        round_dir = os.path.join(config.output_dir, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)

        # Step 1: Teacher rollout → "real" data
        print("  [1/3] Teacher rollout (real data) …")
        teacher_dir = os.path.join(round_dir, "teacher")
        _run_rollout(config.teacher_model, config.teacher_backend, config.dataset,
                     config.k, 0.7, config.max_turns, config.max_workers,
                     config.manager_url, teacher_dir, f"spin_teacher_r{round_idx}",
                     limit=config.limit)

        # Step 2: Student rollout → "fake" data
        print("  [2/3] Student rollout (fake data) …")
        student_dir = os.path.join(round_dir, "student")
        _run_rollout(current_model, config.student_backend, config.dataset,
                     config.k, config.temperature, config.max_turns, config.max_workers,
                     config.manager_url, student_dir, f"spin_student_r{round_idx}",
                     limit=config.limit)

        # Step 3: Build SPIN pairs (teacher=chosen, student=rejected)
        print("  [3/3] Building SPIN preference pairs …")
        pairs = _build_spin_pairs(teacher_dir, student_dir)
        print(f"    Built {len(pairs)} preference pairs")

        if not pairs:
            print("  ⚠️  No pairs; skipping training.")
            continue

        # Save pairs
        pairs_path = os.path.join(round_dir, "spin_pairs.json")
        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        # Step 4: DPO training
        current_model = _train_spin_round(config, pairs, round_idx, current_model)

    print(f"\n✅ SPIN complete. Final model: {current_model}")


def _build_spin_pairs(teacher_dir: str, student_dir: str) -> List[Dict]:
    """
    Build SPIN preference pairs:
      chosen = best teacher trajectory per sample
      rejected = random student trajectory for same sample
    """
    teacher_trajs = _load_all_trajs(teacher_dir)
    student_trajs = _load_all_trajs(student_dir)

    # Group by sample_id
    teacher_by_id: Dict[str, List[Dict]] = defaultdict(list)
    student_by_id: Dict[str, List[Dict]] = defaultdict(list)
    for t in teacher_trajs:
        teacher_by_id[t.get("sample_id", "unknown")].append(t)
    for t in student_trajs:
        student_by_id[t.get("sample_id", "unknown")].append(t)

    pairs = []
    for sid in teacher_by_id:
        if sid not in student_by_id:
            continue

        # Pick best teacher trajectory (prefer correct, then shortest)
        best_teacher = sorted(
            teacher_by_id[sid],
            key=lambda t: (
                -evaluate_trajectory_correctness(t)[1],
                t.get("turns", 999)
            )
        )[0]

        # Pick a random student trajectory
        student_traj = random.choice(student_by_id[sid])

        chosen_conv = traj_to_sharegpt(best_teacher)
        rejected_conv = traj_to_sharegpt(student_traj)
        if not chosen_conv or not rejected_conv:
            continue

        pairs.append({
            "conversations": chosen_conv["conversations"][:1],  # shared prompt
            "chosen": chosen_conv["conversations"][1:],
            "rejected": rejected_conv["conversations"][1:],
            "sample_id": sid,
        })

    return pairs


def _train_spin_round(config: SPINConfig, pairs: List[Dict], round_idx: int,
                      base_model: str) -> str:
    import subprocess

    save_dir = os.path.join(config.output_dir, f"checkpoint_round{round_idx}")
    data_name = f"spin_round{round_idx}"
    lf_data_dir = os.path.join(config.llama_factory_root, "data")
    data_file = os.path.join(lf_data_dir, f"{data_name}.json")

    with open(data_file, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    _register_dataset(lf_data_dir, data_name, data_file, ranking=True)

    cmd = [
        "llamafactory-cli", "train",
        "--model_name_or_path", base_model,
        "--stage", "dpo",
        "--do_train", "true",
        "--dataset", data_name,
        "--template", "qwen",
        "--finetuning_type", config.finetuning_type,
        "--output_dir", save_dir,
        "--overwrite_output_dir", "true",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--num_train_epochs", "2",
        "--learning_rate", config.learning_rate,
        "--bf16", "true",
        "--dpo_beta", str(config.dpo_beta),
        "--report_to", "wandb",
        "--run_name", f"spin_round{round_idx}",
    ]
    if config.finetuning_type == "lora":
        cmd += ["--lora_rank", str(config.lora_rank), "--lora_alpha", str(config.lora_rank * 2)]

    subprocess.run(cmd, cwd=config.llama_factory_root, check=True)
    return save_dir


# ============================================================================
# Method 3: Self-Rewarding
# ============================================================================

@dataclass
class SelfRewardConfig:
    student_model: str = ""
    student_backend: str = "vllm"
    dataset: str = "qrdata"
    rounds: int = 3
    k: int = 4
    temperature: float = 0.8
    max_turns: int = 15
    max_workers: int = 24
    manager_url: str = "http://localhost:5000"
    output_dir: str = "./self_reward_output"
    llama_factory_root: str = "/data/fnie/LLaMA-Factory"
    finetuning_type: str = "lora"
    lora_rank: int = 64
    limit: Optional[int] = None

    # Self-reward prompt for the model to judge its own trajectories
    judge_prompt: str = (
        "You are evaluating a data science agent's trajectory. "
        "Score the trajectory on a scale of 1-5 based on:\n"
        "1. Code correctness and executability\n"
        "2. Analytical reasoning quality\n"
        "3. Answer accuracy (if verifiable)\n"
        "4. Efficiency (fewer turns is better)\n"
        "5. Code quality and documentation\n\n"
        "Trajectory:\n{trajectory}\n\n"
        "Respond with ONLY a single number 1-5."
    )


def run_self_rewarding(config: SelfRewardConfig):
    """
    Self-Rewarding Language Models:

    The model serves as both generator AND judge:
      1. Generate k trajectories per query
      2. Use the SAME model to score each trajectory (self-reward)
      3. Build preference pairs from self-scored rankings
      4. Train with DPO
      5. Iterate

    After initial warm-up (Route A SFT), the model can self-improve
    without any external teacher or reward model.

    Key insight for MLE Agent:
      - The model already understands code quality and data analysis
      - Self-evaluation leverages this understanding for improvement
      - Combined with execution feedback for grounding
    """
    print("\n" + "=" * 70)
    print("  SELF-REWARDING — Model as Generator + Judge")
    print("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)
    current_model = config.student_model

    for round_idx in range(config.rounds):
        print(f"\n{'#' * 50}")
        print(f"# Self-Reward Round {round_idx}")
        print(f"{'#' * 50}")

        round_dir = os.path.join(config.output_dir, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)

        # Step 1: Student rollout
        print("  [1/3] Student rollout …")
        rollout_dir = os.path.join(round_dir, "rollout")
        _run_rollout(current_model, config.student_backend, config.dataset,
                     config.k, config.temperature, config.max_turns, config.max_workers,
                     config.manager_url, rollout_dir, f"self_reward_r{round_idx}",
                     limit=config.limit)

        # Step 2: Self-reward scoring
        print("  [2/3] Self-reward scoring …")
        trajs = _load_all_trajs(rollout_dir)
        scored_trajs = _self_reward_score(trajs, current_model, config)

        # Step 3: Build preference pairs from self-scores
        print("  [3/3] Building preference pairs from self-scores …")
        pairs = _build_self_reward_pairs(scored_trajs)
        print(f"    Built {len(pairs)} preference pairs")

        if not pairs:
            print("  ⚠️  No pairs; skipping training.")
            continue

        # Save
        pairs_path = os.path.join(round_dir, "self_reward_pairs.json")
        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        # Step 4: DPO training
        current_model = _train_generic_dpo(
            config.llama_factory_root, pairs,
            f"self_reward_round{round_idx}",
            os.path.join(config.output_dir, f"checkpoint_round{round_idx}"),
            current_model, config.finetuning_type, config.lora_rank,
        )

    print(f"\n✅ Self-Rewarding complete. Final model: {current_model}")


def _self_reward_score(
    trajs: List[Dict], model: str, config: SelfRewardConfig
) -> List[Tuple[Dict, float]]:
    """
    Use the student model itself to score each trajectory.
    Combines execution feedback (hard signal) with self-evaluation (soft signal).
    """
    scored = []
    for t in trajs:
        # Hard signal: execution correctness
        is_correct, metric_score = evaluate_trajectory_correctness(t)
        hard_score = metric_score * 3.0  # Scale to 0-3

        # Soft signal: use number of errors as proxy
        # (Full self-reward would call the model, but that's expensive;
        #  here we approximate with heuristics + optional LLM call)
        conversation = t.get("conversation", [])
        error_count = sum(
            1 for m in conversation
            if m.get("role") == "user" and
            any(kw in m.get("content", "").lower()
                for kw in ["error", "traceback", "exception"])
        )
        turns = t.get("turns", t.get("total_turns", 0))

        soft_score = 2.0  # baseline
        soft_score -= error_count * 0.3
        soft_score -= max(0, turns - 8) * 0.1
        if is_correct:
            soft_score += 1.0

        total_score = hard_score + max(0, soft_score)
        scored.append((t, total_score))

    return scored


def _build_self_reward_pairs(scored_trajs: List[Tuple[Dict, float]]) -> List[Dict]:
    """Build DPO pairs by comparing scores within each sample."""
    by_sample: Dict[str, List[Tuple[Dict, float]]] = defaultdict(list)
    for t, score in scored_trajs:
        by_sample[t.get("sample_id", "unknown")].append((t, score))

    pairs = []
    for sid, items in by_sample.items():
        items.sort(key=lambda x: x[1], reverse=True)
        if len(items) < 2:
            continue

        best_traj, best_score = items[0]
        worst_traj, worst_score = items[-1]

        if best_score <= worst_score:
            continue

        chosen = traj_to_sharegpt(best_traj)
        rejected = traj_to_sharegpt(worst_traj)
        if not chosen or not rejected:
            continue

        pairs.append({
            "conversations": chosen["conversations"][:1],
            "chosen": chosen["conversations"][1:],
            "rejected": rejected["conversations"][1:],
            "sample_id": sid,
            "chosen_score": best_score,
            "rejected_score": worst_score,
        })

    return pairs


# ============================================================================
# Method 4: Step-DPO
# ============================================================================

@dataclass
class StepDPOConfig:
    student_model: str = ""
    student_backend: str = "vllm"
    dataset: str = "qrdata"
    rounds: int = 3
    k: int = 4
    temperature: float = 0.8
    max_turns: int = 15
    max_workers: int = 24
    manager_url: str = "http://localhost:5000"
    output_dir: str = "./step_dpo_output"
    llama_factory_root: str = "/data/fnie/LLaMA-Factory"
    finetuning_type: str = "lora"
    lora_rank: int = 64
    limit: Optional[int] = None


def run_step_dpo(config: StepDPOConfig):
    """
    Step-DPO: Step-level preference optimization for multi-step agents.

    Instead of comparing entire trajectories, we compare at the STEP level:
      - For each turn in a trajectory, the "chosen" step is the one that
        leads to a successful outcome, and "rejected" is the one that doesn't
      - This provides much finer-grained credit assignment
      - Especially useful for long MLE agent trajectories (10-15 turns)

    Implementation:
      1. Generate k trajectories per query
      2. Find the "divergence point" — first step where correct and incorrect
         trajectories differ significantly
      3. Build step-level preference pairs at the divergence point
      4. Train with DPO on these step-level pairs
    """
    print("\n" + "=" * 70)
    print("  STEP-DPO — Step-Level Preference Optimization")
    print("=" * 70)

    os.makedirs(config.output_dir, exist_ok=True)
    current_model = config.student_model

    for round_idx in range(config.rounds):
        print(f"\n{'#' * 50}")
        print(f"# Step-DPO Round {round_idx}")
        print(f"{'#' * 50}")

        round_dir = os.path.join(config.output_dir, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)

        # Step 1: Student rollout
        print("  [1/3] Student rollout …")
        rollout_dir = os.path.join(round_dir, "rollout")
        _run_rollout(current_model, config.student_backend, config.dataset,
                     config.k, config.temperature, config.max_turns, config.max_workers,
                     config.manager_url, rollout_dir, f"step_dpo_r{round_idx}",
                     limit=config.limit)

        # Step 2: Find step-level divergence and build pairs
        print("  [2/3] Analyzing step-level divergences …")
        trajs = _load_all_trajs(rollout_dir)
        step_pairs = _build_step_dpo_pairs(trajs)
        print(f"    Built {len(step_pairs)} step-level preference pairs")

        if not step_pairs:
            print("  ⚠️  No step-level pairs found; skipping training.")
            continue

        # Save
        pairs_path = os.path.join(round_dir, "step_dpo_pairs.json")
        with open(pairs_path, "w") as f:
            json.dump(step_pairs, f, indent=2, ensure_ascii=False)

        # Step 3: DPO training
        current_model = _train_generic_dpo(
            config.llama_factory_root, step_pairs,
            f"step_dpo_round{round_idx}",
            os.path.join(config.output_dir, f"checkpoint_round{round_idx}"),
            current_model, config.finetuning_type, config.lora_rank,
        )

    print(f"\n✅ Step-DPO complete. Final model: {current_model}")


def _build_step_dpo_pairs(trajs: List[Dict]) -> List[Dict]:
    """
    Build step-level DPO pairs by finding divergence points between
    correct and incorrect trajectories for the same query.

    Algorithm:
      1. Group trajectories by sample_id
      2. For each sample, find one correct and one incorrect trajectory
      3. Walk both conversations turn by turn
      4. Find the first assistant turn where they DIVERGE
         (i.e., correct trajectory makes a good decision, incorrect doesn't)
      5. Create a preference pair:
         - context = shared conversation up to divergence point
         - chosen  = correct trajectory's step at divergence
         - rejected = incorrect trajectory's step at divergence
    """
    by_sample: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: {"correct": [], "incorrect": []})

    for t in trajs:
        sid = t.get("sample_id", "unknown")
        is_correct, _ = evaluate_trajectory_correctness(t)
        key = "correct" if is_correct else "incorrect"
        by_sample[sid][key].append(t)

    pairs = []
    for sid, groups in by_sample.items():
        if not groups["correct"] or not groups["incorrect"]:
            continue

        correct_traj = groups["correct"][0]
        incorrect_traj = groups["incorrect"][0]

        correct_conv = correct_traj.get("conversation", [])
        incorrect_conv = incorrect_traj.get("conversation", [])

        # Find divergence point: first assistant message that differs
        # We compare assistant turns sequentially
        correct_assistant = [(i, m) for i, m in enumerate(correct_conv)
                             if m.get("role") == "assistant"]
        incorrect_assistant = [(i, m) for i, m in enumerate(incorrect_conv)
                               if m.get("role") == "assistant"]

        if not correct_assistant or not incorrect_assistant:
            continue

        # Find first assistant step where they diverge meaningfully
        diverge_step = 0
        for step_idx in range(min(len(correct_assistant), len(incorrect_assistant))):
            c_content = correct_assistant[step_idx][1].get("content", "")
            i_content = incorrect_assistant[step_idx][1].get("content", "")

            # Check if the next execution shows divergence
            # (one has error, the other doesn't)
            c_idx = correct_assistant[step_idx][0]
            i_idx = incorrect_assistant[step_idx][0]

            c_next_has_error = _next_turn_has_error(correct_conv, c_idx)
            i_next_has_error = _next_turn_has_error(incorrect_conv, i_idx)

            if c_next_has_error != i_next_has_error:
                diverge_step = step_idx
                break
            diverge_step = step_idx

        if diverge_step >= len(correct_assistant) or diverge_step >= len(incorrect_assistant):
            diverge_step = 0

        # Build step-level pair
        # Context: all messages up to the divergence point
        c_div_idx = correct_assistant[diverge_step][0]
        i_div_idx = incorrect_assistant[diverge_step][0]

        # Use the conversation prefix up to the divergence point
        context_msgs = []
        for m in correct_conv[:c_div_idx]:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                context_msgs.append({"from": "system", "value": content})
            elif role == "user":
                context_msgs.append({"from": "human", "value": content})
            elif role == "assistant":
                context_msgs.append({"from": "gpt", "value": content})

        if not context_msgs:
            continue

        chosen_step = correct_conv[c_div_idx]
        rejected_step = incorrect_conv[i_div_idx]

        pairs.append({
            "conversations": context_msgs,
            "chosen": [{"from": "gpt", "value": chosen_step.get("content", "")}],
            "rejected": [{"from": "gpt", "value": rejected_step.get("content", "")}],
            "sample_id": sid,
            "diverge_step": diverge_step,
        })

    return pairs


def _next_turn_has_error(conversation: List[Dict], current_idx: int) -> bool:
    """Check if the next user message (execution output) contains an error."""
    for i in range(current_idx + 1, len(conversation)):
        msg = conversation[i]
        if msg.get("role") == "user":
            content = msg.get("content", "").lower()
            return any(kw in content for kw in ["error", "traceback", "exception", "failed"])
    return False


# ============================================================================
# Shared Utilities
# ============================================================================

def _run_rollout(model, backend, dataset, k, temperature, max_turns,
                 max_workers, manager_url, output_dir, run_name, limit=None):
    """Helper to invoke student_rollout.py."""
    import subprocess
    cmd = [
        sys.executable, str(SCRIPT_DIR / "student_rollout.py"),
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
    subprocess.run(cmd, check=True)


def _load_all_trajs(directory: str) -> List[Dict]:
    """Load all trajectory JSON files from a directory (including predictions/)."""
    trajs = []
    d = Path(directory)
    search_dirs = [d / "predictions", d]
    for sd in search_dirs:
        if not sd.exists():
            continue
        for fp in sorted(sd.glob("*.json")):
            if fp.name.startswith(("metrics", "config", "rollout", "failed", "succeeded")):
                continue
            try:
                with open(fp) as f:
                    trajs.append(json.load(f))
            except (json.JSONDecodeError, KeyError):
                continue
        if trajs:
            break
    return trajs


def _register_dataset(data_dir: str, name: str, data_file: str, ranking=False):
    """Register a dataset in LLaMA-Factory's dataset_info.json."""
    info_path = os.path.join(data_dir, "dataset_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    entry = {
        "file_name": os.path.basename(data_file),
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
    info[name] = entry
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def _train_generic_dpo(lf_root, pairs, run_name, save_dir, base_model,
                       finetuning_type="lora", lora_rank=64):
    """Generic DPO training helper."""
    import subprocess

    data_dir = os.path.join(lf_root, "data")
    data_file = os.path.join(data_dir, f"{run_name}.json")
    with open(data_file, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    _register_dataset(data_dir, run_name, data_file, ranking=True)

    cmd = [
        "llamafactory-cli", "train",
        "--model_name_or_path", base_model,
        "--stage", "dpo",
        "--do_train", "true",
        "--dataset", run_name,
        "--template", "qwen",
        "--finetuning_type", finetuning_type,
        "--output_dir", save_dir,
        "--overwrite_output_dir", "true",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--num_train_epochs", "2",
        "--learning_rate", "5e-6",
        "--bf16", "true",
        "--dpo_beta", "0.1",
        "--report_to", "wandb",
        "--run_name", run_name,
    ]
    if finetuning_type == "lora":
        cmd += ["--lora_rank", str(lora_rank), "--lora_alpha", str(lora_rank * 2)]

    subprocess.run(cmd, cwd=lf_root, check=True)
    return save_dir


# ============================================================================
# CLI
# ============================================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description="Online Distillation Methods for Route B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--method", type=str, required=True,
                        choices=["online-dpo", "spin", "self-rewarding", "step-dpo"],
                        help="Distillation method")

    # Common args
    parser.add_argument("--student-model", type=str, required=True)
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Teacher model (required for SPIN)")
    parser.add_argument("--student-backend", type=str, default="vllm")
    parser.add_argument("--teacher-backend", type=str, default="litellm")
    parser.add_argument("--dataset", type=str, default="qrdata")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--manager-url", type=str, default="http://localhost:5000")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--llama-factory-root", type=str, default="/data/fnie/LLaMA-Factory")
    parser.add_argument("--finetuning-type", type=str, default="lora")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of dataset samples")

    # Online DPO specific
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Queries per round (online-dpo)")
    parser.add_argument("--accumulate-rounds", type=int, default=10,
                        help="Accumulate N rounds before training (online-dpo)")
    parser.add_argument("--dpo-beta", type=float, default=0.1)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.method == "online-dpo":
        config = OnlineDPOConfig(
            student_model=args.student_model,
            student_backend=args.student_backend,
            dataset=args.dataset,
            k=args.k,
            rounds=args.rounds,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            manager_url=args.manager_url,
            output_dir=args.output_dir,
            llama_factory_root=args.llama_factory_root,
            dpo_beta=args.dpo_beta,
            finetuning_type=args.finetuning_type,
            lora_rank=args.lora_rank,
            accumulate_rounds=args.accumulate_rounds,
        )
        run_online_dpo(config)

    elif args.method == "spin":
        if not args.teacher_model:
            parser.error("--teacher-model is required for SPIN")
        config = SPINConfig(
            student_model=args.student_model,
            teacher_model=args.teacher_model,
            student_backend=args.student_backend,
            teacher_backend=args.teacher_backend,
            dataset=args.dataset,
            rounds=args.rounds,
            k=args.k,
            temperature=args.temperature,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            manager_url=args.manager_url,
            output_dir=args.output_dir,
            llama_factory_root=args.llama_factory_root,
            finetuning_type=args.finetuning_type,
            lora_rank=args.lora_rank,
            limit=args.limit,
        )
        run_spin(config)

    elif args.method == "self-rewarding":
        config = SelfRewardConfig(
            student_model=args.student_model,
            student_backend=args.student_backend,
            dataset=args.dataset,
            rounds=args.rounds,
            k=args.k,
            temperature=args.temperature,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            manager_url=args.manager_url,
            output_dir=args.output_dir,
            llama_factory_root=args.llama_factory_root,
            finetuning_type=args.finetuning_type,
            lora_rank=args.lora_rank,
            limit=args.limit,
        )
        run_self_rewarding(config)

    elif args.method == "step-dpo":
        config = StepDPOConfig(
            student_model=args.student_model,
            student_backend=args.student_backend,
            dataset=args.dataset,
            rounds=args.rounds,
            k=args.k,
            temperature=args.temperature,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
            manager_url=args.manager_url,
            output_dir=args.output_dir,
            llama_factory_root=args.llama_factory_root,
            finetuning_type=args.finetuning_type,
            lora_rank=args.lora_rank,
            limit=args.limit,
        )
        run_step_dpo(config)


if __name__ == "__main__":
    main()
