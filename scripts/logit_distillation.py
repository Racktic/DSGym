#!/usr/bin/env python3
"""
Logit-Level On-Policy Distillation for DSGym MLE Agent

Implements three state-of-the-art logit distillation methods:

  1. Qwen3-Style: External teacher logit distillation
     - Student on-policy sampling → Teacher forward pass → Token-level KL alignment
     - "RL with one line change": replace KL reference with teacher model
     - Transmits O(N) bits per episode vs O(1) for outcome-based RL

  2. SDPO (Self-Distillation for Policy Optimization)
     - Same model as teacher, conditioned on execution feedback
     - Forward pass WITH feedback (teacher) vs WITHOUT feedback (student)
     - JSD divergence for symmetric alignment
     - No external teacher needed

  3. OPSD (On-Policy Self-Distillation)
     - Same model conditioned on correct answer: p_T(·|x, y*)
     - vs student: p_S(·|x)
     - Full-vocabulary logit distillation (all V tokens per position)
     - Best fine-grained credit assignment

Architecture:
  ┌──────────────────────────────────────────────────────────┐
  │                  Logit Distillation Loop                  │
  │                                                          │
  │  ┌─────────┐  on-policy   ┌──────────────┐              │
  │  │ Student │ ──────────→  │  Rollout in   │              │
  │  │ (vLLM)  │   sampling   │  DSGym Env    │              │
  │  └─────────┘              └──────┬───────┘              │
  │                                  │ trajectories          │
  │                                  ▼                       │
  │  ┌──────────────────────────────────────────┐           │
  │  │  For each trajectory token sequence:     │           │
  │  │                                          │           │
  │  │  Student logits = student.forward(seq)   │  ← grad  │
  │  │  Teacher logits = teacher.forward(seq)   │  ← no grad│
  │  │                                          │           │
  │  │  loss = KL(teacher || student)  [Qwen3]  │           │
  │  │       = JSD(teacher, student)   [SDPO]   │           │
  │  │       = KL_full_vocab(t || s)   [OPSD]   │           │
  │  └──────────────────────────────────────────┘           │
  │                     │                                    │
  │                     ▼                                    │
  │              optimizer.step()                            │
  │              → Updated Student                           │
  └──────────────────────────────────────────────────────────┘

Usage:
    # Qwen3-style with external teacher
    python logit_distillation.py \
        --method qwen3 \
        --student-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --teacher-model Qwen/Qwen3-235B-A22B-Instruct-2507-tput \
        --dataset qrdata \
        --output-dir ./logit_distill_outputs

    # SDPO (self-distillation with execution feedback)
    python logit_distillation.py \
        --method sdpo \
        --student-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset qrdata \
        --output-dir ./sdpo_outputs

    # OPSD (self-distillation with answer conditioning)
    python logit_distillation.py \
        --method opsd \
        --student-model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset qrdata \
        --output-dir ./opsd_outputs

References:
  - Qwen3 Technical Report (2025)
  - Shenfeld et al., "SDPO: Self-Distillation for Policy Optimization" (2026)
  - Zhao et al., "OPSD: On-Policy Self-Distillation" (2026)
  - Kevin Lu / Thinking Machines Lab blog (2026)
"""

import json
import os
import sys
import time
import math
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from peft import PeftModel, LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import deepspeed
    DS_AVAILABLE = True
except ImportError:
    DS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# Loss Functions — the core of logit distillation
# ============================================================================

def kl_divergence_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Forward KL divergence: KL(p_teacher || p_student)

    This is the standard knowledge distillation loss.
    Minimizing KL(teacher || student) encourages the student to cover
    all modes of the teacher distribution (mode-covering / mean-seeking).

    Args:
        teacher_logits: [batch, seq_len, vocab_size]
        student_logits: [batch, seq_len, vocab_size]
        temperature: Softmax temperature (higher = softer distribution)
        mask: [batch, seq_len] binary mask (1 = compute loss, 0 = ignore)
        reduction: "mean" or "sum" or "none"

    Returns:
        Scalar loss or per-token loss if reduction="none"
    """
    # Scale logits by temperature
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
    # = sum_x P(x) * (log P(x) - log Q(x))
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)  # [batch, seq_len]

    # Scale by T^2 (standard KD practice)
    kl = kl * (temperature ** 2)

    if mask is not None:
        kl = kl * mask

    if reduction == "mean":
        if mask is not None:
            return kl.sum() / mask.sum().clamp(min=1)
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl


def reverse_kl_divergence_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Reverse KL divergence: KL(p_student || p_teacher)

    This is mode-seeking: the student focuses on the highest-probability
    modes of the teacher. Used in some RL formulations where we want the
    student to avoid spreading mass on low-quality regions.
    """
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # KL(Q || P) = sum_x Q(x) * (log Q(x) - log P(x))
    kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    kl = kl * (temperature ** 2)

    if mask is not None:
        kl = kl * mask
    if reduction == "mean":
        return kl.sum() / (mask.sum().clamp(min=1) if mask is not None else kl.numel())
    elif reduction == "sum":
        return kl.sum()
    return kl


def jsd_divergence_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
    beta: float = 0.5,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Jensen-Shannon Divergence: JSD(p_teacher, p_student)

    JSD = beta * KL(P || M) + (1-beta) * KL(Q || M)
    where M = beta * P + (1-beta) * Q

    This is the loss used in SDPO. JSD is symmetric and bounded [0, log 2],
    making training more stable than pure KL.

    Args:
        beta: Interpolation weight. 0.5 = symmetric JSD.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.softmax(student_logits / temperature, dim=-1)

    # Mixture distribution M
    m_probs = beta * teacher_probs + (1 - beta) * student_probs
    m_log_probs = m_probs.clamp(min=1e-8).log()

    # JSD = beta * KL(teacher || M) + (1-beta) * KL(student || M)
    teacher_log_probs = teacher_probs.clamp(min=1e-8).log()
    student_log_probs = student_probs.clamp(min=1e-8).log()

    kl_teacher_m = (teacher_probs * (teacher_log_probs - m_log_probs)).sum(dim=-1)
    kl_student_m = (student_probs * (student_log_probs - m_log_probs)).sum(dim=-1)

    jsd = beta * kl_teacher_m + (1 - beta) * kl_student_m
    jsd = jsd * (temperature ** 2)

    if mask is not None:
        jsd = jsd * mask
    if reduction == "mean":
        return jsd.sum() / (mask.sum().clamp(min=1) if mask is not None else jsd.numel())
    elif reduction == "sum":
        return jsd.sum()
    return jsd


def full_vocab_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    alpha_kl: float = 0.5,
    alpha_rkl: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Full-vocabulary distillation loss (OPSD-style).

    Combines forward KL and reverse KL for balanced optimization:
      - Forward KL: ensures student covers all teacher modes (no mode dropping)
      - Reverse KL: ensures student doesn't spread mass on junk tokens

    L = alpha_kl * KL(teacher || student) + alpha_rkl * KL(student || teacher)

    This provides stronger signal than sampled-token objectives because
    the student sees the teacher's full distribution over ALL possible
    next tokens, not just the sampled one.
    """
    fwd_kl = kl_divergence_loss(
        teacher_logits, student_logits, temperature, mask, reduction
    )
    rev_kl = reverse_kl_divergence_loss(
        teacher_logits, student_logits, temperature, mask, reduction
    )
    return alpha_kl * fwd_kl + alpha_rkl * rev_kl


# ============================================================================
# Auxiliary losses
# ============================================================================

def sft_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Standard cross-entropy SFT loss on ground-truth tokens.
    Can be mixed with distillation loss for stability.
    """
    # Shift for next-token prediction
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)

    if mask is not None:
        shift_mask = mask[..., 1:].contiguous()
        loss = loss * shift_mask
        return loss.sum() / shift_mask.sum().clamp(min=1)
    return loss.mean()


def outcome_weighted_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    rewards: torch.Tensor,
    temperature: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    loss_fn: str = "kl",
) -> torch.Tensor:
    """
    Reward-weighted distillation: weight the distillation loss by trajectory reward.

    Correct trajectories get higher weight → model learns more from successful rollouts.
    This bridges pure distillation and RL by incorporating outcome signal.

    Args:
        rewards: [batch] per-trajectory reward (0 to 1 scale)
    """
    if loss_fn == "kl":
        per_token_loss = kl_divergence_loss(
            teacher_logits, student_logits, temperature, mask, reduction="none"
        )
    elif loss_fn == "jsd":
        per_token_loss = jsd_divergence_loss(
            teacher_logits, student_logits, temperature, mask=mask, reduction="none"
        )
    else:
        per_token_loss = full_vocab_distillation_loss(
            teacher_logits, student_logits, temperature, mask, reduction="none"
        )

    # Weight each sequence by its reward
    # rewards: [batch] → [batch, 1] for broadcasting
    seq_weights = rewards.unsqueeze(-1)  # [batch, 1]

    # Weighted per-token loss
    weighted_loss = per_token_loss * seq_weights

    if mask is not None:
        return weighted_loss.sum() / mask.sum().clamp(min=1)
    return weighted_loss.mean()


# ============================================================================
# Dataset for distillation
# ============================================================================

class DistillationDataset(Dataset):
    """
    Dataset that holds tokenized trajectories for distillation training.

    Each item contains:
      - input_ids: full sequence (prompt + response)
      - attention_mask: 1 for real tokens, 0 for padding
      - loss_mask: 1 for response tokens (assistant turns), 0 for prompt/observation tokens
      - reward: trajectory-level reward (0 or 1 for correctness)
      - teacher_input_ids: (SDPO/OPSD) teacher-conditioned sequence
    """

    def __init__(
        self,
        trajectories: List[Dict],
        tokenizer,
        max_length: int = 8192,
        method: str = "qwen3",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.method = method
        self.items = []

        for traj in trajectories:
            item = self._process_trajectory(traj)
            if item is not None:
                self.items.append(item)

    def _process_trajectory(self, traj: Dict) -> Optional[Dict]:
        """Convert a trajectory dict to tokenized training item."""
        conversation = traj.get("conversation", [])
        if not conversation:
            return None

        # Determine reward from metrics
        metrics = traj.get("metrics", {})
        reward = 0.0
        for m_name in ["exact_match", "fuzzy_exact_match"]:
            if m_name in metrics:
                val = metrics[m_name]
                score = val.get("score", 0.0) if isinstance(val, dict) else float(val)
                reward = max(reward, score)

        # Build student input (standard conversation)
        student_messages = conversation

        # Build teacher input for SDPO/OPSD
        teacher_messages = None
        if self.method == "sdpo":
            # SDPO: teacher sees execution feedback injected into context
            # The feedback is already in the conversation as user messages
            # Teacher version: same conversation but with an explicit "feedback summary"
            teacher_messages = self._build_sdpo_teacher_input(conversation, traj)
        elif self.method == "opsd":
            # OPSD: teacher is conditioned on the correct answer
            ground_truth = traj.get("ground_truth", "")
            if ground_truth:
                teacher_messages = self._build_opsd_teacher_input(
                    conversation, ground_truth
                )

        # Tokenize student sequence
        student_text = self.tokenizer.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=False,
        )
        student_enc = self.tokenizer(
            student_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )
        input_ids = student_enc["input_ids"].squeeze(0)
        attention_mask = student_enc["attention_mask"].squeeze(0)

        # Build loss mask: only compute loss on assistant tokens
        loss_mask = self._build_assistant_mask(conversation, input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "reward": torch.tensor(reward, dtype=torch.float32),
        }

        # Tokenize teacher sequence if applicable
        if teacher_messages is not None:
            teacher_text = self.tokenizer.apply_chat_template(
                teacher_messages, tokenize=False, add_generation_prompt=False,
            )
            teacher_enc = self.tokenizer(
                teacher_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
                padding=False,
            )
            result["teacher_input_ids"] = teacher_enc["input_ids"].squeeze(0)
            result["teacher_attention_mask"] = teacher_enc["attention_mask"].squeeze(0)

        return result

    def _build_sdpo_teacher_input(
        self, conversation: List[Dict], traj: Dict
    ) -> List[Dict]:
        """
        SDPO teacher input: inject execution feedback summary into system prompt.

        The teacher version sees a summary of what happened:
        - Which code blocks succeeded/failed
        - Final execution result
        - Error messages

        This gives the teacher "hindsight" to produce better logits,
        which then guide the student via distillation.
        """
        feedback_lines = []
        for i, msg in enumerate(conversation):
            if msg.get("role") == "user" and i > 0:  # Skip the initial task
                content = msg.get("content", "")
                if any(kw in content.lower() for kw in ["error", "traceback", "exception"]):
                    feedback_lines.append(f"Step {i//2}: Code execution ERROR")
                elif "[Code execution output]" in content or "Output:" in content:
                    feedback_lines.append(f"Step {i//2}: Code executed successfully")

        is_correct = traj.get("metrics", {}).get("exact_match", {})
        if isinstance(is_correct, dict):
            is_correct = is_correct.get("score", 0) >= 0.99
        else:
            is_correct = False

        prediction = traj.get("prediction", "")
        ground_truth = traj.get("ground_truth", "")

        feedback_summary = (
            f"[EXECUTION FEEDBACK]\n"
            f"Execution summary: {'; '.join(feedback_lines) if feedback_lines else 'No execution observed'}\n"
            f"Final prediction: {str(prediction)[:200]}\n"
            f"Correct: {'Yes' if is_correct else 'No'}\n"
        )
        if ground_truth and is_correct:
            feedback_summary += f"Ground truth: {ground_truth}\n"

        # Inject feedback into system message
        teacher_conv = []
        for msg in conversation:
            if msg.get("role") == "system":
                teacher_conv.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + feedback_summary,
                })
            else:
                teacher_conv.append(msg)

        if not any(m.get("role") == "system" for m in teacher_conv):
            teacher_conv.insert(0, {"role": "system", "content": feedback_summary})

        return teacher_conv

    def _build_opsd_teacher_input(
        self, conversation: List[Dict], ground_truth: str
    ) -> List[Dict]:
        """
        OPSD teacher input: condition on the correct answer.

        p_T(·|x, y*) where y* is the ground truth.
        The teacher model sees "The correct answer is {y*}" in the system prompt,
        then generates logits conditioned on this knowledge.
        """
        answer_hint = (
            f"[ANSWER CONDITIONING]\n"
            f"The correct final answer to this task is: {ground_truth}\n"
            f"Use this knowledge to generate the optimal step-by-step analysis."
        )

        teacher_conv = []
        for msg in conversation:
            if msg.get("role") == "system":
                teacher_conv.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + answer_hint,
                })
            else:
                teacher_conv.append(msg)

        if not any(m.get("role") == "system" for m in teacher_conv):
            teacher_conv.insert(0, {"role": "system", "content": answer_hint})

        return teacher_conv

    def _build_assistant_mask(
        self, conversation: List[Dict], input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Build a mask that is 1 for assistant-generated tokens, 0 for everything else.
        This ensures we only distill on the model's own outputs, not on
        prompts or environment observations.
        """
        # Simple heuristic: tokenize just the assistant parts and find their spans
        # For efficiency, use a pattern-matching approach
        mask = torch.zeros_like(input_ids, dtype=torch.float32)

        # Tokenize each assistant message to find approximate positions
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        current_pos = 0

        for msg in conversation:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not content:
                continue

            # Find this content in the full text
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
            n_tokens = len(content_tokens)

            # Search for matching token subsequence
            for start in range(current_pos, len(input_ids) - n_tokens + 1):
                if input_ids[start:start + min(5, n_tokens)].tolist() == content_tokens[:min(5, n_tokens)]:
                    mask[start:start + n_tokens] = 1.0
                    current_pos = start + n_tokens
                    break

        # Fallback: if no mask was set, mask all non-special tokens
        if mask.sum() == 0:
            special_ids = set(self.tokenizer.all_special_ids)
            for i, tid in enumerate(input_ids.tolist()):
                if tid not in special_ids:
                    mask[i] = 1.0

        return mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with dynamic padding."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    padded = defaultdict(list)
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        padded["input_ids"].append(
            F.pad(item["input_ids"], (0, pad_len), value=0)
        )
        padded["attention_mask"].append(
            F.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        padded["loss_mask"].append(
            F.pad(item["loss_mask"], (0, pad_len), value=0.0)
        )
        padded["reward"].append(item["reward"])

        # Teacher inputs (for SDPO/OPSD)
        if "teacher_input_ids" in item:
            t_len = item["teacher_input_ids"].size(0)
            t_pad = max_len - t_len
            padded["teacher_input_ids"].append(
                F.pad(item["teacher_input_ids"], (0, max(0, t_pad)), value=0)[:max_len]
            )
            padded["teacher_attention_mask"].append(
                F.pad(item["teacher_attention_mask"], (0, max(0, t_pad)), value=0)[:max_len]
            )

    result = {
        "input_ids": torch.stack(padded["input_ids"]),
        "attention_mask": torch.stack(padded["attention_mask"]),
        "loss_mask": torch.stack(padded["loss_mask"]),
        "reward": torch.stack(padded["reward"]),
    }

    if "teacher_input_ids" in padded:
        result["teacher_input_ids"] = torch.stack(padded["teacher_input_ids"])
        result["teacher_attention_mask"] = torch.stack(padded["teacher_attention_mask"])

    return result


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LogitDistillConfig:
    """Configuration for logit-level distillation."""

    # Method
    method: str = "qwen3"  # qwen3 | sdpo | opsd

    # Models
    student_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    teacher_model: str = ""  # Required for qwen3; auto-set for sdpo/opsd

    # LoRA
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: str = "all"  # "all" or comma-separated list like "q_proj,v_proj"

    # Training
    num_iterations: int = 3     # Outer loop: rollout + distill
    inner_epochs: int = 2       # Inner loop: epochs per distillation round
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    max_seq_length: int = 8192
    bf16: bool = True

    # Distillation hyperparameters
    distill_temperature: float = 2.0    # Logit softening temperature
    alpha_distill: float = 0.8          # Weight for distillation loss
    alpha_sft: float = 0.2             # Weight for SFT loss (auxiliary)
    loss_type: str = "kl"              # kl | reverse_kl | jsd | full_vocab
    jsd_beta: float = 0.5             # JSD interpolation (SDPO default)
    reward_weighted: bool = False      # Weight loss by trajectory reward

    # Rollout settings
    k: int = 4                         # Trajectories per query
    temperature: float = 0.8           # Sampling temperature
    max_turns: int = 15
    max_workers: int = 24
    manager_url: str = "http://localhost:5000"

    # Dataset
    dataset: str = "qrdata"
    dataset_limit: Optional[int] = None

    # Paths
    output_dir: str = "./logit_distill_outputs"

    # DeepSpeed
    use_deepspeed: bool = True
    deepspeed_stage: int = 2

    # Logging
    log_interval: int = 10
    save_interval: int = 1  # Save every N iterations
    use_wandb: bool = True
    wandb_project: str = "dsgym-logit-distill"
    run_name: Optional[str] = None


# ============================================================================
# Main Trainer
# ============================================================================

class LogitDistillationTrainer:
    """
    Main trainer for logit-level on-policy distillation.

    Supports three modes:
      - qwen3: External teacher model provides reference logits
      - sdpo:  Same model with/without feedback context
      - opsd:  Same model with/without answer conditioning
    """

    def __init__(self, config: LogitDistillConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Validate
        if config.method == "qwen3" and not config.teacher_model:
            raise ValueError("--teacher-model is required for qwen3 method")
        if config.method in ["sdpo", "opsd"]:
            config.teacher_model = config.student_model  # Self-distillation

        print(f"\n{'=' * 70}")
        print(f"  LOGIT DISTILLATION TRAINER")
        print(f"  Method:  {config.method.upper()}")
        print(f"  Student: {config.student_model}")
        print(f"  Teacher: {config.teacher_model or '(self)'}")
        print(f"  Loss:    {config.loss_type} (T={config.distill_temperature})")
        print(f"  LoRA:    {'rank=' + str(config.lora_rank) if config.use_lora else 'full'}")
        print(f"{'=' * 70}\n")

    def setup_models(self):
        """Load student and teacher models."""
        print("Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load student model (with gradients)
        print(f"Loading student model: {self.config.student_model} …")
        dtype = torch.bfloat16 if self.config.bf16 else torch.float32
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        # Apply LoRA if configured
        if self.config.use_lora and PEFT_AVAILABLE:
            print(f"Applying LoRA (rank={self.config.lora_rank}) …")
            target_modules = (
                None if self.config.lora_target_modules == "all"
                else self.config.lora_target_modules.split(",")
            )
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
            )
            self.student_model = get_peft_model(self.student_model, lora_config)
            self.student_model.print_trainable_parameters()

        self.student_model.train()

        # Load teacher model (no gradients)
        if self.config.method == "qwen3":
            print(f"Loading teacher model: {self.config.teacher_model} …")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False
        else:
            # SDPO / OPSD: teacher is the same model, but we do a separate
            # forward pass with different input. No separate model needed.
            self.teacher_model = None

    def setup_optimizer(self, num_training_steps: int):
        """Setup optimizer and scheduler."""
        trainable_params = [p for p in self.student_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    def get_deepspeed_config(self) -> Dict:
        """Generate DeepSpeed ZeRO config."""
        return {
            "bf16": {"enabled": self.config.bf16},
            "zero_optimization": {
                "stage": self.config.deepspeed_stage,
                "offload_optimizer": {"device": "none"},
                "offload_param": {"device": "none"},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_scatter": True,
            },
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": self.config.max_grad_norm,
            "train_micro_batch_size_per_gpu": self.config.batch_size,
            "wall_clock_breakdown": False,
        }

    # ------------------------------------------------------------------
    # Core distillation step
    # ------------------------------------------------------------------

    def distillation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute one distillation training step.

        Returns dict of loss components for logging.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)
        rewards = batch["reward"].to(self.device)

        # ── Student forward pass (with gradients) ──
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits  # [B, L, V]

        # ── Teacher forward pass (no gradients) ──
        with torch.no_grad():
            if self.config.method == "qwen3":
                # External teacher: same input, different model
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits

            elif self.config.method in ["sdpo", "opsd"]:
                # Self-distillation: use teacher-conditioned input
                teacher_ids = batch["teacher_input_ids"].to(self.device)
                teacher_mask = batch["teacher_attention_mask"].to(self.device)

                # Use the SAME student model but with different input
                # (conditioned on feedback/answer)
                # Temporarily set to eval mode for teacher pass
                self.student_model.eval()
                teacher_outputs = self.student_model(
                    input_ids=teacher_ids,
                    attention_mask=teacher_mask,
                )
                teacher_logits = teacher_outputs.logits
                self.student_model.train()

                # Align sequence lengths (teacher may be longer due to conditioning)
                min_len = min(student_logits.size(1), teacher_logits.size(1))
                student_logits_aligned = student_logits[:, :min_len, :]
                teacher_logits = teacher_logits[:, :min_len, :]
                loss_mask = loss_mask[:, :min_len]
                input_ids = input_ids[:, :min_len]
            else:
                raise ValueError(f"Unknown method: {self.config.method}")

        # Ensure same shape for Qwen3 method too
        if self.config.method == "qwen3":
            student_logits_aligned = student_logits
        else:
            pass  # already aligned above

        # ── Compute distillation loss ──
        T = self.config.distill_temperature

        if self.config.reward_weighted:
            distill_loss = outcome_weighted_distillation_loss(
                teacher_logits, student_logits_aligned, rewards,
                T, loss_mask, self.config.loss_type,
            )
        elif self.config.loss_type == "kl":
            distill_loss = kl_divergence_loss(
                teacher_logits, student_logits_aligned, T, loss_mask
            )
        elif self.config.loss_type == "reverse_kl":
            distill_loss = reverse_kl_divergence_loss(
                teacher_logits, student_logits_aligned, T, loss_mask
            )
        elif self.config.loss_type == "jsd":
            distill_loss = jsd_divergence_loss(
                teacher_logits, student_logits_aligned, T,
                self.config.jsd_beta, loss_mask,
            )
        elif self.config.loss_type == "full_vocab":
            distill_loss = full_vocab_distillation_loss(
                teacher_logits, student_logits_aligned, T, loss_mask,
            )
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        # ── Auxiliary SFT loss (optional) ──
        sft_loss_val = torch.tensor(0.0, device=self.device)
        if self.config.alpha_sft > 0:
            labels = input_ids.clone()
            labels[loss_mask[:, :labels.size(1)] == 0] = -100
            sft_loss_val = sft_loss(
                student_logits[:, :labels.size(1), :], labels, loss_mask[:, :labels.size(1)]
            )

        # ── Total loss ──
        total_loss = (
            self.config.alpha_distill * distill_loss
            + self.config.alpha_sft * sft_loss_val
        )

        return {
            "loss": total_loss,
            "distill_loss": distill_loss.item(),
            "sft_loss": sft_loss_val.item(),
            "total_loss": total_loss.item(),
        }

    # ------------------------------------------------------------------
    # Rollout (on-policy data collection)
    # ------------------------------------------------------------------

    def collect_rollout_data(self, iteration: int) -> List[Dict]:
        """
        Collect on-policy rollout data from DSGym environment.

        Uses student_rollout.py to generate trajectories, then loads the results.
        """
        import subprocess

        rollout_dir = os.path.join(self.config.output_dir, f"iter_{iteration}", "rollout")
        os.makedirs(rollout_dir, exist_ok=True)

        # Determine model path (initial or checkpoint)
        if iteration == 0:
            model_path = self.config.student_model
        else:
            prev_ckpt = os.path.join(
                self.config.output_dir, f"iter_{iteration - 1}", "checkpoint"
            )
            model_path = prev_ckpt if os.path.isdir(prev_ckpt) else self.config.student_model

        script_dir = Path(__file__).parent
        cmd = [
            sys.executable, str(script_dir / "student_rollout.py"),
            "--model", model_path,
            "--backend", "vllm",
            "--dataset", self.config.dataset,
            "--k", str(self.config.k),
            "--temperature", str(self.config.temperature),
            "--max-turns", str(self.config.max_turns),
            "--max-workers", str(self.config.max_workers),
            "--manager-url", self.config.manager_url,
            "--output-dir", rollout_dir,
            "--run-name", f"logit_distill_iter{iteration}",
        ]
        if self.config.dataset_limit:
            cmd += ["--limit", str(self.config.dataset_limit)]

        print(f"\n  Collecting on-policy rollout data (iter {iteration}) …")
        subprocess.run(cmd, check=True)

        # Load trajectories
        trajs = self._load_trajectories(rollout_dir)
        print(f"  Loaded {len(trajs)} trajectories from rollout")
        return trajs

    def _load_trajectories(self, rollout_dir: str) -> List[Dict]:
        """Load trajectory JSON files."""
        trajs = []
        for subdir in ["predictions", "correct", "incorrect", ""]:
            d = Path(rollout_dir) / subdir if subdir else Path(rollout_dir)
            if not d.exists():
                continue
            for fp in sorted(d.glob("*.json")):
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

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        """Main training loop: alternating rollout and distillation."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Save config
        with open(os.path.join(self.config.output_dir, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Setup wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            run_name = self.config.run_name or f"{self.config.method}_{self.config.dataset}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config=asdict(self.config),
            )

        iteration_results = []

        for iteration in range(self.config.num_iterations):
            print(f"\n{'#' * 70}")
            print(f"#  ITERATION {iteration}/{self.config.num_iterations}")
            print(f"{'#' * 70}\n")

            iter_dir = os.path.join(self.config.output_dir, f"iter_{iteration}")
            os.makedirs(iter_dir, exist_ok=True)

            # ── Step 1: On-policy rollout ──
            t0 = time.time()
            trajectories = self.collect_rollout_data(iteration)
            rollout_time = time.time() - t0

            if not trajectories:
                print("  ⚠️  No trajectories collected; skipping iteration.")
                continue

            # Count correct trajectories
            n_correct = sum(
                1 for t in trajectories
                if any(
                    (t.get("metrics", {}).get(m, {}).get("score", 0) if isinstance(t.get("metrics", {}).get(m, {}), dict) else 0) >= 0.99
                    for m in ["exact_match", "fuzzy_exact_match"]
                )
            )
            print(f"  Trajectories: {len(trajectories)} total, {n_correct} correct")

            # ── Step 2: Setup models (reload for each iteration to get fresh rollout model) ──
            if iteration == 0:
                self.setup_models()

            # ── Step 3: Create dataset ──
            dataset = DistillationDataset(
                trajectories, self.tokenizer,
                max_length=self.config.max_seq_length,
                method=self.config.method,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=True,
            )

            print(f"  Dataset: {len(dataset)} items, "
                  f"{len(dataloader)} batches per epoch")

            # ── Step 4: Setup optimizer ──
            num_steps = len(dataloader) * self.config.inner_epochs
            self.setup_optimizer(num_steps)

            # Move student to device
            if not self.config.use_deepspeed:
                self.student_model = self.student_model.to(self.device)
                if self.teacher_model is not None:
                    self.teacher_model = self.teacher_model.to(self.device)

            # ── Step 5: Inner training loop ──
            t1 = time.time()
            global_step = 0
            epoch_losses = []

            for epoch in range(self.config.inner_epochs):
                print(f"\n  --- Epoch {epoch}/{self.config.inner_epochs} ---")
                epoch_loss = 0.0
                epoch_distill = 0.0
                epoch_sft = 0.0
                n_batches = 0

                for batch_idx, batch in enumerate(dataloader):
                    loss_dict = self.distillation_step(batch)

                    loss = loss_dict["loss"]
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()

                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.student_model.parameters(),
                            self.config.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1

                    epoch_loss += loss_dict["total_loss"]
                    epoch_distill += loss_dict["distill_loss"]
                    epoch_sft += loss_dict["sft_loss"]
                    n_batches += 1

                    if (batch_idx + 1) % self.config.log_interval == 0:
                        avg_loss = epoch_loss / n_batches
                        avg_distill = epoch_distill / n_batches
                        lr = self.scheduler.get_last_lr()[0]
                        print(
                            f"    step {global_step} | "
                            f"loss={avg_loss:.4f} | "
                            f"distill={avg_distill:.4f} | "
                            f"lr={lr:.2e}"
                        )

                        if self.config.use_wandb and WANDB_AVAILABLE:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/distill_loss": avg_distill,
                                "train/sft_loss": epoch_sft / n_batches,
                                "train/lr": lr,
                                "train/iteration": iteration,
                                "train/epoch": epoch,
                                "train/global_step": global_step,
                            })

                avg_epoch_loss = epoch_loss / max(n_batches, 1)
                epoch_losses.append(avg_epoch_loss)
                print(f"  Epoch {epoch} complete. Avg loss: {avg_epoch_loss:.4f}")

            train_time = time.time() - t1

            # ── Step 6: Save checkpoint ──
            ckpt_dir = os.path.join(iter_dir, "checkpoint")
            print(f"\n  Saving checkpoint → {ckpt_dir}")
            self.student_model.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)

            # ── Step 7: Record results ──
            result = {
                "iteration": iteration,
                "n_trajectories": len(trajectories),
                "n_correct": n_correct,
                "pass_rate": n_correct / max(len(trajectories), 1),
                "avg_loss": float(sum(epoch_losses) / len(epoch_losses)),
                "rollout_time_sec": rollout_time,
                "train_time_sec": train_time,
                "checkpoint": ckpt_dir,
            }
            iteration_results.append(result)

            # Save progress
            with open(os.path.join(self.config.output_dir, "results.json"), "w") as f:
                json.dump(iteration_results, f, indent=2)

            print(f"\n  Iteration {iteration} complete:")
            print(f"    Pass rate:    {result['pass_rate']:.3f}")
            print(f"    Avg loss:     {result['avg_loss']:.4f}")
            print(f"    Rollout time: {rollout_time:.0f}s")
            print(f"    Train time:   {train_time:.0f}s")

        # Final summary
        self._print_final_summary(iteration_results)

        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()

    def _print_final_summary(self, results: List[Dict]):
        print(f"\n\n{'=' * 70}")
        print("  LOGIT DISTILLATION — FINAL SUMMARY")
        print(f"  Method: {self.config.method.upper()}")
        print(f"{'=' * 70}")
        header = f"{'Iter':>4} {'Trajs':>6} {'Correct':>7} {'Pass%':>7} {'Loss':>8} {'Time':>8}"
        print(f"  {header}")
        print(f"  {'─' * 50}")
        for r in results:
            print(
                f"  {r['iteration']:>4} {r['n_trajectories']:>6} "
                f"{r['n_correct']:>7} {r['pass_rate']:>7.3f} "
                f"{r['avg_loss']:>8.4f} {r['train_time_sec']:>7.0f}s"
            )
        if results:
            best = max(results, key=lambda r: r["pass_rate"])
            print(f"\n  🏆 Best: iter {best['iteration']} (pass={best['pass_rate']:.3f})")
        print(f"{'=' * 70}\n")


# ============================================================================
# CLI
# ============================================================================

def create_parser():
    p = argparse.ArgumentParser(
        description="Logit-Level On-Policy Distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  qwen3   External teacher logit distillation (Qwen3 Technical Report)
  sdpo    Self-Distillation with execution feedback (Shenfeld et al. 2026)
  opsd    On-Policy Self-Distillation with answer conditioning (Zhao et al. 2026)

Loss Types:
  kl          Forward KL(teacher || student) — standard, mode-covering
  reverse_kl  Reverse KL(student || teacher) — mode-seeking
  jsd         Jensen-Shannon Divergence — symmetric, bounded, used in SDPO
  full_vocab  Combined forward+reverse KL — used in OPSD

Examples:
  # Qwen3-style (need separate teacher model)
  python logit_distillation.py --method qwen3 \\
      --student-model Qwen/Qwen2.5-Coder-7B-Instruct \\
      --teacher-model Qwen/Qwen3-14B-Instruct \\
      --loss-type kl --distill-temperature 2.0

  # SDPO (self-distillation, no external teacher)
  python logit_distillation.py --method sdpo \\
      --student-model Qwen/Qwen2.5-Coder-7B-Instruct \\
      --loss-type jsd --distill-temperature 1.0

  # OPSD (self-distillation with answer conditioning)
  python logit_distillation.py --method opsd \\
      --student-model Qwen/Qwen2.5-Coder-7B-Instruct \\
      --loss-type full_vocab --distill-temperature 2.0
        """,
    )

    # Method
    p.add_argument("--method", type=str, required=True,
                   choices=["qwen3", "sdpo", "opsd"])

    # Models
    p.add_argument("--student-model", type=str, required=True)
    p.add_argument("--teacher-model", type=str, default="",
                   help="External teacher model (required for qwen3)")

    # LoRA
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)

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
    p.add_argument("--loss-type", type=str, default="kl",
                   choices=["kl", "reverse_kl", "jsd", "full_vocab"])
    p.add_argument("--jsd-beta", type=float, default=0.5)
    p.add_argument("--reward-weighted", action="store_true")

    # Rollout
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--max-workers", type=int, default=24)
    p.add_argument("--manager-url", type=str, default="http://localhost:5000")

    # Dataset
    p.add_argument("--dataset", type=str, default="qrdata")
    p.add_argument("--dataset-limit", type=int, default=None)

    # Paths
    p.add_argument("--output-dir", type=str, default="./logit_distill_outputs")

    # DeepSpeed
    p.add_argument("--use-deepspeed", action="store_true", default=False)
    p.add_argument("--deepspeed-stage", type=int, default=2)

    # Logging
    p.add_argument("--use-wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="dsgym-logit-distill")
    p.add_argument("--run-name", type=str, default=None)

    # Local-rank for distributed (DeepSpeed / torchrun)
    p.add_argument("--local_rank", type=int, default=-1)

    return p


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = LogitDistillConfig(
        method=args.method,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_iterations=args.num_iterations,
        inner_epochs=args.inner_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        distill_temperature=args.distill_temperature,
        alpha_distill=args.alpha_distill,
        alpha_sft=args.alpha_sft,
        loss_type=args.loss_type,
        jsd_beta=args.jsd_beta,
        reward_weighted=args.reward_weighted,
        k=args.k,
        temperature=args.temperature,
        max_turns=args.max_turns,
        max_workers=args.max_workers,
        manager_url=args.manager_url,
        dataset=args.dataset,
        dataset_limit=args.dataset_limit,
        output_dir=args.output_dir,
        use_deepspeed=args.use_deepspeed,
        deepspeed_stage=args.deepspeed_stage,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )

    # Set method-specific defaults
    if config.method == "sdpo" and args.loss_type == "kl":
        config.loss_type = "jsd"  # SDPO default
        print(f"  ℹ️  SDPO: auto-setting loss_type=jsd")

    if config.method == "opsd" and args.loss_type == "kl":
        config.loss_type = "full_vocab"  # OPSD default
        print(f"  ℹ️  OPSD: auto-setting loss_type=full_vocab")

    trainer = LogitDistillationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
