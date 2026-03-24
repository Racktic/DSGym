"""
Post-hoc trajectory processing for SFT data preparation.

Two main functions:
1. label_terminate_steps() — post-hoc terminate labeling on collected trajectories
2. convert_trajectory_to_sft_samples() — convert trajectories to SFT training format
"""

import json
import os
from typing import List, Dict, Any, Optional


def label_terminate_steps(
    trajectory_data: Dict[str, Any],
    epsilon_ratio: float = 0.005,
) -> Dict[str, Any]:
    """
    Post-hoc terminate labeling on a structured trajectory.

    Finds the peak score step, then labels steps as terminate-positive if:
      (a) their score is within epsilon of the best score, AND
      (b) total improvement after that step < epsilon

    epsilon = epsilon_ratio * |best_score - baseline_score|

    Args:
        trajectory_data: Deserialized trajectory dict (from TeacherAgent._save_trajectory)
        epsilon_ratio: Relative threshold (default 0.5%)

    Returns:
        trajectory_data with added 'terminate_label' field per turn
    """
    turns = trajectory_data.get("turns", [])
    baseline = trajectory_data.get("baseline_score")
    best = trajectory_data.get("final_best_score")

    if baseline is None or best is None:
        for t in turns:
            t["terminate_label"] = False
        return trajectory_data

    epsilon = epsilon_ratio * abs(best - baseline)
    if epsilon == 0:
        epsilon = 1e-6

    # Determine metric direction
    higher_is_better = best > baseline

    # Collect turns with scores
    scored_turns = [
        (i, t) for i, t in enumerate(turns) if t.get("score") is not None
    ]

    if not scored_turns:
        for t in turns:
            t["terminate_label"] = False
        return trajectory_data

    # Find peak step
    if higher_is_better:
        peak_idx, peak_turn = max(scored_turns, key=lambda x: x[1]["score"])
    else:
        peak_idx, peak_turn = min(scored_turns, key=lambda x: x[1]["score"])

    peak_score = peak_turn["score"]

    for i, turn in enumerate(turns):
        score = turn.get("score")
        if score is None:
            turn["terminate_label"] = False
            continue

        # (a) within epsilon of peak
        within_epsilon = abs(score - peak_score) <= epsilon

        # (b) no significant improvement after this step
        future_improvement = 0.0
        for j in range(i + 1, len(turns)):
            future_score = turns[j].get("score")
            if future_score is not None:
                if higher_is_better:
                    future_improvement = max(
                        future_improvement, future_score - score
                    )
                else:
                    future_improvement = max(
                        future_improvement, score - future_score
                    )

        no_future_improvement = future_improvement < epsilon

        turn["terminate_label"] = within_epsilon and no_future_improvement

    return trajectory_data


def convert_trajectory_to_sft_samples(
    trajectory_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Convert a structured trajectory into SFT training samples.

    Each sample is:
      - messages: conversation context up to this turn (input)
      - target: the structured XML output for this turn (output)
      - metadata: phase, score, delta, terminate_label, etc.

    Only includes turns where XML parsing succeeded.

    Args:
        trajectory_data: Deserialized trajectory dict with terminate labels

    Returns:
        List of {"messages": [...], "target": str, "metadata": {...}} dicts
    """
    samples = []
    turns = trajectory_data.get("turns", [])
    conversation = trajectory_data.get("conversation", [])

    # Build a mapping of turn index to conversation position
    # Each turn produces: assistant message + user observation (+ optional state update)
    conv_positions = _map_turns_to_conversation(turns, conversation)

    for i, turn in enumerate(turns):
        if not turn.get("parse_success", False):
            continue

        # Get conversation context up to this turn
        conv_end = conv_positions.get(i, 0)
        context_messages = conversation[:conv_end] if conv_end > 0 else []

        sample = {
            "messages": context_messages,
            "target": turn["raw_response"],
            "metadata": {
                "phase": turn["phase"],
                "step": turn["turn"],
                "score": turn.get("score"),
                "score_delta": turn.get("score_delta"),
                "predicted_delta": turn.get("predicted_delta"),
                "terminate_label": turn.get("terminate_label", False),
                "task_id": trajectory_data.get("task_id"),
                "challenge_name": trajectory_data.get("challenge_name"),
                "model": trajectory_data.get("model"),
            },
        }

        samples.append(sample)

    return samples


def _map_turns_to_conversation(
    turns: List[Dict[str, Any]],
    conversation: List[Dict[str, str]],
) -> Dict[int, int]:
    """
    Map turn indices to conversation positions.

    Returns dict mapping turn_index -> conversation_index where
    the assistant message for that turn starts.
    """
    mapping = {}
    # Skip system message(s) and initial user prompt
    conv_idx = 0
    for msg in conversation:
        if msg.get("role") in ("system", "user"):
            conv_idx += 1
        else:
            break

    for turn_idx in range(len(turns)):
        mapping[turn_idx] = conv_idx
        # Each turn: assistant message
        conv_idx += 1
        # Then observation(s) and possible state update
        while conv_idx < len(conversation) and conversation[conv_idx].get("role") == "user":
            conv_idx += 1

    return mapping


def load_trajectory(filepath: str) -> Dict[str, Any]:
    """Load a trajectory JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def process_trajectory_dir(
    input_dir: str,
    output_path: str,
    epsilon_ratio: float = 0.005,
) -> Dict[str, Any]:
    """
    Process all trajectories in a directory: label terminates, convert to SFT format.

    Args:
        input_dir: Directory containing trajectory JSON files
        output_path: Path to save the combined SFT dataset (JSONL)
        epsilon_ratio: Relative threshold for terminate labeling

    Returns:
        Stats dict with counts
    """
    all_samples = []
    stats = {
        "total_trajectories": 0,
        "total_turns": 0,
        "total_sft_samples": 0,
        "terminate_positive_count": 0,
        "exploration_count": 0,
        "optimization_count": 0,
    }

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith("_trajectory.json"):
            continue

        filepath = os.path.join(input_dir, filename)
        traj = load_trajectory(filepath)

        # Label terminate steps
        traj = label_terminate_steps(traj, epsilon_ratio)
        stats["total_trajectories"] += 1
        stats["total_turns"] += len(traj.get("turns", []))

        # Convert to SFT samples
        samples = convert_trajectory_to_sft_samples(traj)
        all_samples.extend(samples)

        for s in samples:
            meta = s.get("metadata", {})
            stats["total_sft_samples"] += 1
            if meta.get("terminate_label"):
                stats["terminate_positive_count"] += 1
            if meta.get("phase") == "exploration":
                stats["exploration_count"] += 1
            else:
                stats["optimization_count"] += 1

    # Save as JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if stats["total_sft_samples"] > 0:
        stats["terminate_rate"] = (
            stats["terminate_positive_count"] / stats["total_sft_samples"]
        )
    else:
        stats["terminate_rate"] = 0.0

    print(f"Processed {stats['total_trajectories']} trajectories")
    print(f"Generated {stats['total_sft_samples']} SFT samples")
    print(
        f"Terminate rate: {stats['terminate_rate']:.1%} "
        f"({stats['terminate_positive_count']}/{stats['total_sft_samples']})"
    )

    return stats
