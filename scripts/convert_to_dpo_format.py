#!/usr/bin/env python3
"""
Convert DSGym trajectories to DPO/Preference format for preference learning.

This script creates preference pairs from trajectory data:
- chosen: trajectories with correct answers / high scores
- rejected: trajectories with wrong answers / low scores

Usage:
    python convert_to_dpo_format.py \
        --input-dir ./trajectory_results/qrdata_teacher_k5 \
        --output-file /data/fnie/LLaMA-Factory/data/qrdata_dpo.json
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from tqdm import tqdm
from collections import defaultdict


def load_trajectory(file_path: str) -> Dict[str, Any]:
    """Load a single trajectory file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_sample_key(trajectory: Dict[str, Any], file_name: str) -> str:
    """Extract unique sample identifier from trajectory or filename."""
    sample_id = trajectory.get("sample_id", "")
    if sample_id:
        return re.sub(r'_traj_\d+$', '', sample_id)
    
    match = re.search(r'(?:prediction|conversation_sample)_(\d+)', file_name)
    if match:
        return f"sample_{match.group(1)}"
    
    return file_name


def get_trajectory_score(trajectory: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Get trajectory correctness and score.
    
    Returns:
        Tuple of (is_correct, score)
    """
    metrics = trajectory.get("metrics", {})
    
    # Check various correctness metrics
    for metric_name in ["exact_match", "fuzzy_exact_match", "list_match", 
                        "equivalence_by_llm", "fast_equivalence_by_llm"]:
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            score = metric_value.get("score", 0) if isinstance(metric_value, dict) else metric_value
            return score >= 0.99, score
    
    # Check judge_score if available
    judge_score = trajectory.get("judge_score", trajectory.get("self_judge_score"))
    if judge_score is not None:
        return judge_score >= 0.6, judge_score
    
    # Fallback to success field
    if trajectory.get("success", False):
        return True, 1.0
    
    return False, 0.0


def extract_conversation_parts(trajectory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract system prompt, user query, and assistant response from trajectory.
    
    Returns:
        Dict with 'system', 'user_query', and 'full_response' keys
    """
    conversation = (
        trajectory.get("conversation") or 
        trajectory.get("trajectory") or 
        []
    )
    
    if not conversation:
        return None
    
    system_prompt = ""
    user_query = ""
    assistant_responses = []
    
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        
        if role == "system" and not system_prompt:
            system_prompt = content
        elif role == "user" and not user_query:
            user_query = content
        elif role == "assistant":
            assistant_responses.append(content)
    
    if not user_query:
        return None
    
    # Combine all assistant responses into full response
    full_response = "\n\n".join(assistant_responses) if assistant_responses else ""
    
    return {
        "system": system_prompt,
        "user_query": user_query,
        "full_response": full_response
    }


def create_dpo_pair(
    chosen_traj: Dict[str, Any],
    rejected_traj: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Create a DPO preference pair from two trajectories.
    
    Returns:
        DPO format dict or None if extraction fails
    """
    chosen_parts = extract_conversation_parts(chosen_traj)
    rejected_parts = extract_conversation_parts(rejected_traj)
    
    if not chosen_parts or not rejected_parts:
        return None
    
    if not chosen_parts["full_response"] or not rejected_parts["full_response"]:
        return None
    
    # Use chosen trajectory's system prompt and user query
    messages = []
    
    if chosen_parts["system"]:
        messages.append({
            "role": "system",
            "content": chosen_parts["system"]
        })
    
    messages.append({
        "role": "user",
        "content": chosen_parts["user_query"]
    })
    
    return {
        "messages": messages,
        "chosen": {
            "role": "assistant",
            "content": chosen_parts["full_response"]
        },
        "rejected": {
            "role": "assistant",
            "content": rejected_parts["full_response"]
        }
    }


def convert_to_dpo(
    input_dir: str,
    output_file: str,
    min_score_diff: float = 0.3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Convert trajectories to DPO format by creating preference pairs.
    
    Args:
        input_dir: Directory containing trajectory JSON files
        output_file: Output path for DPO data
        min_score_diff: Minimum score difference between chosen and rejected
        verbose: Print detailed information
    
    Returns:
        Dictionary with conversion statistics
    """
    input_path = Path(input_dir)
    
    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        json_files = list(input_path.glob("**/*.json"))
    
    print(f"Found {len(json_files)} trajectory files")
    
    # Group trajectories by sample
    sample_trajectories = defaultdict(list)
    
    stats = {
        "total_files": len(json_files),
        "loaded": 0,
        "unique_samples": 0,
        "samples_with_pairs": 0,
        "dpo_pairs_created": 0,
        "errors": 0
    }
    
    # Load and group trajectories
    for file_path in tqdm(json_files, desc="Loading trajectories"):
        try:
            trajectory = load_trajectory(str(file_path))
            stats["loaded"] += 1
            
            is_correct, score = get_trajectory_score(trajectory)
            sample_key = extract_sample_key(trajectory, file_path.name)
            
            sample_trajectories[sample_key].append({
                "trajectory": trajectory,
                "is_correct": is_correct,
                "score": score,
                "file_name": file_path.name
            })
            
        except Exception as e:
            if verbose:
                print(f"Error loading {file_path}: {e}")
            stats["errors"] += 1
    
    stats["unique_samples"] = len(sample_trajectories)
    
    # Create DPO pairs
    dpo_data = []
    
    for sample_key, trajectories in tqdm(sample_trajectories.items(), desc="Creating DPO pairs"):
        if len(trajectories) < 2:
            continue
        
        # Separate by correctness
        correct_trajs = [t for t in trajectories if t["is_correct"]]
        incorrect_trajs = [t for t in trajectories if not t["is_correct"]]
        
        # Also consider score-based pairs
        trajectories.sort(key=lambda x: x["score"], reverse=True)
        
        pairs_created = 0
        
        # Strategy 1: Pair correct with incorrect
        if correct_trajs and incorrect_trajs:
            # Use best correct and worst incorrect
            best_correct = max(correct_trajs, key=lambda x: x["score"])
            worst_incorrect = min(incorrect_trajs, key=lambda x: x["score"])
            
            dpo_pair = create_dpo_pair(
                best_correct["trajectory"],
                worst_incorrect["trajectory"]
            )
            
            if dpo_pair:
                dpo_data.append(dpo_pair)
                pairs_created += 1
        
        # Strategy 2: Pair by score difference (even within correct/incorrect)
        if len(trajectories) >= 2:
            best = trajectories[0]
            worst = trajectories[-1]
            
            if best["score"] - worst["score"] >= min_score_diff:
                dpo_pair = create_dpo_pair(
                    best["trajectory"],
                    worst["trajectory"]
                )
                
                if dpo_pair and pairs_created == 0:
                    dpo_data.append(dpo_pair)
                    pairs_created += 1
        
        if pairs_created > 0:
            stats["samples_with_pairs"] += 1
            stats["dpo_pairs_created"] += pairs_created
    
    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("DPO Conversion Summary")
    print(f"{'='*50}")
    print(f"Total files found:      {stats['total_files']}")
    print(f"Successfully loaded:    {stats['loaded']}")
    print(f"Unique samples:         {stats['unique_samples']}")
    print(f"Samples with pairs:     {stats['samples_with_pairs']}")
    print(f"DPO pairs created:      {stats['dpo_pairs_created']}")
    print(f"Errors:                 {stats['errors']}")
    print(f"{'='*50}")
    print(f"Output saved to: {output_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert DSGym trajectories to DPO preference format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic DPO conversion
  python convert_to_dpo_format.py \\
      --input-dir ./trajectory_results \\
      --output-file ./dpo_data.json

  # Require larger score difference for pairs
  python convert_to_dpo_format.py \\
      --input-dir ./trajectory_results \\
      --output-file ./dpo_data.json \\
      --min-score-diff 0.5
        """
    )
    
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Input directory containing trajectory JSON files"
    )
    parser.add_argument(
        "--output-file", "-o", required=True,
        help="Output JSON file path for DPO data"
    )
    parser.add_argument(
        "--min-score-diff", type=float, default=0.3,
        help="Minimum score difference for creating pairs (default: 0.3)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed error information"
    )
    
    args = parser.parse_args()
    
    convert_to_dpo(
        input_dir=args.input_dir,
        output_file=args.output_file,
        min_score_diff=args.min_score_diff,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
