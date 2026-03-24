"""
TeacherAgent for structured trajectory data collection.

Extends DSPredictReActAgent to generate structured reasoning trajectories
using a frontier LLM (Claude/GPT/Qwen-235B) as the teacher model.

Key additions over DSPredictReActAgent:
1. Teacher system prompt instructing XML structured output
2. Search state tracking across turns (best_score, baseline, history)
3. Auto-detection of exploration -> optimization phase transition
4. Score parsing from execution output (validation scores printed by model)
5. Structured trajectory recording for SFT data generation
"""

import os
import re
import json
import time
import copy
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

from dsgym.agents.dspredict_react_agent import DSPredictReActAgent
from dsgym.agents.environment import AllocatedCodeEnv
from .structured_output import (
    parse_structured_output,
    build_search_state_xml,
    Attempt,
    StructuredOutput,
)
from .prompts import (
    TEACHER_SYSTEM_PROMPT_EXPLORATION,
    TEACHER_SYSTEM_PROMPT_OPTIMIZATION,
)


# Regex patterns for detecting validation scores in execution output.
# Ordered from most specific to least specific; first match wins.
SCORE_PATTERNS = [
    # "Mean CV RMSLE: 0.125" — common format from cross-validation
    re.compile(r"(?:Mean|Avg|Average)?\s*(?:Validation|Val|CV|Test|Best)?\s*(?:Score|RMSE|RMSLE|MAE|MSE|AUC|ROC.AUC|F1|Accuracy|R2|R²|Log\s*Loss|Logloss|MCC|QWK|MAP)\s*(?:\(mean\))?\s*[:=]\s*([-+]?\d+\.?\d*(?:e[+-]?\d+)?)", re.IGNORECASE),
    # "Best Score: 0.123" pattern
    re.compile(r"(?:Best\s+)?Score\s*[:=]\s*([-+]?\d+\.?\d*(?:e[+-]?\d+)?)", re.IGNORECASE),
    # Bare metric name: "RMSE: 12345.6"
    re.compile(r"(?:RMSE|RMSLE|MAE|MSE|AUC|ROC.AUC|F1|Accuracy|R2|R²|Log\s*Loss|Logloss|MCC|QWK|MAP)\s*[:=]\s*([-+]?\d+\.?\d*(?:e[+-]?\d+)?)", re.IGNORECASE),
]


@dataclass
class TurnRecord:
    """Record of a single turn for trajectory data."""
    turn: int
    phase: str
    raw_response: str
    parsed_output: Optional[dict]
    execution_output: str
    score: Optional[float]
    score_delta: Optional[float]
    predicted_delta: Optional[float]
    parse_success: bool
    step_time: float


@dataclass
class StructuredTrajectory:
    """Complete structured trajectory for one task run."""
    task_id: str
    challenge_name: str
    model: str
    temperature: float
    turns: List[TurnRecord] = field(default_factory=list)
    final_best_score: Optional[float] = None
    baseline_score: Optional[float] = None
    total_time: float = 0.0
    success: bool = False
    conversation: List[Dict[str, str]] = field(default_factory=list)


class TeacherAgent(DSPredictReActAgent):
    """
    Teacher agent that wraps a frontier LLM to generate structured trajectories.

    Extends DSPredictReActAgent with:
    1. Teacher system prompt instructing XML structured output
    2. Search state tracking across turns
    3. Phase transition detection (exploration -> optimization)
    4. Score parsing from execution output
    5. Structured trajectory recording for SFT data
    """

    def __init__(self, backend: str, model: str, **kwargs):
        self.trajectory_output_dir = kwargs.pop(
            "trajectory_output_dir", "./teacher_trajectories"
        )
        super().__init__(backend, model, **kwargs)
        os.makedirs(self.trajectory_output_dir, exist_ok=True)

    def solve_task(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Solve task with structured reasoning, recording full trajectory.

        Overrides DSPredictReActAgent.solve_task() to:
        1. Prepend teacher system prompt
        2. Track search state and inject updates each turn
        3. Detect phase transitions
        4. Parse scores from execution output
        5. Record structured trajectory data
        """
        start_time = time.time()

        # Search state
        current_phase = "exploration"
        current_step = 0
        best_score: Optional[float] = None
        baseline_score: Optional[float] = None
        history: List[Attempt] = []
        metric_direction: Optional[str] = None  # "lower" or "higher"

        # Trajectory
        turn_records: List[TurnRecord] = []
        trajectory: List[Dict[str, Any]] = []

        try:
            conversation = sample.get("prompt", [])
            if not conversation:
                raise ValueError("Sample must contain 'prompt' field")

            extras = {
                "reward_spec": sample.get("reward_spec", {"ground_truth": ""}),
                "extra_info": sample.get("extra_info", {}),
                "max_turns": self.max_turns,
            }

            env = AllocatedCodeEnv(
                manager_url=self.manager_url,
                max_turns=self.max_turns,
                output_dir=self.output_dir,
                time_out=self.time_out,
            )

            conversation, _ = env.init(conversation, **extras)
            container_id = env.tool_group.allocated_container

            # Detect metric direction from task description
            metric_direction = self._detect_metric_direction(sample)

            # Inject initial teacher system prompt
            conversation = self._inject_system_prompt(conversation, current_phase)

            total_tokens = 0
            final_answer = ""
            actual_turns = 0

            for turn in range(self.max_turns):
                current_step = turn + 1
                turn_start = time.time()

                try:
                    # Generate response from teacher model
                    response = self.backend_instance.generate(conversation)
                    total_tokens += len(response.split())
                    actual_turns = turn + 1

                    # Try to parse structured output
                    parsed: Optional[StructuredOutput] = None
                    parse_success = False
                    try:
                        parsed = parse_structured_output(response)
                        parse_success = True
                    except (ValueError, Exception) as e:
                        print(
                            f"  Warning: XML parse failed at turn {current_step}: {e}"
                        )

                    # Step environment (extracts <python> and executes)
                    step_output = env.step(response)
                    step_time = time.time() - turn_start

                    # Parse score from execution output
                    exec_output = step_output.get("metadata", {}).get(
                        "execution_output", ""
                    )
                    detected_score = self._parse_score_from_output(exec_output)

                    # Update search state
                    score_delta: Optional[float] = None
                    predicted_delta: Optional[float] = None

                    if detected_score is not None:
                        if baseline_score is None:
                            baseline_score = detected_score
                            best_score = detected_score
                            if current_phase == "exploration":
                                current_phase = "optimization"
                                conversation = self._update_system_prompt(
                                    conversation, "optimization"
                                )
                                print(
                                    f"  Phase transition -> optimization "
                                    f"(baseline={baseline_score})"
                                )
                        else:
                            score_delta = detected_score - best_score
                            if self._is_new_best(
                                detected_score, best_score, metric_direction
                            ):
                                best_score = detected_score

                        # Record attempt in history
                        if parsed and parsed.decision and not parsed.is_terminate:
                            predicted_delta = parsed.get_chosen_value_score()
                            history.append(
                                Attempt(
                                    step=current_step,
                                    action=parsed.decision.action,
                                    idea=(
                                        parsed.decision.idea
                                        or parsed.decision.reasoning[:50]
                                    ),
                                    predicted_delta=predicted_delta or 0.0,
                                    actual_delta=score_delta or 0.0,
                                )
                            )

                    # Record turn
                    turn_records.append(
                        TurnRecord(
                            turn=current_step,
                            phase=current_phase,
                            raw_response=response,
                            parsed_output=asdict(parsed) if parsed else None,
                            execution_output=exec_output,
                            score=detected_score,
                            score_delta=score_delta,
                            predicted_delta=predicted_delta,
                            parse_success=parse_success,
                            step_time=step_time,
                        )
                    )

                    # Update conversation
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": step_output.get(
                                "postprocessed_action", response
                            ),
                        }
                    )
                    trajectory = self.append_traj(
                        trajectory,
                        turn,
                        "assistant",
                        response,
                        step_output.get("done", False),
                        step_output.get("reward", 0.0),
                        step_time,
                    )

                    if step_output["observations"]:
                        conversation.extend(step_output["observations"])
                        trajectory = self.append_traj(
                            trajectory,
                            turn,
                            "user",
                            step_output["observations"][0]["content"],
                            step_output.get("done", False),
                            step_output.get("reward", 0.0),
                            step_time,
                        )

                        # Inject search state update for next turn
                        state_update = self._build_state_update(
                            current_phase,
                            current_step,
                            best_score,
                            baseline_score,
                            history,
                        )
                        conversation.append(
                            {"role": "user", "content": state_update}
                        )
                    else:
                        step_output["done"] = True

                    # Check if done
                    if step_output["done"]:
                        final_answer = step_output["metadata"].get(
                            "final_answer", response
                        )
                        break

                except Exception as step_err:
                    error_msg = f"Turn {turn + 1} failed: {step_err}"
                    conversation.append(
                        {
                            "role": "user",
                            "content": f"Error: {error_msg}. Please try a different approach.",
                        }
                    )
                    trajectory = self.append_traj(
                        trajectory, turn, "user", error_msg, False, 0.0, 0.0
                    )
                    continue

            # Save prediction (same as parent)
            if final_answer:
                prefix = sample.get("extra_info", {}).get("id", "temp")
                env.save_prediction(final_answer, filename_prefix=prefix)

            execution_time = time.time() - start_time

            # Handle submission file (same as parent)
            container_dir = os.path.join(
                self.submission_dir, f"container_{container_id:03d}"
            )
            submission_file = os.path.join(container_dir, "submission.csv")
            submission_path = ""
            success = False
            if os.path.exists(submission_file):
                success = True
                challenge_name = sample.get("extra_info", {}).get(
                    "challenge_name", ""
                )
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_name = (
                    f"{challenge_name}_{container_id}_{timestamp}_submission.csv"
                )
                unique_path = os.path.join(container_dir, unique_name)
                import shutil

                shutil.copy2(submission_file, unique_path)
                print(f"Submission file saved: {unique_path}")
                submission_path = unique_path

            # Save structured trajectory
            traj_data = StructuredTrajectory(
                task_id=sample.get("extra_info", {}).get("id", "unknown"),
                challenge_name=sample.get("extra_info", {}).get(
                    "challenge_name", "unknown"
                ),
                model=self.model,
                temperature=getattr(self, "temperature", 0.0),
                turns=turn_records,
                final_best_score=best_score,
                baseline_score=baseline_score,
                total_time=execution_time,
                success=success,
                conversation=conversation,
            )
            self._save_trajectory(traj_data, sample)

            return {
                "solution": submission_path,
                "success": success,
                "turns": actual_turns,
                "error": None,
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "teacher": True,
                    "max_turns": self.max_turns,
                    "total_tokens": total_tokens,
                    "execution_time": execution_time,
                    "conversation_length": len(conversation),
                    "best_score": best_score,
                    "baseline_score": baseline_score,
                    "num_optimization_turns": sum(
                        1 for t in turn_records if t.phase == "optimization"
                    ),
                    "parse_success_rate": (
                        sum(1 for t in turn_records if t.parse_success)
                        / len(turn_records)
                        if turn_records
                        else 0.0
                    ),
                },
                "conversation": conversation,
                "trajectory": trajectory,
                "raw_result": {
                    "prediction": submission_path,
                    "turns": actual_turns,
                    "total_tokens": total_tokens,
                },
            }

        except Exception as e:
            execution_time = time.time() - start_time
            error_trace = traceback.format_exc()
            print(f"Error in TeacherAgent: {error_trace}")

            return {
                "solution": "",
                "success": False,
                "turns": 0,
                "error": str(e),
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "teacher": True,
                    "max_turns": self.max_turns,
                    "execution_time": execution_time,
                    "error_trace": error_trace,
                },
                "conversation": [],
                "trajectory": trajectory,
                "raw_result": None,
            }
        finally:
            if "env" in locals():
                env.close()

    # ================================================================
    # Helper methods
    # ================================================================

    def _inject_system_prompt(
        self, conversation: List[Dict[str, str]], phase: str
    ) -> List[Dict[str, str]]:
        """Prepend teacher system prompt to conversation."""
        prompt = (
            TEACHER_SYSTEM_PROMPT_EXPLORATION
            if phase == "exploration"
            else TEACHER_SYSTEM_PROMPT_OPTIMIZATION
        )

        # Check if system message already exists
        if conversation and conversation[0].get("role") == "system":
            # Store original system content for later prompt updates
            self._original_system_content = conversation[0]["content"]
            conversation[0]["content"] = prompt + "\n\n" + conversation[0]["content"]
        else:
            self._original_system_content = ""
            conversation.insert(0, {"role": "system", "content": prompt})

        return conversation

    def _update_system_prompt(
        self, conversation: List[Dict[str, str]], phase: str
    ) -> List[Dict[str, str]]:
        """Replace teacher system prompt when phase changes."""
        prompt = (
            TEACHER_SYSTEM_PROMPT_EXPLORATION
            if phase == "exploration"
            else TEACHER_SYSTEM_PROMPT_OPTIMIZATION
        )

        original = getattr(self, "_original_system_content", "")
        if conversation and conversation[0].get("role") == "system":
            conversation[0]["content"] = prompt + "\n\n" + original
        else:
            conversation.insert(0, {"role": "system", "content": prompt})

        return conversation

    def _build_state_update(
        self,
        phase: str,
        step: int,
        best_score: Optional[float],
        baseline_score: Optional[float],
        history: List[Attempt],
    ) -> str:
        """Build a search state update message to inject between turns."""
        state_xml = build_search_state_xml(
            phase=phase,
            step=step + 1,  # next step number
            best_score=best_score,
            baseline_score=baseline_score,
            history=history if phase == "optimization" else None,
            goal=None,
        )

        if phase == "exploration":
            return (
                f"[Search State Update] You are in the exploration phase, "
                f"step {step + 1}. Continue exploring the data and building "
                f"a baseline model. Remember to print validation scores."
            )
        else:
            return (
                f"[Search State Update] Current search state for your next response:\n\n"
                f"{state_xml}\n\n"
                f"IMPORTANT: Your response MUST contain ALL of these blocks in order:\n"
                f"1. <search_state> — copy the state above, update goal\n"
                f"2. <candidates> — propose 3 diverse ideas (A, B, C)\n"
                f"3. <value_estimation> — estimate predicted_delta for all 5 actions "
                f"(explore A, explore B, explore C, exploit, terminate)\n"
                f"4. <decision> — choose which action to take and why\n"
                f"5. <python> — implement the chosen action's code "
                f"(omit only if decision is terminate)\n\n"
                f"Do NOT stop after <candidates>. You MUST continue with "
                f"<value_estimation>, <decision>, and <python>."
            )

    @staticmethod
    def _parse_score_from_output(output: str) -> Optional[float]:
        """
        Parse validation score from execution output.

        Tries multiple regex patterns. Returns the match at the highest
        text position (i.e., the last score printed by the model).
        """
        if not output:
            return None

        best_pos = -1
        best_score = None
        for pattern in SCORE_PATTERNS:
            for match in pattern.finditer(output):
                try:
                    score = float(match.group(1))
                    if match.end() > best_pos:
                        best_pos = match.end()
                        best_score = score
                except (ValueError, TypeError):
                    continue

        return best_score

    @staticmethod
    def _detect_metric_direction(sample: Dict[str, Any]) -> str:
        """
        Detect whether lower or higher scores are better.

        Uses heuristics from the task description.
        """
        query = sample.get("extra_info", {}).get("query", "")
        prompt_text = ""
        for msg in sample.get("prompt", []):
            prompt_text += msg.get("content", "")

        combined = (query + " " + prompt_text).lower()

        higher_is_better_keywords = [
            "accuracy",
            "auc",
            "f1",
            "precision",
            "recall",
            "r2",
            "r²",
            "mcc",
            "matthews",
            "cohen",
            "kappa",
            "map@",
            "ndcg",
        ]
        lower_is_better_keywords = [
            "rmse",
            "rmsle",
            "mae",
            "mse",
            "log loss",
            "logloss",
            "error",
            "loss",
        ]

        # Check lower_is_better first — metric-specific names (RMSE, MAE)
        # are more definitive than generic words (accuracy) that often appear
        # in data field descriptions.
        for kw in lower_is_better_keywords:
            if kw in combined:
                return "lower"
        for kw in higher_is_better_keywords:
            if kw in combined:
                return "higher"

        return "lower"  # default

    @staticmethod
    def _is_new_best(
        new_score: float,
        current_best: float,
        metric_direction: str,
    ) -> bool:
        """Check if new score is better than current best."""
        if metric_direction == "higher":
            return new_score > current_best
        else:
            return new_score < current_best

    def _save_trajectory(
        self, traj: StructuredTrajectory, sample: Dict[str, Any]
    ) -> None:
        """Save structured trajectory to JSON."""
        task_id = traj.task_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{task_id}_{timestamp}_trajectory.json"
        filepath = os.path.join(self.trajectory_output_dir, filename)

        data = {
            "task_id": traj.task_id,
            "challenge_name": traj.challenge_name,
            "model": traj.model,
            "temperature": traj.temperature,
            "final_best_score": traj.final_best_score,
            "baseline_score": traj.baseline_score,
            "total_time": traj.total_time,
            "success": traj.success,
            "num_turns": len(traj.turns),
            "turns": [asdict(t) for t in traj.turns],
            "conversation": traj.conversation,
            "sample_extra_info": sample.get("extra_info", {}),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved structured trajectory: {filepath}")
