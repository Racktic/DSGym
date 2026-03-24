"""
EETAgent — Explore / Exploit / Terminate structured agent.

A simplified structured agent that adds explicit explore/exploit/terminate
decision-making over vanilla ReAct, without candidates or value estimation.

Every turn outputs: search_state + decision + python (3 blocks).
No phase transitions — single prompt throughout.
"""

import os
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

from dsgym.agents.dspredict_react_agent import DSPredictReActAgent
from dsgym.agents.environment import AllocatedCodeEnv
from .structured_output import (
    parse_eet_output,
    StructuredOutput,
)
from .eet_prompts import (
    EET_SYSTEM_PROMPT,
    EET_SYSTEM_PROMPT_NO_TERMINATE,
    EET_SYSTEM_PROMPT_SELF_CONTAINED,
    EET_SYSTEM_PROMPT_NO_TERMINATE_SELF_CONTAINED,
)
from .teacher_agent import (
    TurnRecord,
    StructuredTrajectory,
)
from .memory import CrossTaskMemory


class EETAgent(DSPredictReActAgent):
    """
    Explore-Exploit-Terminate agent with simplified structured reasoning.

    Every turn: search_state + decision (explore/exploit/terminate) + python.
    No candidates, no value estimation, no phase transitions.
    """

    def __init__(self, backend: str, model: str, **kwargs):
        self.trajectory_output_dir = kwargs.pop(
            "trajectory_output_dir", "./eet_trajectories"
        )
        self.no_terminate = kwargs.pop("no_terminate", False)
        self.self_contained = kwargs.pop("self_contained", False)
        memory_path = kwargs.pop(
            "memory_path", "./cross_task_memory.json"
        )
        super().__init__(backend, model, **kwargs)
        os.makedirs(self.trajectory_output_dir, exist_ok=True)
        self.cross_task_memory = CrossTaskMemory(memory_path)

    def solve_task(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        # Search state — system-level tracking (not relying on model self-report)
        current_step = 0
        best_score: Optional[float] = None
        baseline_score: Optional[float] = None
        score_direction: Optional[str] = None  # 'lower' or 'higher'
        all_turn_scores: List[Optional[float]] = []  # per-turn reported scores

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

            # Clear stale submission file from previous task on this container
            _stale = os.path.join(
                self.submission_dir, f"container_{container_id:03d}", "submission.csv"
            )
            if os.path.exists(_stale):
                os.remove(_stale)

            # Inject EET system prompt (single prompt, no phase switching)
            conversation = self._inject_system_prompt(conversation)

            # Task info for cross-task memory
            challenge_name = sample.get("extra_info", {}).get(
                "challenge_name", ""
            )
            task_description = sample.get("extra_info", {}).get(
                "question", ""
            )[:500]

            # Inject cross-task memory into system prompt if available
            cross_task_context = self.cross_task_memory.format_for_prompt(
                challenge_name=challenge_name,
                task_description=task_description,
            )
            if cross_task_context:
                # Append to system message
                if conversation and conversation[0].get("role") == "system":
                    conversation[0]["content"] += "\n\n" + cross_task_context

            total_tokens = 0
            final_answer = ""
            actual_turns = 0

            for turn in range(self.max_turns):
                current_step = turn + 1
                turn_start = time.time()
                prev_best_before_turn = best_score

                try:
                    response = self.backend_instance.generate(conversation)
                    total_tokens += len(response.split())
                    actual_turns = turn + 1

                    # Parse structured output
                    parsed: Optional[StructuredOutput] = None
                    parse_success = False
                    try:
                        parsed = parse_eet_output(response)
                        parse_success = True
                    except Exception as e:
                        print(f"  Warning: XML parse failed at turn {current_step}: {e}")

                    # Execute code in container
                    step_output = env.step(response)
                    step_time = time.time() - turn_start

                    exec_output = step_output.get("metadata", {}).get(
                        "execution_output", ""
                    )

                    # Read scores from model's parsed <search_state>
                    # System-level best_score tracking: don't blindly trust model
                    reported_score: Optional[float] = None
                    score_delta: Optional[float] = None

                    if parsed and parsed.search_state:
                        ss = parsed.search_state
                        if ss.baseline_score is not None and baseline_score is None:
                            baseline_score = ss.baseline_score

                        model_best = ss.best_score
                        if model_best is not None:
                            # Infer metric direction from first meaningful comparison
                            if score_direction is None and baseline_score is not None:
                                if model_best < baseline_score:
                                    score_direction = "lower"
                                elif model_best > baseline_score:
                                    score_direction = "higher"

                            # System-level best: only update if actually better
                            if best_score is None:
                                best_score = model_best
                            elif score_direction == "higher":
                                best_score = max(best_score, model_best)
                            else:  # "lower" or None (default lower)
                                best_score = min(best_score, model_best)

                            reported_score = best_score
                            all_turn_scores.append(best_score)
                        else:
                            all_turn_scores.append(None)

                        if baseline_score is not None and best_score is not None:
                            score_delta = best_score - baseline_score

                    # Detect if this turn was buggy
                    is_buggy = bool(
                        exec_output
                        and "Traceback (most recent call last)" in exec_output
                    )

                    # Record turn
                    turn_records.append(
                        TurnRecord(
                            turn=current_step,
                            phase="eet",
                            raw_response=response,
                            parsed_output=asdict(parsed) if parsed else None,
                            execution_output=exec_output,
                            score=reported_score,
                            score_delta=score_delta,
                            predicted_delta=None,
                            parse_success=parse_success,
                            step_time=step_time,
                        )
                    )

                    # Store to cross-task memory (only on score improvement)
                    if reported_score is not None and not is_buggy:
                        is_new_best = False
                        if prev_best_before_turn is None:
                            is_new_best = True  # first score
                        elif reported_score != prev_best_before_turn:
                            is_new_best = True
                        if is_new_best:
                            code = ""
                            if parsed and parsed.python_code:
                                code = parsed.python_code
                            plan = ""
                            if parsed and parsed.search_state and parsed.search_state.goal:
                                plan = parsed.search_state.goal
                            action_name = "eet"
                            if parsed and parsed.decision:
                                action_name = parsed.decision.action or "eet"
                            self.cross_task_memory.store_improvement(
                                challenge_name=challenge_name,
                                task_description=task_description,
                                turn=current_step,
                                action=action_name,
                                plan=plan,
                                code=code,
                                new_score=reported_score,
                                prev_best_score=prev_best_before_turn,
                                score_direction=score_direction or "lower",
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
                        trajectory, turn, "assistant", response,
                        step_output.get("done", False),
                        step_output.get("reward", 0.0), step_time,
                    )

                    if step_output["observations"]:
                        conversation.extend(step_output["observations"])
                        trajectory = self.append_traj(
                            trajectory, turn, "user",
                            step_output["observations"][0]["content"],
                            step_output.get("done", False),
                            step_output.get("reward", 0.0), step_time,
                        )

                        # Remind model of output format for next turn
                        conversation.append(
                            {"role": "user", "content": self._build_format_reminder(
                                current_step, system_best_score=best_score
                            )}
                        )
                    else:
                        step_output["done"] = True

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

            # Save prediction
            if final_answer:
                prefix = sample.get("extra_info", {}).get("id", "temp")
                env.save_prediction(final_answer, filename_prefix=prefix)

            execution_time = time.time() - start_time

            # Handle submission file
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

            # Save trajectory
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

            # Store task summary to cross-task memory
            best_plan = ""
            if turn_records:
                # Find the plan from the best-scoring turn
                for tr in reversed(turn_records):
                    po = tr.parsed_output
                    if po and po.get("search_state", {}).get("goal"):
                        best_plan = po["search_state"]["goal"]
                        break
            self.cross_task_memory.store_task_summary(
                challenge_name=challenge_name,
                task_description=task_description,
                total_turns=actual_turns,
                best_score=best_score,
                baseline_score=baseline_score,
                best_plan=best_plan,
                best_code="",
                success=success,
            )

            return {
                "solution": submission_path,
                "success": success,
                "turns": actual_turns,
                "error": None,
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "agent_type": "eet",
                    "max_turns": self.max_turns,
                    "total_tokens": total_tokens,
                    "execution_time": execution_time,
                    "conversation_length": len(conversation),
                    "best_score": best_score,
                    "baseline_score": baseline_score,
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
            print(f"Error in EETAgent: {error_trace}")

            return {
                "solution": "",
                "success": False,
                "turns": 0,
                "error": str(e),
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "agent_type": "eet",
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
        self, conversation: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepend EET system prompt to conversation."""
        if self.self_contained:
            prompt = EET_SYSTEM_PROMPT_NO_TERMINATE_SELF_CONTAINED if self.no_terminate else EET_SYSTEM_PROMPT_SELF_CONTAINED
        else:
            prompt = EET_SYSTEM_PROMPT_NO_TERMINATE if self.no_terminate else EET_SYSTEM_PROMPT
        if conversation and conversation[0].get("role") == "system":
            conversation[0]["content"] = (
                prompt + "\n\n" + conversation[0]["content"]
            )
        else:
            conversation.insert(0, {"role": "system", "content": prompt})
        return conversation

    def _build_format_reminder(self, step: int, system_best_score: Optional[float] = None) -> str:
        """Remind model of required output format for next turn."""
        if self.no_terminate:
            is_last_turn = (step + 1 >= self.max_turns)
            if is_last_turn:
                reminder = (
                    f"[Step {step + 1} — FINAL TURN] This is your last turn.\n"
                    f"Generate your final submission using your best model. "
                    f"Save predictions to /submission/submission.csv.\n"
                    f"Your response MUST contain:\n"
                    f"1. <search_state> — update step, best_score, baseline_score, history, goal\n"
                    f"2. <decision> — choose explore or exploit\n"
                    f"3. <python> — generate and save final submission\n"
                )
            else:
                remaining = self.max_turns - step - 1
                reminder = (
                    f"[Step {step + 1}] ({remaining} turns remaining) "
                    f"Your response MUST contain these blocks in order:\n"
                    f"1. <search_state> — update step, best_score (CV/validation only), baseline_score, history, goal\n"
                    f"2. <decision> — choose explore or exploit\n"
                    f"3. <python> — implement your decision\n"
                )
        else:
            reminder = (
                f"[Step {step + 1}] Your response MUST contain these blocks in order:\n"
                f"1. <search_state> — update step, best_score (CV/validation only), baseline_score, history, goal\n"
                f"2. <decision> — choose explore, exploit, or terminate\n"
                f"3. <python> — implement your decision (or <answer> for terminate)\n"
            )
        if system_best_score is not None:
            reminder += (
                f"\n[System] Your verified best score across all turns is {system_best_score}. "
                f"Use this as your <best_score> in <search_state>."
            )
        return reminder

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
            "agent_type": "eet",
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

        print(f"Saved EET trajectory: {filepath}")
