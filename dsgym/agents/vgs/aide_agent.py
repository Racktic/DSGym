"""
AIDEAgent — Draft / Improve / Debug structured agent.

Faithfully adapted from AIDE (WecoAI/aideml, arXiv:2502.13138).
Uses AIDE's hard-coded action selection rules and per-turn prompts,
but operates in a linear conversation (no tree search) for fair
comparison with EET.

Decision logic (hard-coded, not model-chosen):
  1. First num_drafts turns → Draft
  2. With debug_prob probability, if last turn was buggy → Debug
  3. Otherwise → Improve the current best approach
"""

import os
import json
import time
import random
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any, List, Optional

from dsgym.agents.dspredict_react_agent import DSPredictReActAgent
from dsgym.agents.environment import AllocatedCodeEnv
from .structured_output import (
    parse_aide_output,
    StructuredOutput,
)
from .aide_prompts import (
    AIDE_SYSTEM_PROMPT,
    AIDE_DRAFT_INSTRUCTION,
    AIDE_IMPROVE_INSTRUCTION,
    AIDE_DEBUG_INSTRUCTION,
    AIDE_FINAL_SUBMISSION_INSTRUCTION,
)
from .teacher_agent import (
    TurnRecord,
    StructuredTrajectory,
)
from .memory import CrossTaskMemory


class AIDEAgent(DSPredictReActAgent):
    """
    AIDE-style agent with Draft/Improve/Debug actions.

    Action selection is hard-coded (faithful to AIDE):
    - First num_drafts turns: Draft new solutions
    - If last turn was buggy (with debug_prob): Debug
    - Otherwise: Improve the best solution
    """

    # AIDE hyperparameters (from AIDE config.yaml defaults)
    DEFAULT_NUM_DRAFTS = 5
    DEFAULT_DEBUG_PROB = 0.5
    DEFAULT_MAX_DEBUG_DEPTH = 3

    def __init__(self, backend: str, model: str, **kwargs):
        self.trajectory_output_dir = kwargs.pop(
            "trajectory_output_dir", "./aide_trajectories"
        )
        self.num_drafts = kwargs.pop("num_drafts", self.DEFAULT_NUM_DRAFTS)
        self.debug_prob = kwargs.pop("debug_prob", self.DEFAULT_DEBUG_PROB)
        self.max_debug_depth = kwargs.pop(
            "max_debug_depth", self.DEFAULT_MAX_DEBUG_DEPTH
        )
        memory_path = kwargs.pop(
            "memory_path", "./cross_task_memory.json"
        )
        self.no_draft_memory = kwargs.pop("no_draft_memory", False)
        self.best_node_strategy = kwargs.pop("best_node_strategy", "latest")
        super().__init__(backend, model, **kwargs)
        os.makedirs(self.trajectory_output_dir, exist_ok=True)
        self.cross_task_memory = CrossTaskMemory(memory_path)

    def solve_task(self, sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        # AIDE decision state
        completed_drafts = 0
        consecutive_debug_count = 0
        last_was_buggy = False
        last_error_output = ""

        # Score tracking (from model's <search_state>)
        best_score: Optional[float] = None
        baseline_score: Optional[float] = None

        # Memory: list of good (non-buggy) nodes
        # Each entry: {"plan": str, "score": float, "analysis": str}
        good_nodes: List[Dict[str, Any]] = []

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

            # Inject AIDE system prompt
            conversation = self._inject_system_prompt(conversation)

            # Task info for cross-task memory
            challenge_name = sample.get("extra_info", {}).get(
                "challenge_name", ""
            )
            task_description = sample.get("extra_info", {}).get(
                "question", ""
            )[:500]  # Truncate long descriptions

            total_tokens = 0
            final_answer = ""
            actual_turns = 0

            for turn in range(self.max_turns):
                current_step = turn + 1
                turn_start = time.time()
                prev_best_before_turn = best_score
                prev_turn_was_buggy = last_was_buggy
                prev_error_output = last_error_output

                try:
                    # === AIDE hard-coded action selection ===
                    action = self._select_action(
                        completed_drafts=completed_drafts,
                        last_was_buggy=last_was_buggy,
                        consecutive_debug_count=consecutive_debug_count,
                        good_nodes=good_nodes,
                        is_last_turn=(turn == self.max_turns - 1),
                    )

                    # Build per-turn instruction based on action
                    turn_instruction = self._build_turn_instruction(
                        action=action,
                        step=current_step,
                        good_nodes=good_nodes,
                        best_score=best_score,
                        last_error_output=last_error_output,
                        challenge_name=challenge_name,
                        task_description=task_description,
                    )

                    # Inject turn instruction into conversation
                    conversation.append(
                        {"role": "user", "content": turn_instruction}
                    )

                    # Generate response
                    response = self.backend_instance.generate(conversation)
                    total_tokens += len(response.split())
                    actual_turns = turn + 1

                    # Parse structured output
                    parsed: Optional[StructuredOutput] = None
                    parse_success = False
                    try:
                        parsed = parse_aide_output(response)
                        parse_success = True
                    except Exception as e:
                        print(
                            f"  Warning: XML parse failed at turn {current_step}: {e}"
                        )

                    # Execute code in container
                    step_output = env.step(response)
                    step_time = time.time() - turn_start

                    exec_output = step_output.get("metadata", {}).get(
                        "execution_output", ""
                    )

                    # Detect if this turn was buggy
                    is_buggy = self._is_buggy(exec_output, step_output)

                    # Extract plan/goal from parsed output
                    plan = ""
                    if parsed and parsed.search_state and parsed.search_state.goal:
                        plan = parsed.search_state.goal

                    # Read scores from model's <search_state>
                    reported_score: Optional[float] = None
                    score_delta: Optional[float] = None

                    if parsed and parsed.search_state:
                        ss = parsed.search_state
                        if ss.best_score is not None:
                            reported_score = ss.best_score
                            best_score = ss.best_score
                        if ss.baseline_score is not None:
                            baseline_score = ss.baseline_score
                        if baseline_score is not None and best_score is not None:
                            score_delta = best_score - baseline_score

                    # Update AIDE state
                    if action == "draft":
                        completed_drafts += 1

                    if is_buggy:
                        last_was_buggy = True
                        last_error_output = exec_output[-5000:]  # Truncate
                        if action == "debug":
                            consecutive_debug_count += 1
                        else:
                            consecutive_debug_count = 1
                    else:
                        last_was_buggy = False
                        last_error_output = ""
                        consecutive_debug_count = 0

                        # Add to good_nodes (Memory) if we got a score
                        if reported_score is not None:
                            analysis = exec_output[-2000:] if exec_output else ""
                            # Use model's self-reported current_score if available,
                            # otherwise fallback to best_score
                            current_score = reported_score  # fallback (best_score)
                            if parsed and parsed.search_state and parsed.search_state.current_score is not None:
                                current_score = parsed.search_state.current_score
                            good_nodes.append({
                                "plan": plan,
                                "score": current_score,
                                "analysis": analysis,
                            })

                    # Record turn
                    turn_records.append(
                        TurnRecord(
                            turn=current_step,
                            phase=action,  # "draft", "improve", "debug"
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

                    # Store to cross-task memory (only on significant events)
                    code = ""
                    if parsed and parsed.python_code:
                        code = parsed.python_code

                    # Case 1: Score improved (new best)
                    if reported_score is not None and not is_buggy:
                        is_new_best = False
                        if prev_best_before_turn is None:
                            is_new_best = True  # first score = baseline
                        elif reported_score != prev_best_before_turn:
                            is_new_best = True
                        if is_new_best:
                            self.cross_task_memory.store_improvement(
                                challenge_name=challenge_name,
                                task_description=task_description,
                                turn=current_step,
                                action=action,
                                plan=plan,
                                code=code,
                                new_score=reported_score,
                                prev_best_score=prev_best_before_turn,
                            )

                    # Case 2: Successful debug (was buggy, now fixed)
                    if action == "debug" and not is_buggy and prev_turn_was_buggy:
                        error_type = ""
                        error_message = ""
                        import re as _re
                        err_match = _re.search(
                            r"(TypeError|ValueError|KeyError|TimeoutError|"
                            r"AttributeError|NameError|ImportError):\s*(.{0,150})",
                            prev_error_output or "",
                        )
                        if err_match:
                            error_type = err_match.group(1)
                            error_message = err_match.group(2).strip()
                        self.cross_task_memory.store_debug_fix(
                            challenge_name=challenge_name,
                            task_description=task_description,
                            turn=current_step,
                            plan=plan,
                            code=code,
                            error_type=error_type,
                            error_message=error_message,
                            fix_description=plan,
                        )

                    print(
                        f"  Turn {current_step}: action={action}, "
                        f"buggy={is_buggy}, score={reported_score}, "
                        f"drafts={completed_drafts}, "
                        f"good_nodes={len(good_nodes)}"
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
                    last_was_buggy = True
                    last_error_output = str(step_err)
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
            best_code = ""
            if good_nodes:
                best_plan = good_nodes[-1].get("plan", "")
            self.cross_task_memory.store_task_summary(
                challenge_name=challenge_name,
                task_description=task_description,
                total_turns=actual_turns,
                best_score=best_score,
                baseline_score=baseline_score,
                best_plan=best_plan,
                best_code=best_code,
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
                    "agent_type": "aide",
                    "max_turns": self.max_turns,
                    "total_tokens": total_tokens,
                    "execution_time": execution_time,
                    "conversation_length": len(conversation),
                    "best_score": best_score,
                    "baseline_score": baseline_score,
                    "num_drafts_completed": completed_drafts,
                    "num_good_nodes": len(good_nodes),
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
            print(f"Error in AIDEAgent: {error_trace}")

            return {
                "solution": "",
                "success": False,
                "turns": 0,
                "error": str(e),
                "metadata": {
                    "model": self.model,
                    "backend": self.backend,
                    "dspredict": True,
                    "agent_type": "aide",
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
    # AIDE action selection (hard-coded, faithful to AIDE source)
    # ================================================================

    def _select_action(
        self,
        completed_drafts: int,
        last_was_buggy: bool,
        consecutive_debug_count: int,
        good_nodes: List[Dict[str, Any]],
        is_last_turn: bool,
    ) -> str:
        """
        AIDE's hard-coded action selection policy.

        Faithful to AIDE's search_policy() in agent.py:
        1. If completed_drafts < num_drafts → draft
        2. If last turn was buggy, with debug_prob probability → debug
           (up to max_debug_depth consecutive debugs)
        3. If no good nodes exist → draft
        4. Otherwise → improve
        """
        if is_last_turn:
            return "final_submission"

        # Step 1: Initial drafting phase
        if completed_drafts < self.num_drafts:
            return "draft"

        # Step 2: Debug phase (probabilistic)
        if last_was_buggy and consecutive_debug_count < self.max_debug_depth:
            if random.random() < self.debug_prob:
                return "debug"

        # Step 3: If no good nodes, draft a new solution
        if not good_nodes:
            return "draft"

        # Step 4: Improve the best solution
        return "improve"

    # ================================================================
    # Prompt construction
    # ================================================================

    def _inject_system_prompt(
        self, conversation: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepend AIDE system prompt to conversation."""
        if conversation and conversation[0].get("role") == "system":
            conversation[0]["content"] = (
                AIDE_SYSTEM_PROMPT + "\n\n" + conversation[0]["content"]
            )
        else:
            conversation.insert(
                0, {"role": "system", "content": AIDE_SYSTEM_PROMPT}
            )
        return conversation

    def _build_turn_instruction(
        self,
        action: str,
        step: int,
        good_nodes: List[Dict[str, Any]],
        best_score: Optional[float],
        last_error_output: str,
        challenge_name: str = "",
        task_description: str = "",
    ) -> str:
        """Build per-turn instruction based on the selected action."""
        memory_section = self._build_memory(good_nodes)

        # Append cross-task memory to the memory section
        # Skip during draft phase if no_draft_memory is set (V3 behavior)
        skip_memory = self.no_draft_memory and action == "draft"
        if not skip_memory:
            cross_task_context = self.cross_task_memory.format_for_prompt(
                challenge_name=challenge_name,
                task_description=task_description,
                current_action=action,
            )
            if cross_task_context:
                memory_section = memory_section + "\n\n" + cross_task_context

        if action == "draft":
            instruction = AIDE_DRAFT_INSTRUCTION.format(
                memory_section=memory_section
            )
        elif action == "improve":
            best_summary = self._get_best_approach_summary(good_nodes, best_score)
            instruction = AIDE_IMPROVE_INSTRUCTION.format(
                best_approach_summary=best_summary,
                memory_section=memory_section,
            )
        elif action == "debug":
            instruction = AIDE_DEBUG_INSTRUCTION.format(
                error_output=last_error_output
            )
        elif action == "final_submission":
            best_summary = self._get_best_approach_summary(good_nodes, best_score)
            instruction = AIDE_FINAL_SUBMISSION_INSTRUCTION.format(
                best_approach_summary=best_summary
            )
        else:
            instruction = f"[Step {step}] Continue working on the task."

        return f"[Step {step}/{self.max_turns}]\n{instruction}"

    def _build_memory(self, good_nodes: List[Dict[str, Any]]) -> str:
        """
        Build Memory section from good nodes.

        Faithful to AIDE's journal.generate_summary():
        only non-buggy nodes with valid metrics are included.
        """
        if not good_nodes:
            return "Memory: No previous successful attempts yet."

        lines = ["Memory of previous successful attempts:"]
        for i, node in enumerate(good_nodes):
            lines.append(f"-------------------------------")
            lines.append(f"Design: {node['plan']}")
            lines.append(f"Validation Metric: {node['score']}")
        lines.append(f"-------------------------------")
        return "\n".join(lines)

    def _get_best_approach_summary(
        self, good_nodes: List[Dict[str, Any]], best_score: Optional[float] = None
    ) -> str:
        """Get summary of the best approach for Improve/FinalSubmission."""
        if not good_nodes:
            return "No previous successful approach available."

        if self.best_node_strategy == "best" and best_score is not None:
            # Find the node whose score matches the system-tracked best_score
            for node in reversed(good_nodes):
                if node.get("score") == best_score:
                    return f"Design: {node['plan']}\nValidation Metric: {node['score']}"
            # Fallback if no exact match
            return f"Design: {good_nodes[-1]['plan']}\nValidation Metric: {good_nodes[-1]['score']}"
        else:
            # Default: most recent good node
            best = good_nodes[-1]
            return f"Design: {best['plan']}\nValidation Metric: {best['score']}"

    # ================================================================
    # Bug detection
    # ================================================================

    def _is_buggy(
        self, exec_output: str, step_output: Dict[str, Any]
    ) -> bool:
        """
        Detect if a turn's execution was buggy.

        Faithful to AIDE: a node is buggy if:
        1. Python exception occurred (Traceback in output)
        2. No valid output produced
        """
        if not exec_output:
            return False

        # Check for Python exceptions in output
        # Use "Traceback" as the primary indicator (most reliable),
        # plus specific exception types that won't false-positive on
        # metric names like "Mean Squared Error: 0.123"
        if "Traceback (most recent call last)" in exec_output:
            return True

        # Check for specific exception lines (e.g. "KeyError: 'col'")
        import re
        exception_pattern = re.compile(
            r"^(ModuleNotFoundError|ImportError|FileNotFoundError|"
            r"KeyError|ValueError|TypeError|IndexError|RuntimeError|"
            r"NameError|AttributeError|SyntaxError|ZeroDivisionError|"
            r"MemoryError|OSError|StopIteration|RecursionError|"
            r"TimeoutError):",
            re.MULTILINE,
        )
        if exception_pattern.search(exec_output):
            return True

        return False

    # ================================================================
    # Trajectory saving
    # ================================================================

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
            "agent_type": "aide",
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

        print(f"Saved AIDE trajectory: {filepath}")
