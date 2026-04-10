"""
AIDEAgent — Draft / Improve / Debug structured agent (V4 with turn summaries).

Uses AIDE's hard-coded action selection rules and per-turn prompts.
After each turn's code execution, calls LLM to generate a structured summary
that feeds into both task-internal memory and cross-task memory.

Decision logic (hard-coded, not model-chosen):
  1. First num_drafts turns → Draft
  2. With debug_prob probability, if last turn was buggy → Debug
  3. Otherwise → Improve the current best approach
"""

import os
import re
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
    parse_aide_output_v6,
    StructuredOutput,
)
from .aide_prompts import (
    AIDE_SYSTEM_PROMPT,
    AIDE_SYSTEM_PROMPT_V6,
    AIDE_DRAFT_INSTRUCTION,
    AIDE_IMPROVE_INSTRUCTION,
    AIDE_DEBUG_INSTRUCTION,
    AIDE_FINAL_SUBMISSION_INSTRUCTION,
    AIDE_SUMMARY_PROMPT,
    AIDE_SUMMARY_PROMPT_V5,
    AIDE_SUMMARY_PROMPT_V6,
)
from .teacher_agent import (
    TurnRecord,
    StructuredTrajectory,
)
from .memory import CrossTaskMemory


def _extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between XML tags. Returns None if not found."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _safe_float(s: Optional[str]) -> Optional[float]:
    """Convert string to float, returning None on failure."""
    if not s or s.lower() in ("", "none", "error", "n/a", "null"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


class AIDEAgent(DSPredictReActAgent):
    """
    AIDE-style agent with Draft/Improve/Debug actions and turn summaries.

    After each turn's code execution, calls LLM to generate a structured
    summary that updates task-internal memory and cross-task memory.
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
        self.memory_version = kwargs.pop("memory_version", "v4")
        self.no_task_memory = kwargs.pop("no_task_memory", False)
        self.no_cross_memory = kwargs.pop("no_cross_memory", False)
        self.log_degradation = kwargs.pop("log_degradation", False)
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

        # Task-internal memory: list of per-turn summaries
        # Each entry: {"step": int, "model": str, "score": float|None, "notes": str}
        task_memory: List[Dict[str, Any]] = []

        # Track previous turn's execution output (for summary context)
        prev_exec_output: str = ""

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
            task_description_full = sample.get("extra_info", {}).get(
                "question", ""
            )
            task_description = task_description_full[:3000] if self.memory_version == "v6" else task_description_full[:500]

            total_tokens = 0
            final_answer = ""
            actual_turns = 0

            for turn in range(self.max_turns):
                current_step = turn + 1
                turn_start = time.time()

                try:
                    # === AIDE hard-coded action selection ===
                    action = self._select_action(
                        completed_drafts=completed_drafts,
                        last_was_buggy=last_was_buggy,
                        consecutive_debug_count=consecutive_debug_count,
                        task_memory=task_memory,
                        is_last_turn=(turn == self.max_turns - 1),
                    )

                    # Build per-turn instruction based on action
                    turn_instruction = self._build_turn_instruction(
                        action=action,
                        step=current_step,
                        task_memory=task_memory,
                        last_error_output=last_error_output,
                        challenge_name=challenge_name,
                        task_description=task_description,
                    )

                    # Inject turn instruction into conversation
                    conversation.append(
                        {"role": "user", "content": turn_instruction}
                    )

                    # Merge consecutive user messages before sending to model
                    # (training data has merged users; inference must match)
                    merged_conv = []
                    for _m in conversation:
                        if merged_conv and merged_conv[-1]["role"] == "user" and _m["role"] == "user":
                            merged_conv[-1] = {
                                "role": "user",
                                "content": merged_conv[-1]["content"] + "\n\n" + _m["content"],
                            }
                        else:
                            merged_conv.append({"role": _m["role"], "content": _m["content"]})

                    # Generate response
                    response = self.backend_instance.generate(merged_conv)
                    total_tokens += len(response.split())
                    actual_turns = turn + 1

                    # Parse structured output
                    parsed: Optional[StructuredOutput] = None
                    parse_success = False
                    try:
                        if self.memory_version == "v6":
                            parsed = parse_aide_output_v6(response)
                        else:
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

                    # Read scores from model's <search_state> (v4/v5)
                    # V6: scores are tracked via summary LLM + is_best, not model output
                    reported_score: Optional[float] = None
                    score_delta: Optional[float] = None

                    if self.memory_version != "v6" and parsed and parsed.search_state:
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
                        last_error_output = exec_output[-5000:]
                        if action == "debug":
                            consecutive_debug_count += 1
                        else:
                            consecutive_debug_count = 1
                    else:
                        last_was_buggy = False
                        last_error_output = ""
                        consecutive_debug_count = 0

                    # Record turn
                    turn_records.append(
                        TurnRecord(
                            turn=current_step,
                            phase=action,
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

                    # === Generate turn summary via LLM ===
                    # V5/V6: pass reference approach notes for key_change comparison
                    ref_approach_text = ""
                    if self.memory_version in ("v5", "v6"):
                        ref_entry = self._get_best_task_memory_entry(task_memory)
                        if ref_entry:
                            ref_approach_text = (
                                f"Step {ref_entry.get('step', '?')} | "
                                f"Model: {ref_entry.get('model', 'unknown')} | "
                                f"Score: {ref_entry.get('score', 'N/A')} | "
                                f"{ref_entry.get('notes', '')}"
                            )

                    summary = self._generate_turn_summary(
                        task_memory=task_memory,
                        prev_exec_output=prev_exec_output,
                        current_step=current_step,
                        action=action,
                        plan=plan,
                        exec_output=exec_output or "",
                        reference_approach=ref_approach_text,
                        task_description=task_description if self.memory_version == "v6" else "",
                        code=response if self.memory_version == "v6" else "",
                    )

                    if summary:
                        # Update task-internal memory
                        te = summary["task_entry"]

                        # V6: track is_best and derive best_score/baseline_score from task_memory
                        if self.memory_version == "v6":
                            summary_best = te.get("best_score")
                            # Check if best_score changed compared to previous entry
                            prev_best = None
                            for prev_e in reversed(task_memory):
                                if prev_e.get("best_score") is not None:
                                    prev_best = prev_e["best_score"]
                                    break
                            if summary_best is not None and (prev_best is None or abs(summary_best - prev_best) > 1e-9):
                                # New best — mark this entry and unmark previous
                                te["is_best"] = True
                                for prev_e in task_memory:
                                    prev_e["is_best"] = False
                                best_score = summary_best
                            else:
                                te["is_best"] = False
                            # baseline_score = first entry with a valid score
                            if baseline_score is None:
                                for prev_e in task_memory:
                                    if prev_e.get("score") is not None:
                                        baseline_score = prev_e["score"]
                                        break
                                if baseline_score is None and te.get("score") is not None:
                                    baseline_score = te["score"]
                            # Update reported_score and score_delta for turn record
                            if best_score is not None:
                                reported_score = best_score
                            if baseline_score is not None and best_score is not None:
                                score_delta = best_score - baseline_score

                        task_memory.append(te)
                        print(
                            f"  [TaskMemory] step={te.get('step')}, "
                            f"model={te.get('model')}, "
                            f"score={te.get('score')}, "
                            f"best_score={te.get('best_score')}, "
                            f"notes={te.get('notes', '')[:120]}"
                        )

                        # Write to cross-task memory if model deemed it worthy
                        if summary.get("cross_task_entry"):
                            ct = summary["cross_task_entry"]

                            # V5/V6: build V2-style insight with plan + score + key_change
                            if self.memory_version in ("v5", "v6"):
                                ct_score = te.get("score")
                                key_change = ct.get("key_change", "")
                                # Use reference entry's score as the "before" score
                                # ref_entry was computed above based on best_node_strategy
                                ref_score = ref_entry.get("score") if ref_entry else None
                                # Build V2-style insight template
                                if ct.get("type") == "improvement" and ct_score is not None:
                                    if ref_score is not None:
                                        try:
                                            pct = abs((ct_score - ref_score) / ref_score * 100)
                                        except ZeroDivisionError:
                                            pct = 0
                                        ct_insight = (
                                            f"Score: {ref_score} -> {ct_score} ({pct:.1f}% change). "
                                            f"Method: {ct.get('model', 'unknown')}. "
                                            f"Key change: {key_change}"
                                        )
                                    else:
                                        ct_insight = (
                                            f"Baseline score: {ct_score}. "
                                            f"Method: {ct.get('model', 'unknown')}. "
                                            f"Key change: {key_change}"
                                        )
                                else:
                                    # debug_fix or no score
                                    ct_insight = (
                                        f"Method: {ct.get('model', 'unknown')}. "
                                        f"Key change: {key_change}"
                                    )
                            else:
                                ct_insight = ct.get("insight", "")

                            if not self.no_cross_memory:
                                self.cross_task_memory.store_from_summary(
                                    challenge_name=challenge_name,
                                    task_description=task_description,
                                    turn=current_step,
                                    action=action,
                                    entry_type=ct.get("type", "improvement"),
                                    model_type=ct.get("model", "unknown"),
                                    insight=ct_insight,
                                    plan=plan if self.memory_version in ("v5", "v6") else "",
                                    score=te.get("score") if self.memory_version in ("v5", "v6") else None,
                                )
                                print(
                                    f"  [CrossTaskMemory] type={ct.get('type')}, "
                                    f"model={ct.get('model')}, "
                                    f"insight={ct_insight[:120]}"
                                )

                        # V5/V6: write degradation entry if improve failed
                        if (
                            self.log_degradation
                            and self.memory_version in ("v5", "v6")
                            and not summary.get("cross_task_entry")
                            and action == "improve"
                            and te.get("score") is not None
                            and ref_entry is not None
                            and ref_entry.get("score") is not None
                        ):
                            # Check if best_score didn't change (no improvement)
                            prev_best = None
                            for prev_e in reversed(task_memory[:-1]):
                                if prev_e.get("best_score") is not None:
                                    prev_best = prev_e["best_score"]
                                    break
                            curr_best = te.get("best_score")
                            if prev_best is not None and curr_best is not None and abs(prev_best - curr_best) < 1e-9:
                                # best_score unchanged and score differs from ref → degradation
                                deg_score = te["score"]
                                deg_ref = ref_entry["score"]
                                if abs(deg_score - deg_ref) > 1e-9:
                                    try:
                                        deg_pct = abs((deg_score - deg_ref) / deg_ref * 100)
                                    except ZeroDivisionError:
                                        deg_pct = 0
                                    deg_insight = (
                                        f"Score: {deg_ref} -> {deg_score} ({deg_pct:.1f}% change, NOT an improvement). "
                                        f"Method: {te.get('model', 'unknown')}. "
                                        f"Attempted: {te.get('notes', '')}"
                                    )
                                    if not self.no_cross_memory:
                                        self.cross_task_memory.store_from_summary(
                                            challenge_name=challenge_name,
                                            task_description=task_description,
                                            turn=current_step,
                                            action=action,
                                            entry_type="degradation",
                                            model_type=te.get("model", "unknown"),
                                            insight=deg_insight,
                                            plan=plan,
                                            score=deg_score,
                                        )
                                        print(
                                            f"  [CrossTaskMemory] type=degradation, "
                                            f"model={te.get('model')}, "
                                            f"insight={deg_insight[:120]}"
                                        )

                    # Track previous execution output for next summary
                    prev_exec_output = exec_output or ""

                    print(
                        f"  Turn {current_step}: action={action}, "
                        f"buggy={is_buggy}, score={reported_score}, "
                        f"drafts={completed_drafts}, "
                        f"task_memory_size={len(task_memory)}, "
                        f"node_strategy={self.best_node_strategy}"
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
            self._save_trajectory(traj_data, sample, task_memory=task_memory)

            # Store task summary to cross-task memory
            best_entry = self._get_best_task_memory_entry(task_memory)
            best_plan = best_entry.get("notes", "") if best_entry else ""
            best_model = best_entry.get("model", "") if best_entry else ""
            if not self.no_cross_memory:
                self.cross_task_memory.store_task_summary(
                    challenge_name=challenge_name,
                    task_description=task_description,
                    total_turns=actual_turns,
                    best_score=best_score,
                    baseline_score=baseline_score,
                    best_plan=best_plan,
                    best_code=best_model,
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
                    "num_task_memory": len(task_memory),
                    "parse_success_rate": (
                        sum(1 for t in turn_records if t.parse_success)
                        / len(turn_records)
                        if turn_records
                        else 0.0
                    ),
                },
                "conversation": conversation,
                "trajectory": trajectory,
                "task_memory": task_memory,
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
    # Turn summary generation
    # ================================================================

    def _generate_turn_summary(
        self,
        task_memory: List[Dict[str, Any]],
        prev_exec_output: str,
        current_step: int,
        action: str,
        plan: str,
        exec_output: str,
        reference_approach: str = "",
        task_description: str = "",
        code: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM to generate a structured summary after code execution.

        Returns dict with 'task_entry' and optional 'cross_task_entry',
        or None if generation/parsing fails.
        """
        try:
            task_memory_text = self._format_task_memory(task_memory)

            if self.memory_version == "v6":
                prompt = AIDE_SUMMARY_PROMPT_V6.format(
                    task_description=(task_description or "N/A")[:3000],
                    task_memory_text=task_memory_text or "No previous attempts yet.",
                    reference_approach=reference_approach or "N/A (first attempt or new draft)",
                    step=current_step,
                    action=action,
                    plan=plan or "N/A",
                    code=(code or "N/A")[-3000:],
                    exec_output=(exec_output or "No output")[-3000:],
                )
            elif self.memory_version == "v5":
                prompt = AIDE_SUMMARY_PROMPT_V5.format(
                    task_memory_text=task_memory_text or "No previous attempts yet.",
                    reference_approach=reference_approach or "N/A (first attempt or new draft)",
                    prev_exec_output=(prev_exec_output or "N/A")[-3000:],
                    step=current_step,
                    action=action,
                    plan=plan or "N/A",
                    exec_output=(exec_output or "No output")[-3000:],
                )
            else:
                prompt = AIDE_SUMMARY_PROMPT.format(
                    task_memory_text=task_memory_text or "No previous attempts yet.",
                    prev_exec_output=(prev_exec_output or "N/A")[-3000:],
                    step=current_step,
                    action=action,
                    plan=plan or "N/A",
                    exec_output=(exec_output or "No output")[-3000:],
                )

            summary_response = self.backend_instance.generate([
                {"role": "user", "content": prompt}
            ])

            return self._parse_summary_response(summary_response, current_step)

        except Exception as e:
            print(f"  Warning: Summary generation failed at step {current_step}: {e}")
            # Fallback: create a minimal task_entry from available info
            return {
                "task_entry": {
                    "step": current_step,
                    "model": "unknown",
                    "score": None,
                    "best_score": None,
                    "notes": f"[{action}] {plan[:100]}" if plan else f"[{action}] summary failed",
                },
                "cross_task_entry": None,
            }

    def _parse_summary_response(
        self, response: str, current_step: int
    ) -> Optional[Dict[str, Any]]:
        """Parse the XML summary response from LLM."""
        result = {"task_entry": None, "cross_task_entry": None}

        # Parse task_memory block
        step = _safe_float(_extract_xml_tag(response, "step"))
        model = _extract_xml_tag(response, "model") or "unknown"
        score_str = _extract_xml_tag(response, "score")
        score = _safe_float(score_str)
        best_score_str = _extract_xml_tag(response, "best_score")
        best_score = _safe_float(best_score_str)
        notes = _extract_xml_tag(response, "notes") or ""

        result["task_entry"] = {
            "step": int(step) if step else current_step,
            "model": model,
            "score": score,
            "best_score": best_score,
            "notes": notes,
        }

        # Parse cross_task_memory block (optional)
        cross_block = re.search(
            r"<cross_task_memory>(.*?)</cross_task_memory>",
            response, re.DOTALL
        )
        if cross_block:
            block = cross_block.group(1)
            ct_type = _extract_xml_tag(block, "type") or "improvement"
            ct_model = _extract_xml_tag(block, "model") or model
            # V5 uses <key_change>, V4 uses <insight>
            ct_key_change = _extract_xml_tag(block, "key_change") or ""
            ct_insight = _extract_xml_tag(block, "insight") or ""
            ct_content = ct_key_change or ct_insight  # V5 first, fallback V4
            if ct_content:
                result["cross_task_entry"] = {
                    "type": ct_type,
                    "model": ct_model,
                    "key_change": ct_key_change,
                    "insight": ct_insight,
                }

        return result

    # ================================================================
    # Task memory formatting
    # ================================================================

    def _format_task_memory(self, task_memory: List[Dict[str, Any]]) -> str:
        """Format task-internal memory for prompt injection."""
        if not task_memory:
            return ""

        lines = ["=== TASK MEMORY (Previous Attempts) ==="]
        for entry in task_memory:
            step = entry.get("step", "?")
            model = entry.get("model", "unknown")
            score = entry.get("score")
            score_str = f"{score}" if score is not None else "error"
            best_score = entry.get("best_score")
            best_str = f"{best_score}" if best_score is not None else "N/A"
            notes = entry.get("notes", "")
            lines.append(
                f"Step {step} | Model: {model} | Score: {score_str} | "
                f"Best so far: {best_str} | {notes}"
            )
        lines.append("=== END TASK MEMORY ===")
        return "\n".join(lines)

    def _get_best_task_memory_entry(
        self, task_memory: List[Dict[str, Any]], strategy: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the task memory entry to improve upon.

        Args:
            strategy: "latest" (most recent scored entry) or "best" (entry whose
                      score matches the model-reported best_score).
                      Falls back to self.best_node_strategy if None.
        """
        strategy = strategy or self.best_node_strategy
        scored = [e for e in task_memory if e.get("score") is not None]
        if not scored:
            return task_memory[-1] if task_memory else None

        if strategy == "best":
            # V6: use is_best flag directly
            if self.memory_version == "v6":
                for e in reversed(task_memory):
                    if e.get("is_best"):
                        return e
                # Fallback: return latest scored
                return scored[-1]

            # V4/V5: Use the model-reported best_score from the most recent entry
            # to find which earlier entry actually achieved that score.
            # This lets the model handle metric direction (higher/lower is better).
            latest_best = None
            for e in reversed(task_memory):
                if e.get("best_score") is not None:
                    latest_best = e["best_score"]
                    break

            if latest_best is not None:
                # Find the entry whose own score matches best_score
                for e in reversed(scored):
                    if e.get("score") is not None and abs(e["score"] - latest_best) < 1e-9:
                        return e

            # Fallback: if no best_score tracked yet, return latest scored
            return scored[-1]

        # "latest" — most recent scored entry
        return scored[-1]

    # ================================================================
    # AIDE action selection
    # ================================================================

    def _select_action(
        self,
        completed_drafts: int,
        last_was_buggy: bool,
        consecutive_debug_count: int,
        task_memory: List[Dict[str, Any]],
        is_last_turn: bool,
    ) -> str:
        """AIDE's hard-coded action selection policy."""
        if is_last_turn:
            return "final_submission"

        if completed_drafts < self.num_drafts:
            return "draft"

        if last_was_buggy and consecutive_debug_count < self.max_debug_depth:
            if random.random() < self.debug_prob:
                return "debug"

        # Check if any task_memory entry has a valid score
        has_scored = any(e.get("score") is not None for e in task_memory)
        if not has_scored:
            return "draft"

        return "improve"

    # ================================================================
    # Prompt construction
    # ================================================================

    def _inject_system_prompt(
        self, conversation: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Prepend AIDE system prompt to conversation."""
        system_prompt = AIDE_SYSTEM_PROMPT_V6 if self.memory_version == "v6" else AIDE_SYSTEM_PROMPT
        if conversation and conversation[0].get("role") == "system":
            conversation[0]["content"] = (
                system_prompt + "\n\n" + conversation[0]["content"]
            )
        else:
            conversation.insert(
                0, {"role": "system", "content": system_prompt}
            )
        return conversation

    def _build_turn_instruction(
        self,
        action: str,
        step: int,
        task_memory: List[Dict[str, Any]],
        last_error_output: str,
        challenge_name: str = "",
        task_description: str = "",
    ) -> str:
        """Build per-turn instruction based on the selected action."""
        # Build memory section: task-internal memory + cross-task memory
        if self.no_task_memory:
            memory_section = ""
        else:
            memory_section = self._format_task_memory(task_memory)
            if not memory_section:
                memory_section = "No previous attempts yet."

        # Append cross-task memory
        skip_cross_memory = self.no_cross_memory or (self.no_draft_memory and action == "draft")
        if not skip_cross_memory:
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
            # Inject reference node info based on best_node_strategy
            ref_entry = self._get_best_task_memory_entry(task_memory)
            if ref_entry:
                ref_info = (
                    f"\n--- Reference approach ({self.best_node_strategy} scored) ---\n"
                    f"Step {ref_entry.get('step', '?')} | "
                    f"Model: {ref_entry.get('model', 'unknown')} | "
                    f"Score: {ref_entry.get('score', 'N/A')} | "
                    f"{ref_entry.get('notes', '')}\n"
                    f"--- End reference ---\n"
                )
                memory_section = ref_info + "\n" + memory_section
            instruction = AIDE_IMPROVE_INSTRUCTION.format(
                memory_section=memory_section,
            )
        elif action == "debug":
            instruction = AIDE_DEBUG_INSTRUCTION.format(
                error_output=last_error_output,
                memory_section=memory_section,
            )
        elif action == "final_submission":
            instruction = AIDE_FINAL_SUBMISSION_INSTRUCTION.format(
                memory_section=memory_section,
            )
        else:
            instruction = f"[Step {step}] Continue working on the task."

        result = f"[Step {step}/{self.max_turns}]\n{instruction}"

        # V6: replace references to <search_state><goal> with just <goal>
        if self.memory_version == "v6":
            result = result.replace("inside <search_state><goal>", "inside <goal>")
            result = result.replace("<search_state><goal>", "<goal>")

        return result

    # ================================================================
    # Bug detection
    # ================================================================

    def _is_buggy(
        self, exec_output: str, step_output: Dict[str, Any]
    ) -> bool:
        """Detect if a turn's execution was buggy."""
        if not exec_output:
            return False

        if "Traceback (most recent call last)" in exec_output:
            return True

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
        self, traj: StructuredTrajectory, sample: Dict[str, Any],
        task_memory: Optional[List[Dict[str, Any]]] = None,
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
            "best_node_strategy": self.best_node_strategy,
            "final_best_score": traj.final_best_score,
            "baseline_score": traj.baseline_score,
            "total_time": traj.total_time,
            "success": traj.success,
            "num_turns": len(traj.turns),
            "turns": [asdict(t) for t in traj.turns],
            "task_memory": task_memory or [],
            "conversation": traj.conversation,
            "sample_extra_info": sample.get("extra_info", {}),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved AIDE trajectory: {filepath}")
