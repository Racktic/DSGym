"""
CrossTaskMemory — persistent cross-task experience memory.

Stores high-value experiences across tasks:
  1. Score improvements (new best score achieved)
  2. Successful debug fixes (AIDE only)
  3. Task-end summaries (best approach + final score)

Uses file locking for safe concurrent access by multiple workers.
"""

import os
import json
import re
import fcntl
from datetime import datetime
from typing import Dict, Any, List, Optional


class MemoryEntry:
    """A single experience record."""

    def __init__(
        self,
        challenge_name: str,
        task_description: str,
        turn: int,
        action: str,
        plan: str,
        model_type: str,
        score: Optional[float],
        score_improved: bool,
        buggy: bool,
        insight: str,
        entry_type: str = "turn",  # "improvement", "debug_fix", "task_summary"
        timestamp: str = "",
    ):
        self.challenge_name = challenge_name
        self.task_description = task_description
        self.turn = turn
        self.action = action
        self.plan = plan
        self.model_type = model_type
        self.score = score
        self.score_improved = score_improved
        self.buggy = buggy
        self.insight = insight
        self.entry_type = entry_type
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_name": self.challenge_name,
            "task_description": self.task_description,
            "turn": self.turn,
            "action": self.action,
            "plan": self.plan,
            "model_type": self.model_type,
            "score": self.score,
            "score_improved": self.score_improved,
            "buggy": self.buggy,
            "insight": self.insight,
            "entry_type": self.entry_type,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            challenge_name=d["challenge_name"],
            task_description=d.get("task_description", ""),
            turn=d["turn"],
            action=d["action"],
            plan=d["plan"],
            model_type=d.get("model_type", ""),
            score=d.get("score"),
            score_improved=d.get("score_improved", False),
            buggy=d.get("buggy", False),
            insight=d.get("insight", ""),
            entry_type=d.get("entry_type", "turn"),
            timestamp=d.get("timestamp", ""),
        )


# Common model keywords to detect from code/plan text
_MODEL_PATTERNS = [
    (r"catboost|CatBoost", "CatBoost"),
    (r"lightgbm|lgbm|LGBMClassifier|LGBMRegressor", "LightGBM"),
    (r"\bxgb|xgboost|XGB", "XGBoost"),
    (r"random.?forest|RandomForest", "RandomForest"),
    (r"gradient.?boost|GradientBoosting", "GradientBoosting"),
    (r"logistic.?regression|LogisticRegression", "LogisticRegression"),
    (r"\bRidge\b", "Ridge"),
    (r"\bLasso\b", "Lasso"),
    (r"\bSVC\b|\bSVR\b|support.?vector", "SVM"),
    (r"neural.?net|\bMLP\b|keras|torch|nn\.", "NeuralNet"),
    (r"\bKNN\b|KNeighbors", "KNN"),
    (r"decision.?tree|DecisionTree", "DecisionTree"),
    (r"extra.?trees?|ExtraTrees", "ExtraTrees"),
    (r"ensembl|VotingClassifier|VotingRegressor|StackingClassifier|StackingRegressor", "Ensemble"),
]


def detect_model_type(text: str) -> str:
    """Extract model type from code or plan text."""
    if not text:
        return "unknown"
    found = []
    for pattern, name in _MODEL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(name)
    return ", ".join(found) if found else "unknown"


def _summarize_plan(plan: str, max_len: int = 150) -> str:
    """Extract the core idea from a plan/goal string."""
    if not plan:
        return "unknown approach"
    # Remove common filler prefixes
    plan = re.sub(
        r"^(I will |Let me |Try to |Attempt to |The goal is to |"
        r"We should |This step will |In this step, )",
        "", plan, flags=re.IGNORECASE,
    ).strip()
    # Take first sentence if it's concise enough
    first_sentence = re.split(r"[.!?\n]", plan)[0].strip()
    if len(first_sentence) <= max_len:
        return first_sentence
    return plan[:max_len].rsplit(" ", 1)[0] + "..."


def _error_avoidance_tip(error_type: str) -> str:
    """Generate a brief avoidance tip based on error type."""
    tips = {
        "TypeError": "Check API parameter names match the library version",
        "ValueError": "Validate data shapes and value ranges before passing to model",
        "KeyError": "Verify column names exist in the DataFrame before accessing",
        "TimeoutError": "Reduce data size or model complexity to stay within time limits",
        "AttributeError": "Check that the object type supports the method being called",
        "NameError": "Ensure all variables are defined before use (self-contained code)",
        "ImportError": "Verify the package is available in the execution environment",
        "MemoryError": "Use smaller batch sizes or reduce feature dimensions",
    }
    return tips.get(error_type, "Check code carefully before execution")


class CrossTaskMemory:
    """
    Persistent cross-task memory with file locking for concurrent access.

    Only stores high-value experiences:
      - Score improvements (new best achieved)
      - Successful debug fixes (AIDE)
      - Task-end summaries
    """

    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path) or ".", exist_ok=True)

    def _read_all(self) -> List[Dict[str, Any]]:
        """Read all entries from disk (no caching — always fresh)."""
        if not os.path.exists(self.memory_path):
            return []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_SH)  # shared read lock
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                return data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load memory from {self.memory_path}: {e}")
            return []

    def _append_entry(self, entry: MemoryEntry):
        """Append a single entry with exclusive file lock (safe for concurrent writes)."""
        lock_path = self.memory_path + ".lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)  # exclusive lock
            try:
                # Read current state
                existing = []
                if os.path.exists(self.memory_path):
                    try:
                        with open(self.memory_path, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                    except (json.JSONDecodeError, KeyError):
                        existing = []

                # Append new entry
                existing.append(entry.to_dict())

                # Write back atomically
                tmp_path = self.memory_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, self.memory_path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    # ================================================================
    # High-level store methods (only called on significant events)
    # ================================================================

    def store_improvement(
        self,
        challenge_name: str,
        task_description: str,
        turn: int,
        action: str,
        plan: str,
        code: str,
        new_score: float,
        prev_best_score: Optional[float],
        score_direction: str = "lower",
    ):
        """Store when a new best score is achieved."""
        model_type = detect_model_type(code or plan)
        prev_model_type = ""  # Could be passed in future for richer insight

        if prev_best_score is not None:
            delta = new_score - prev_best_score
            pct_change = abs(delta / prev_best_score * 100) if prev_best_score != 0 else 0
            # Concise insight: what changed and by how much
            insight = (
                f"Score: {prev_best_score} -> {new_score} ({pct_change:.1f}% improvement). "
                f"Method: {model_type}. "
                f"Key change: {_summarize_plan(plan)}"
            )
        else:
            insight = (
                f"Baseline score: {new_score}. "
                f"Method: {model_type}. "
                f"Approach: {_summarize_plan(plan)}"
            )

        entry = MemoryEntry(
            challenge_name=challenge_name,
            task_description=task_description,
            turn=turn,
            action=action,
            plan=plan,
            model_type=model_type,
            score=new_score,
            score_improved=prev_best_score is not None,
            buggy=False,
            insight=insight,
            entry_type="improvement",
        )
        self._append_entry(entry)

    def store_debug_fix(
        self,
        challenge_name: str,
        task_description: str,
        turn: int,
        plan: str,
        code: str,
        error_type: str,
        error_message: str,
        fix_description: str,
    ):
        """Store when a debug action successfully fixes a bug (AIDE only)."""
        model_type = detect_model_type(code or plan)
        # Actionable insight: what broke, why, how to avoid
        insight = (
            f"Bug: {error_type}"
            f"{': ' + error_message[:100] if error_message else ''}. "
            f"Fix: {_summarize_plan(fix_description)}. "
            f"Avoid: {_error_avoidance_tip(error_type)}"
        )

        entry = MemoryEntry(
            challenge_name=challenge_name,
            task_description=task_description,
            turn=turn,
            action="debug",
            plan=plan,
            model_type=model_type,
            score=None,
            score_improved=False,
            buggy=False,
            insight=insight,
            entry_type="debug_fix",
        )
        self._append_entry(entry)

    def store_task_summary(
        self,
        challenge_name: str,
        task_description: str,
        total_turns: int,
        best_score: Optional[float],
        baseline_score: Optional[float],
        best_plan: str,
        best_code: str,
        success: bool,
    ):
        """Store a summary when a task completes."""
        model_type = detect_model_type(best_code or best_plan)

        if success and best_score is not None:
            improvement = ""
            if baseline_score is not None and baseline_score != 0:
                pct = abs((best_score - baseline_score) / baseline_score * 100)
                improvement = f" ({pct:.1f}% improvement over baseline)"
            insight = (
                f"Best model: {model_type}. "
                f"Score: {best_score}{improvement}. "
                f"Strategy: {_summarize_plan(best_plan)}"
            )
        else:
            insight = (
                f"Task failed. Model tried: {model_type}. "
                f"Issue: {_summarize_plan(best_plan) if best_plan else 'no valid approach found'}"
            )

        entry = MemoryEntry(
            challenge_name=challenge_name,
            task_description=task_description,
            turn=total_turns,
            action="summary",
            plan=best_plan,
            model_type=model_type,
            score=best_score,
            score_improved=False,
            buggy=False,
            insight=insight,
            entry_type="task_summary",
        )
        self._append_entry(entry)

    # ================================================================
    # Retrieval
    # ================================================================

    def retrieve(
        self,
        challenge_name: str = "",
        task_description: str = "",
        current_action: str = "",
        top_k: int = 20,
        max_per_task: int = 3,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant past experiences (always reads fresh from disk).

        Priority depends on current_action:
          - draft/explore: task_summary first (what approaches work)
          - improve/exploit: improvement first (what specific changes help)
          - debug: debug_fix first (what errors to avoid and how to fix)

        Limits entries per task to ensure diversity across tasks.
        Excludes entries from the current task.
        """
        raw = self._read_all()
        entries = [MemoryEntry.from_dict(d) for d in raw]

        # Exclude current task
        if challenge_name:
            entries = [e for e in entries if e.challenge_name != challenge_name]

        if not entries:
            return []

        # Priority depends on what the agent is currently doing
        if current_action in ("debug",):
            type_priority = {
                "debug_fix": 3.0,
                "improvement": 1.5,
                "task_summary": 1.0,
            }
        elif current_action in ("improve", "exploit"):
            type_priority = {
                "improvement": 3.0,
                "task_summary": 1.5,
                "debug_fix": 1.0,
            }
        else:  # draft, explore, or unknown
            type_priority = {
                "task_summary": 3.0,
                "improvement": 1.5,
                "debug_fix": 1.0,
            }

        scored = []
        for entry in entries:
            relevance = type_priority.get(entry.entry_type, 0.5)
            scored.append((relevance, entry))

        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        # Limit per task to ensure diversity
        result = []
        task_counts: Dict[str, int] = {}
        for _, entry in scored:
            cn = entry.challenge_name
            if task_counts.get(cn, 0) >= max_per_task:
                continue
            result.append(entry)
            task_counts[cn] = task_counts.get(cn, 0) + 1
            if len(result) >= top_k:
                break

        return result

    def format_for_prompt(
        self,
        challenge_name: str = "",
        task_description: str = "",
        current_action: str = "",
        top_k: int = 15,
    ) -> str:
        """
        Format retrieved memories as a text block for prompt injection.
        Returns empty string if no relevant memories exist.
        """
        entries = self.retrieve(
            challenge_name=challenge_name,
            task_description=task_description,
            current_action=current_action,
            top_k=top_k,
        )

        if not entries:
            return ""

        lines = [
            "=== CROSS-TASK EXPERIENCE MEMORY ===",
            "Below are key experiences from solving OTHER similar tasks. "
            "Use these to inform your approach — leverage strategies that worked well.",
            "",
        ]

        # Group by challenge for readability
        by_challenge: Dict[str, List[MemoryEntry]] = {}
        for entry in entries:
            by_challenge.setdefault(entry.challenge_name, []).append(entry)

        for cname, c_entries in by_challenge.items():
            lines.append(f"--- Task: {cname} ---")
            for entry in c_entries:
                tag = {
                    "task_summary": "SUMMARY",
                    "improvement": "IMPROVED",
                    "debug_fix": "DEBUG_FIX",
                }.get(entry.entry_type, "INFO")
                score_str = f"{entry.score:.6f}" if entry.score is not None else "N/A"
                lines.append(
                    f"  [{tag}] Model: {entry.model_type} | "
                    f"Score: {score_str} | {entry.insight}"
                )
            lines.append("")

        # Add model frequency stats from ALL memory entries (not just retrieved top_k)
        all_raw = self._read_all()
        all_entries = [MemoryEntry.from_dict(d) for d in all_raw]
        if challenge_name:
            all_entries = [e for e in all_entries if e.challenge_name != challenge_name]
        if all_entries:
            model_counts: Dict[str, int] = {}
            for entry in all_entries:
                if entry.model_type and entry.model_type != "unknown":
                    for m in entry.model_type.split(", "):
                        model_counts[m] = model_counts.get(m, 0) + 1
            if model_counts:
                sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
                freq_str = ", ".join(f"{name} ({count}x)" for name, count in sorted_models)
                lines.append(f"Model usage frequency across tasks: {freq_str}")
                lines.append("Consider exploring under-represented approaches for diversity.")
                lines.append("")

        lines.append("=== END CROSS-TASK MEMORY ===")
        return "\n".join(lines)
