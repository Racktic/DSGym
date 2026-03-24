"""
XML structured output parser for VGS-MLE.

Parses two phases of structured output:
- Exploration: <search_state phase="exploration"> + <python>
- Optimization: <search_state> + <candidates> + <value_estimation> + <decision> + <python>

All parsing is regex-based for robustness with LLM-generated XML.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Literal


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class Attempt:
    """A single historical attempt record."""
    step: int
    action: str  # "explore" or "exploit"
    idea: str
    predicted_delta: float
    actual_delta: float


@dataclass
class SearchState:
    """Parsed <search_state> block."""
    phase: Literal["exploration", "optimization"]
    step: int
    goal: Optional[str] = None
    best_score: Optional[float] = None
    baseline_score: Optional[float] = None
    current_score: Optional[float] = None
    history: List[Attempt] = field(default_factory=list)


@dataclass
class Candidate:
    """A single candidate idea."""
    id: str  # "A", "B", "C"
    description: str


@dataclass
class ValueAction:
    """A single value estimation entry."""
    action_type: str  # "explore", "exploit", or "terminate"
    idea: Optional[str] = None  # idea ID for explore actions
    predicted_delta: float = 0.0
    reasoning: str = ""


@dataclass
class Decision:
    """Parsed <decision> block."""
    action: str  # "explore", "exploit", "terminate"
    idea: Optional[str] = None
    reasoning: str = ""


@dataclass
class StructuredOutput:
    """Complete parsed structured output from one turn."""
    search_state: SearchState
    candidates: List[Candidate] = field(default_factory=list)
    value_estimation: List[ValueAction] = field(default_factory=list)
    decision: Optional[Decision] = None
    python_code: Optional[str] = None
    raw_text: str = ""

    @property
    def is_exploration(self) -> bool:
        return self.search_state.phase == "exploration"

    @property
    def is_terminate(self) -> bool:
        return self.decision is not None and self.decision.action == "terminate"

    def get_chosen_value_score(self) -> Optional[float]:
        """Get the predicted_delta for the chosen action."""
        if self.decision is None:
            return None
        for va in self.value_estimation:
            if va.action_type == self.decision.action:
                if va.action_type == "explore" and va.idea != self.decision.idea:
                    continue
                return va.predicted_delta
        return None


# ============================================================
# Parser functions
# ============================================================

def parse_structured_output(text: str) -> StructuredOutput:
    """
    Parse XML structured output from LLM response.

    Handles both exploration and optimization phases.
    Tolerant of malformed XML — uses regex fallbacks.

    Args:
        text: Raw LLM response text

    Returns:
        StructuredOutput dataclass

    Raises:
        ValueError: If no <search_state> block found
    """
    result = StructuredOutput(
        search_state=_parse_search_state(text),
        raw_text=text,
    )

    result.python_code = _parse_python_code(text)

    if result.search_state.phase == "optimization":
        result.candidates = _parse_candidates(text)
        result.value_estimation = _parse_value_estimation(text)
        result.decision = _parse_decision(text)

    return result


def _parse_search_state(text: str) -> SearchState:
    """Extract and parse <search_state> block."""
    match = re.search(r"<search_state>(.*?)</search_state>", text, re.DOTALL)
    if not match:
        raise ValueError("No <search_state> block found in response")

    block = match.group(1)

    phase = _extract_tag_text(block, "phase") or "exploration"
    step = int(_extract_tag_text(block, "step") or "0")
    goal = _extract_tag_text(block, "goal")
    best_score = _safe_float(_extract_tag_text(block, "best_score"))
    baseline_score = _safe_float(_extract_tag_text(block, "baseline_score"))
    current_score = _safe_float(_extract_tag_text(block, "current_score"))

    history: List[Attempt] = []
    for m in re.finditer(
        r'<attempt\s+'
        r'step="(\d+)"\s+'
        r'action="(\w+)"\s+'
        r'idea="([^"]*)"\s+'
        r'predicted_delta="([^"]*)"\s+'
        r'actual_delta="([^"]*)"',
        block,
    ):
        history.append(Attempt(
            step=int(m.group(1)),
            action=m.group(2),
            idea=m.group(3),
            predicted_delta=float(m.group(4)),
            actual_delta=float(m.group(5)),
        ))

    return SearchState(
        phase=phase,
        step=step,
        goal=goal,
        best_score=best_score,
        baseline_score=baseline_score,
        current_score=current_score,
        history=history,
    )


def _parse_candidates(text: str) -> List[Candidate]:
    """Extract <candidates> block and parse <idea> entries."""
    match = re.search(r"<candidates>(.*?)</candidates>", text, re.DOTALL)
    if not match:
        return []
    block = match.group(1)
    candidates = []
    for m in re.finditer(r'<idea\s+id="([^"]+)">(.*?)</idea>', block, re.DOTALL):
        candidates.append(Candidate(
            id=m.group(1),
            description=m.group(2).strip(),
        ))
    return candidates


def _parse_value_estimation(text: str) -> List[ValueAction]:
    """Extract <value_estimation> block and parse <action> entries."""
    match = re.search(
        r"<value_estimation>(.*?)</value_estimation>", text, re.DOTALL
    )
    if not match:
        return []
    block = match.group(1)

    actions = []
    for m in re.finditer(
        r'<action\s+type="(\w+)"(?:\s+idea="([^"]*)")?\s*>(.*?)</action>',
        block,
        re.DOTALL,
    ):
        action_type = m.group(1)
        idea = m.group(2)
        inner = m.group(3)

        predicted_delta = _safe_float(
            _extract_tag_text(inner, "predicted_delta")
        ) or 0.0
        reasoning = _extract_tag_text(inner, "reasoning") or ""

        actions.append(ValueAction(
            action_type=action_type,
            idea=idea,
            predicted_delta=predicted_delta,
            reasoning=reasoning,
        ))
    return actions


def _parse_decision(text: str) -> Optional[Decision]:
    """Extract <decision> block."""
    match = re.search(r"<decision>(.*?)</decision>", text, re.DOTALL)
    if not match:
        return None
    block = match.group(1)

    action = _extract_tag_text(block, "action") or ""
    idea = _extract_tag_text(block, "idea")
    reasoning = _extract_tag_text(block, "reasoning") or ""

    return Decision(action=action, idea=idea, reasoning=reasoning)


def _parse_python_code(text: str) -> Optional[str]:
    """Extract code from <python>...</python> tags.

    Uses the same regex pattern as AllocatedCodeEnv._parse_action().
    """
    match = re.search(r"<python>(.*?)</python>", text, re.DOTALL)
    if match:
        code = match.group(1)
        if "```python" in code and "```" in code:
            inner = re.search(r"```python(.*?)```", code, re.DOTALL)
            if inner:
                return inner.group(1)
        return code
    return None


# ============================================================
# Builder functions
# ============================================================

def build_search_state_xml(
    phase: str,
    step: int,
    best_score: Optional[float] = None,
    baseline_score: Optional[float] = None,
    history: Optional[List[Attempt]] = None,
    goal: Optional[str] = None,
) -> str:
    """Build <search_state> XML string for injection into conversation."""
    lines = ["<search_state>"]
    lines.append(f"  <phase>{phase}</phase>")
    lines.append(f"  <step>{step}</step>")
    if goal:
        lines.append(f"  <goal>{goal}</goal>")
    if best_score is not None:
        lines.append(f"  <best_score>{best_score}</best_score>")
    if baseline_score is not None:
        lines.append(f"  <baseline_score>{baseline_score}</baseline_score>")
    if history:
        lines.append("  <history>")
        for a in history:
            lines.append(
                f'    <attempt step="{a.step}" action="{a.action}" '
                f'idea="{a.idea}" predicted_delta="{a.predicted_delta}" '
                f'actual_delta="{a.actual_delta}" />'
            )
        lines.append("  </history>")
    lines.append("</search_state>")
    return "\n".join(lines)


# ============================================================
# Helpers
# ============================================================

def parse_aide_output(text: str) -> StructuredOutput:
    """
    Parse AIDE-format output (search_state + python, no decision block).

    AIDE's action is determined by the agent (hard-coded rules), not the model,
    so there is no <decision> block in the response.
    """
    return StructuredOutput(
        search_state=_parse_search_state(text),
        python_code=_parse_python_code(text),
        raw_text=text,
    )


def parse_eet_output(text: str) -> StructuredOutput:
    """
    Parse EET-format output (simplified: no candidates, no value_estimation).

    Reuses existing parser components for search_state, decision, and python.
    """
    return StructuredOutput(
        search_state=_parse_search_state(text),
        decision=_parse_decision(text),
        python_code=_parse_python_code(text),
        raw_text=text,
    )


def _extract_tag_text(text: str, tag: str) -> Optional[str]:
    """Extract text content of a simple XML tag."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Parse float from string, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
