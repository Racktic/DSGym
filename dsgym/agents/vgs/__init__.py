"""
VGS-MLE: Value-Guided Search for MLE agents.

Structured reasoning with Explore/Exploit/Terminate framework
for overcoming simplicity bias in data science competition tasks.
"""

from .structured_output import (
    StructuredOutput,
    SearchState,
    Candidate,
    ValueAction,
    Decision,
    Attempt,
    parse_structured_output,
    build_search_state_xml,
)
from .structured_output import parse_aide_output
from .teacher_agent import TeacherAgent
from .aide_agent import AIDEAgent
from .memory import CrossTaskMemory

__all__ = [
    "StructuredOutput",
    "SearchState",
    "Candidate",
    "ValueAction",
    "Decision",
    "Attempt",
    "parse_structured_output",
    "parse_aide_output",
    "build_search_state_xml",
    "TeacherAgent",
    "AIDEAgent",
    "CrossTaskMemory",
]
