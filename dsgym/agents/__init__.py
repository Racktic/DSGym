"""
DSGym Agents Module

ReActDSAgent scaffold with integrated lightweight environment system.
"""

from .base_agent import BaseAgent
from .react_ds_agent import ReActDSAgent
from .dspredict_react_agent import DSPredictReActAgent
from .vgs import TeacherAgent, VGSAgent, EETAgent, AIDEAgent

__all__ = [
    'BaseAgent',
    'ReActDSAgent',
    'DSPredictReActAgent',
    'TeacherAgent',
    'VGSAgent',
    'EETAgent',
    'AIDEAgent',
]
