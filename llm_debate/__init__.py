"""
최적 프롬프트 토론 시스템 패키지

이 패키지는 Activity Cliff 분석을 위한 최적 프롬프트 토론 시스템을 제공합니다.
3명의 전문가 에이전트가 토론을 통해 최적의 프롬프트와 가설을 생성합니다.
"""

__version__ = "2.0.0"
__author__ = "SAR 분석 시스템"

from .agents.base_agent import BaseAgent
from .agents.structural_agent import StructuralAgent
from .agents.biological_agent import BiologicalAgent
from .agents.futurehouse_agent import FutureHouseAgent
from .debate.optimal_prompt_debate_manager import OptimalPromptDebateManager

__all__ = [
    "BaseAgent",
    "StructuralAgent", 
    "BiologicalAgent",
    "FutureHouseAgent",
    "OptimalPromptDebateManager"
]