"""
SAR 분석 시스템 - Co-Scientist 기반 전문가 협업 시스템

이 패키지는 구조-활성 관계(SAR) 분석을 위한 전문가 협업 시스템을 제공합니다.
Co-Scientist 방법론을 기반으로 구조화학, 생체분자 상호작용, QSAR 전문가들이
Activity Cliff 쌍을 분석하여 과학적 가설을 생성하고 평가합니다.

모듈 구성:
- llm_client: OpenAI/Google Gemini API 통합 클라이언트
- experts: 전문가 에이전트 클래스들 (구조화학, 생체분자상호작용, QSAR, 평가)
- system: 메인 시스템 로직 및 결과 표시 함수들

주요 함수:
- run_online_discussion_system: 메인 SAR 분석 시스템
- display_simplified_results: 분석 결과 표시
- display_docking_results: 도킹 시뮬레이션 결과 표시
"""

from .system import run_online_discussion_system, display_simplified_results, display_docking_results

__all__ = ['run_online_discussion_system', 'display_simplified_results', 'display_docking_results']