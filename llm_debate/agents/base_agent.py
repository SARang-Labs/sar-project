"""
기본 에이전트 클래스

모든 전문가 에이전트가 상속받는 기본 클래스입니다.
공통 기능과 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import time
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    """분석 결과를 담는 데이터 클래스"""
    agent_name: str
    expertise: str
    analysis: str
    round_number: int
    timestamp: float
    confidence_score: float = 0.0
    citations: List[str] = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []

class BaseAgent(ABC):
    """
    모든 LLM 에이전트의 기본 클래스
    
    각 전문가 에이전트는 이 클래스를 상속받아 구현됩니다.
    공통 기능: 모델 설정, 프롬프트 관리, 응답 파싱 등
    """
    
    def __init__(self, 
                 model_provider: str = "openai",
                 model_name: str = "gpt-4o",
                 api_key: str = None,
                 temperature: float = 0.3,
                 max_tokens: int = 2000):
        """
        에이전트 초기화
        
        Args:
            model_provider: 모델 제공업체 ("openai", "gemini", "futurehouse")
            model_name: 사용할 모델명
            api_key: API 키
            temperature: 생성 온도 (0.0-1.0)
            max_tokens: 최대 토큰 수
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.expertise = self._get_expertise()
        self.agent_name = self.__class__.__name__
        
        # 모델 클라이언트 초기화
        self._initialize_client()
        
    def _initialize_client(self):
        """모델 제공업체별 클라이언트 초기화"""
        if self.model_provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        elif self.model_provider == "gemini":
            import google.generativeai as genai
            if self.api_key:
                try:
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel(self.model_name)
                except Exception as e:
                    print(f"⚠️ Gemini API 키 설정 실패: {e}")
                    self.client = None
            else:
                print("⚠️ Gemini API 키가 제공되지 않음")
                self.client = None
        elif self.model_provider == "futurehouse":
            # FutureHouse 에이전트는 자체적으로 클라이언트를 설정
            try:
                from futurehouse_client import FutureHouseClient
                if self.api_key:
                    self.client = FutureHouseClient(api_key=self.api_key)
                else:
                    print("⚠️ FutureHouse API 키가 제공되지 않음")
                    self.client = None
            except ImportError:
                print("⚠️ FutureHouse 클라이언트가 설치되지 않음")
                self.client = None
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환합니다"""
        return {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "expertise": self.expertise
        }
    
    @abstractmethod
    def _get_expertise(self) -> str:
        """전문 분야를 반환합니다"""
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트를 반환합니다"""
        pass
    
    @abstractmethod
    def _build_analysis_prompt(self, activity_cliff: Dict, context_info: Dict = None) -> str:
        """분석용 프롬프트를 구성합니다"""
        pass
    
    def analyze_activity_cliff(self, 
                              activity_cliff: Dict, 
                              context_info: Dict = None,
                              round_number: int = 1) -> AnalysisResult:
        """
        Activity Cliff를 분석합니다
        
        Args:
            activity_cliff: Activity cliff 데이터
            context_info: RAG에서 가져온 문헌 정보
            round_number: 토론 라운드 번호
            
        Returns:
            AnalysisResult: 분석 결과
        """
        try:
            # 프롬프트 구성
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_analysis_prompt(activity_cliff, context_info)
            
            # LLM 호출
            response = self._call_llm(system_prompt, user_prompt)
            
            # 결과 파싱 및 반환
            return AnalysisResult(
                agent_name=self.agent_name,
                expertise=self.expertise,
                analysis=response,
                round_number=round_number,
                timestamp=time.time(),
                confidence_score=self._calculate_confidence(response)
            )
            
        except Exception as e:
            error_analysis = f"분석 중 오류 발생: {str(e)}"
            return AnalysisResult(
                agent_name=self.agent_name,
                expertise=self.expertise,
                analysis=error_analysis,
                round_number=round_number,
                timestamp=time.time(),
                confidence_score=0.0
            )
    
    def review_other_analyses(self, 
                             other_analyses: List[AnalysisResult],
                             activity_cliff: Dict,
                             context_info: Dict = None) -> AnalysisResult:
        """
        다른 에이전트들의 분석을 검토합니다 (라운드 2용)
        
        Args:
            other_analyses: 다른 에이전트들의 분석 결과
            activity_cliff: Activity cliff 데이터
            context_info: RAG 문헌 정보
            
        Returns:
            AnalysisResult: 검토 결과
        """
        try:
            system_prompt = self._get_review_system_prompt()
            user_prompt = self._build_review_prompt(other_analyses, activity_cliff, context_info)
            
            response = self._call_llm(system_prompt, user_prompt)
            
            return AnalysisResult(
                agent_name=self.agent_name,
                expertise=self.expertise,
                analysis=response,
                round_number=2,
                timestamp=time.time(),
                confidence_score=self._calculate_confidence(response)
            )
            
        except Exception as e:
            error_analysis = f"검토 중 오류 발생: {str(e)}"
            return AnalysisResult(
                agent_name=self.agent_name,
                expertise=self.expertise,
                analysis=error_analysis,
                round_number=2,
                timestamp=time.time(),
                confidence_score=0.0
            )
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """모델 제공업체별 LLM 호출"""
        if self.model_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
            
        elif self.model_provider == "gemini":
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
            return response.text
            
        elif self.model_provider == "futurehouse":
            # FutureHouse 에이전트는 자체적으로 _call_llm 메서드를 오버라이드해야 함
            raise NotImplementedError("FutureHouse 에이전트는 자체 _call_llm 메서드를 오버라이드해야 합니다")
            
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def _get_review_system_prompt(self) -> str:
        """라운드 2 검토용 시스템 프롬프트"""
        return f"""
        당신은 {self.expertise} 전문가입니다.
        다른 전문가들의 Activity Cliff 분석을 검토하고 비판적 의견을 제시해주세요.
        
        검토 관점:
        1. 동의하는 부분과 그 이유
        2. 문제가 있다고 생각하는 부분과 구체적 근거
        3. 놓친 중요한 관점이나 추가 고려사항
        4. 최종 통합 결론 제안
        
        각 의견은 구체적 과학적 근거와 함께 제시하세요.
        """
    
    def _build_review_prompt(self, 
                           other_analyses: List[AnalysisResult], 
                           activity_cliff: Dict,
                           context_info: Dict = None) -> str:
        """라운드 2 검토용 프롬프트 구성"""
        prompt = f"""
        다음은 다른 전문가들의 같은 Activity Cliff에 대한 분석입니다:
        
        **분석 대상 Activity Cliff:**
        - 화합물 A: {activity_cliff['mol_1']['SMILES']} (pKi: {activity_cliff['mol_1']['pKi']:.2f})
        - 화합물 B: {activity_cliff['mol_2']['SMILES']} (pKi: {activity_cliff['mol_2']['pKi']:.2f})
        - 유사도: {activity_cliff['similarity']:.3f}
        - 활성도 차이: {activity_cliff['activity_diff']:.2f}
        
        """
        
        # 다른 에이전트들의 분석 결과 추가
        for i, analysis in enumerate(other_analyses, 1):
            if analysis.agent_name != self.agent_name:  # 자신의 분석은 제외
                prompt += f"""
        **{analysis.expertise} 전문가 분석:**
        {analysis.analysis}
        
        """
        
        # RAG 정보가 있으면 추가
        if context_info:
            prompt += f"""
        **참고 문헌 정보:**
        - 제목: {context_info.get('title', 'N/A')}
        - 초록: {context_info.get('abstract', 'N/A')}
        
        """
        
        prompt += f"""
        당신의 전문 영역인 {self.expertise} 관점에서 위 분석들을 검토해주세요.
        """
        
        return prompt
    
    def _calculate_confidence(self, analysis: str) -> float:
        """분석 결과의 신뢰도 점수를 계산합니다"""
        # 간단한 휴리스틱 기반 신뢰도 계산
        # 실제로는 더 정교한 방법 사용 가능
        
        confidence = 0.5  # 기본값
        
        # 분석 길이가 적절한지 확인
        if 500 <= len(analysis) <= 3000:
            confidence += 0.2
            
        # 과학적 용어 포함 여부 확인
        scientific_terms = ['분자', '구조', '활성', '결합', '상호작용', '메커니즘', 'SMILES', 'pKi']
        term_count = sum(1 for term in scientific_terms if term in analysis)
        confidence += min(0.2, term_count * 0.05)
        
        # 구체적 수치 제시 여부 확인
        if any(char.isdigit() for char in analysis):
            confidence += 0.1
            
        return min(1.0, confidence)
    
