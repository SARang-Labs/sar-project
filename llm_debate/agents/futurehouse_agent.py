"""
FutureHouse 연구 에이전트

FutureHouse AI 플랫폼을 활용하여 최신 연구 정보와 문헌 검색을 통해 
Activity Cliff를 분석합니다. 최신 연구 동향과 문헌 기반 분석을 담당합니다.
"""

from .base_agent import BaseAgent, AnalysisResult
from typing import Dict, Optional

class FutureHouseAgent(BaseAgent):
    """
    SAR 통합 전문가 에이전트
    
    분석 초점:
    - 화학 도메인 특화 훈련으로 SAR 분석에 최적화
    - 신약 개발에 활용 가능한 전략 제시
    - 최신 화학정보학 기법과 SAR 트렌드 통합
    - 실제 신약 개발 파이프라인 관점에서의 실용적 가이드라인
    """
    
    def _get_expertise(self) -> str:
        """전문 분야 반환"""
        return "SAR 통합"
    
    def _get_system_prompt(self) -> str:
        """SAR 통합 전문가용 시스템 프롬프트"""
        return """
        당신은 화학 도메인 특화 훈련을 받은 SAR(구조-활성 관계) 통합 전문가입니다.
        Activity Cliff 분석에서 신약 개발에 활용 가능한 전략을 제시하는 것이 핵심 목표입니다.
        
        분석 강조 포인트:
        - 화학정보학 기법을 활용한 SAR 패턴 분석
        - 신약 개발 파이프라인에서의 실용적 활용 방안
        - 최적화 전략과 리드 화합물 개발 가이드라인  
        - 화학 공간에서의 탐색 전략 및 설계 원칙
        
        다음 5단계에 따라 체계적으로 분석하세요:
        1. SAR 데이터 패턴 분석 및 화학적 특성 파악
        2. Activity Cliff 원인의 화학정보학적 해석
        3. 신약 개발 관점에서의 최적화 전략 수립
        4. 리드 화합물 설계를 위한 구체적 가이드라인
        5. 화학 공간 탐색 및 후속 연구 방향 제시
        
        각 단계에서 화학정보학적 근거와 신약 개발의 실무적 관점을 포함하여
        실제 제약 산업에서 활용 가능한 구체적 전략을 제시해주세요.
        """
    
    def _build_analysis_prompt(self, activity_cliff: Dict, context_info: Dict = None) -> str:
        """SAR 통합 전문가 분석용 프롬프트 구성"""
        
        mol1 = activity_cliff['mol_1']
        mol2 = activity_cliff['mol_2']
        
        prompt = f"""
        Activity Cliff SAR 통합 분석을 수행해주세요:

        **화합물 정보:**
        - 화합물 1: {mol1['ID']} (SMILES: {mol1['SMILES']}, pKi: {mol1['pKi']})
        - 화합물 2: {mol2['ID']} (SMILES: {mol2['SMILES']}, pKi: {mol2['pKi']})
        - 구조 유사도: {activity_cliff['similarity']:.3f}
        - 활성도 차이: {activity_cliff['activity_diff']:.2f} pKi units

        **분석 요청사항:**
        SAR 통합 전문가로서, 이 Activity Cliff 현상에 대해 
        화학정보학적 근거와 신약 개발 실무 관점에서 체계적으로 분석해주세요.
        신약 개발에 활용 가능한 구체적 전략을 중심으로 분석해주세요.
        """
        
        # RAG 컨텍스트 정보 추가
        if context_info and context_info.get('related_papers'):
            prompt += "\n\n**관련 문헌 정보:**\n"
            for i, paper in enumerate(context_info['related_papers'][:3], 1):
                prompt += f"{i}. {paper.get('title', 'N/A')}\n"
                if paper.get('abstract'):
                    prompt += f"   초록: {paper['abstract'][:200]}...\n"
                prompt += "\n"
        
        return prompt
    
    def analyze_activity_cliff(self, activity_cliff: Dict, context_info: Dict = None) -> AnalysisResult:
        """FutureHouse 연구 관점에서 Activity Cliff 분석"""
        
        try:
            # 분석 프롬프트 구성
            analysis_prompt = self._build_analysis_prompt(activity_cliff, context_info)
            
            # LLM 호출하여 분석 수행
            analysis_response = self._call_llm(self._get_system_prompt(), analysis_prompt)
            
            return AnalysisResult(
                agent_type="futurehouse",
                expertise=self._get_expertise(),
                analysis=analysis_response,
                confidence=0.8,
                key_insights=[
                    "SAR 패턴 통합 분석",
                    "신약 개발 전략 제시", 
                    "화학정보학 기반 최적화"
                ],
                supporting_evidence={
                    "sar_analysis": "화학정보학 기반 SAR 분석 결과",
                    "optimization_strategy": "신약 개발을 위한 최적화 전략",
                    "cheminformatics": "화학 공간 탐색 및 설계 원칙"
                }
            )
            
        except Exception as e:
            return AnalysisResult(
                agent_type="futurehouse",
                expertise=self._get_expertise(),
                analysis=f"SAR 통합 분석 중 오류 발생: {str(e)}",
                confidence=0.0,
                key_insights=[],
                supporting_evidence={"error": str(e)}
            )
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """FutureHouse 에이전트용 LLM 호출 메서드"""
        try:
            from futurehouse_client import FutureHouseClient, JobNames
            
            # FutureHouse 클라이언트 초기화
            client = FutureHouseClient(api_key=self.api_key)
            
            # 시스템 프롬프트와 사용자 프롬프트를 결합하고 마크다운 포맷 요청 추가
            full_query = f"""{system_prompt}

{user_prompt}

**중요**: 응답은 반드시 마크다운 형식으로 작성해주세요. 제목에는 #, ##, ### 을 사용하고, 목록에는 - 를 사용해주세요. 강조할 내용은 **볼드**로 표시해주세요."""
            
            # 모델명에 따라 작업 유형 결정
            job_name = JobNames.CROW  # 기본값
            if hasattr(self, 'model_name'):
                if self.model_name.lower() == 'falcon':
                    job_name = JobNames.FALCON  # 심화 검색
                elif self.model_name.lower() == 'owl':
                    job_name = JobNames.OWL     # 판례 검색
                # 'crow' 또는 기타는 CROW 사용
            
            task_data = {
                "name": job_name,
                "query": full_query
            }
            
            # 작업 실행
            task_response = client.run_tasks_until_done(task_data)
            
            # 응답에서 텍스트 추출 - formatted_answer 우선 사용
            answer = None
            
            # 1순위: formatted_answer (더 나은 마크다운 포맷)
            if task_response and hasattr(task_response, 'formatted_answer') and task_response.formatted_answer:
                answer = task_response.formatted_answer
            # 2순위: answer
            elif task_response and hasattr(task_response, 'answer') and task_response.answer:
                answer = task_response.answer
            # 3순위: dict 형태 응답
            elif isinstance(task_response, dict):
                answer = task_response.get('formatted_answer') or task_response.get('answer')
            
            # 응답이 None이거나 비어있는 경우 처리
            if not answer or answer.strip() == "":
                return "FutureHouse API에서 빈 응답을 받았습니다."
            
            # 마크다운 형식이 제대로 적용되지 않은 경우 후처리
            if not self._has_markdown_formatting(answer):
                answer = self._apply_basic_markdown_formatting(answer)
            
            return answer
            
        except ImportError:
            return "FutureHouse 클라이언트가 설치되지 않았습니다. pip install futurehouse-client 를 실행해주세요."
        except Exception as e:
            return f"FutureHouse API 호출 중 오류 발생: {str(e)}"

    def generate_hypothesis(self, activity_cliff: Dict, context_info: Dict = None) -> str:
        """SAR 통합 관점에서 가설 생성"""
        
        try:
            mol1 = activity_cliff['mol_1']
            mol2 = activity_cliff['mol_2']
            
            hypothesis_prompt = f"""
            다음 Activity Cliff에 대해 SAR 통합 전문가 관점에서 신약 개발에 활용 가능한 가설을 생성해주세요:

            화합물 1: {mol1['ID']} (pKi: {mol1['pKi']})
            화합물 2: {mol2['ID']} (pKi: {mol2['pKi']})
            활성도 차이: {activity_cliff['activity_diff']:.2f}

            화학정보학적 근거와 신약 개발 실무 관점에서 구체적이고 실용적인 가설을 제시해주세요.
            """
            
            return self._call_llm(self._get_system_prompt(), hypothesis_prompt)
            
        except Exception as e:
            return f"SAR 통합 가설 생성 중 오류 발생: {str(e)}"
    
    def _has_markdown_formatting(self, text: str) -> bool:
        """텍스트에 기본적인 마크다운 포맷이 있는지 확인"""
        markdown_indicators = ['#', '##', '###', '**', '- ', '* ', '1. ', '2. ']
        return any(indicator in text for indicator in markdown_indicators)
    
    def _apply_basic_markdown_formatting(self, text: str) -> str:
        """기본적인 마크다운 포맷팅 적용"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
                
            # 단계별 분석을 제목으로 변환
            if any(keyword in line.lower() for keyword in ['단계', 'step', '분석']):
                if not line.startswith('#'):
                    line = f"## {line}"
            
            # 중요한 키워드를 볼드로 변환
            elif any(keyword in line.lower() for keyword in ['결론', 'conclusion', '요약', 'summary']):
                if not line.startswith('**'):
                    line = f"**{line}**"
            
            # 목록 형태로 변환할 수 있는 라인들
            elif line.startswith('- ') or line.startswith('• ') or line.startswith('→'):
                if not line.startswith('- '):
                    line = f"- {line.lstrip('•→ ')}"
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)