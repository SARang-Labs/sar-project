"""
Co-Scientist 기반 SAR 분석 시스템 - 3단계 전문가 협업 가설 생성

이 모듈은 Co-Scientist 방법론을 SAR 분석에 특화하여 구현합니다:
- Phase 1: 데이터 준비 + RAG 통합 (기존 시스템 활용)
- Phase 2: 전문가 분석 (3개 도메인 전문가 독립 생성)
- Phase 3: 전문가 평가 (HypothesisEvaluationExpert 기반 품질 평가)
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from openai import OpenAI
from utils import search_pubmed_for_context, get_activity_cliff_summary

class UnifiedLLMClient:
    """통합 LLM 클라이언트"""
    
    def __init__(self, api_key: str, llm_provider: str = "OpenAI"):
        self.llm_provider = llm_provider
        if llm_provider == "OpenAI":
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4o"
        elif llm_provider in ["Gemini", "Google Gemini"]:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel("gemini-2.5-pro")
            self.model = "gemini-2.5-pro"
        else:
            raise ValueError(f"지원하지 않는 LLM 공급자: {llm_provider}")
    
    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """통합 응답 생성"""
        if self.llm_provider == "OpenAI":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        elif self.llm_provider in ["Gemini", "Google Gemini"]:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.client.generate_content(full_prompt)
            return response.text
        else:
            raise ValueError(f"지원하지 않는 LLM 공급자: {self.llm_provider}")


class StructuralChemistryExpert:
    """구조화학 전문가 에이전트"""
    
    def __init__(self, llm_client: UnifiedLLMClient):
        self.llm_client = llm_client
        self.persona = """당신은 20년 경력의 선임 약화학자로, 분자 구조와 전자적 특성 변화 분석의 전문가입니다.
        특히 분자 기하학, SMILES 구조 차이점, 화학적 직관과 구조-기능 관계 규명에 특화되어 있습니다."""
    
    def generate(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """구조화학 관점의 가설 생성"""
        prompt = self._build_structural_prompt(shared_context)
        hypothesis = self.llm_client.generate_response(self.persona, prompt, temperature=0.7)
        
        return {
            'agent_type': 'structural_chemistry',
            'agent_name': '구조화학 전문가',
            'hypothesis': hypothesis,
            'confidence': self._extract_confidence_from_text(hypothesis),
            'key_insights': self._extract_key_insights(hypothesis),
            'reasoning_steps': self._extract_reasoning_steps(hypothesis),
            'timestamp': time.time()
        }
    
    def _build_structural_prompt(self, shared_context: Dict[str, Any]) -> str:
        """구조화학 전문가용 특화 프롬프트 생성 - CoT.md 지침 반영"""
        cliff_summary = shared_context['cliff_summary']
        target_name = shared_context['target_name']  # target_name 추가
        high_active = cliff_summary['high_activity_compound']
        low_active = cliff_summary['low_activity_compound']
        metrics = cliff_summary['cliff_metrics']
        prop_diffs = cliff_summary['property_differences']
        
        literature_info = ""
        if shared_context.get('literature_context'):
            lit = shared_context['literature_context']
            literature_info = f"""
            **참고 문헌 정보 (RAG 검색 결과 - 할루시네이션 방지용):**
            - 제목: {lit.get('title', 'N/A')}
            - 초록: {lit.get('abstract', 'N/A')[:500]}...
            - PubMed ID: {lit.get('pmid', 'N/A')}
            - 키워드: {target_name}, 구조-활성 관계, Activity Cliff
            - 이 문헌을 전문가 지식의 근거로 활용하여 논리적 추론을 수행하세요.
            """
        
        # Few-Shot 예시 (실제 SAR 사례)
        few_shot_example = """
        **Few-Shot 예시 - 전문가 분석 과정 참조:**
        
        [예시] 벤조디아제핀 유도체 Activity Cliff 분석:
        구조 A: 클로르디아제폭시드 (pKi: 7.2) vs 구조 B: 디아제팜 (pKi: 8.9)
        
        1. 구조 비교: A는 N-옥사이드 형태, B는 7번 위치에 염소 치환
        2. 물리화학적 영향: N-옥사이드 제거로 전자밀도 증가, 지용성 향상 (LogP +0.8)
        3. 생체 상호작용: GABA 수용체와의 결합 기하학 개선, π-π 스택킹 강화
        4. 활성 변화 연결: 개선된 단백질 적합성으로 1.7 pKi 단위 활성 증가
        5. 추가 실험: 분자 도킹 시뮬레이션으로 결합 모드 확인, ADMET 예측
        
        [귀하의 분석 과제]
        """
        
        return f"""
        당신은 20년 경력의 선임 약화학자입니다. SAR과 Activity Cliff 분석에서 분자 구조와 전자적 특성 변화 분석의 전문가로서, 실제 신약 개발 현장에서 사용하는 체계적 분석 절차를 따라 정확하고 신뢰할 수 있는 가설을 생성해주세요.
        
        {few_shot_example}
        
        **Activity Cliff 분석 대상:**
        
        **화합물 정보:**
        - 고활성 화합물: {high_active['id']} (pKi: {high_active['pki']:.2f})
          SMILES: {high_active['smiles']}
        - 저활성 화합물: {low_active['id']} (pKi: {low_active['pki']:.2f})
          SMILES: {low_active['smiles']}
        
        **In-Context 구조적 특성 (할루시네이션 방지용):**
        - Tanimoto 유사도: {metrics['similarity']:.3f}
        - 활성도 차이: {metrics['activity_difference']:.2f}
        - 구조적 차이 유형: {metrics['structural_difference_type']}
        - 입체이성질체 여부: {metrics['is_stereoisomer_pair']}
        - 분자량 차이: {prop_diffs['mw_diff']:.2f} Da
        - LogP 차이: {prop_diffs['logp_diff']:.2f}
        - TPSA 차이: {prop_diffs.get('tpsa_diff', 0):.2f} Ų
        
        {literature_info}
        
        **단계별 Chain-of-Thought 분석 수행:**
        실제 약화학자가 사용하는 분석 절차를 따라 다음 5단계로 체계적으로 분석하세요:
        
        1. **구조 비교**: 두 구조 A와 B의 차이점을 정확히 식별하세요. SMILES 구조를 상세히 비교하여 치환기, 고리 구조, 입체화학의 정확한 변화를 기술하세요.
        
        2. **물리화학적 영향**: 식별된 변경이 소수성(LogP), 수소 결합 능력, 전자 분포, 극성 표면적(TPSA)에 미치는 영향을 추론하세요. 정량적 변화값을 활용하세요.
        
        3. **생체 상호작용 가설**: 이 변경이 표적 단백질 결합 친화도나 대사 안정성에 어떻게 작용할지 구체적인 분자 수준 메커니즘을 가설로 제시하세요.
        
        4. **활성 변화 연결**: 이 가설이 관찰된 Activity Cliff ({metrics['activity_difference']:.2f} pKi 단위 차이)를 어떻게 설명하는지 논리적으로 연결하세요.
        
        5. **추가 실험 제안**: 검증을 위한 분자 도킹, ADMET 예측, 계산화학 실험 등 후속 실험을 구체적으로 제안하세요.
        
        **필수 요구사항 - 전문가 수준의 분석:**
        1. 구체적 수치 데이터 포함 (LogP, MW, TPSA 등)
        2. 원자 단위 구조 차이 명시 (C-N 결합 → C-O 결합 등)
        3. 정량적 활성 예측 ("대략 1.5 pKi 단위 감소" 등)
        4. 구체적 실험 프로토콜 ("AutoDock4로 100회 도킹" 등)
        5. 특정 분자 대상 제시 ("메틸에스터 치환체" 등)
        
        **금지 사항 - 다음과 같은 모호한 표현 금지:**
        - "~일 것으로 예상된다", "~로 추정된다"
        - "가능성이 있다", "보인다", "생각된다"
        - "일반적으로", "대개", "보통"
        
        **실제 제약회사 수준의 분석을 수행하여 즉시 합성 대상으로 사용할 수 있는 구체적 가설을 제시하세요.**
        
        **결과 형식 (반드시 이 형식을 정확히 따르세요):**
        
        신뢰도: [구체적 수치와 근거, 예: 85% - RDKit 계산 결과와 문헌 근거 기반]
        
        핵심 가설: [구체적이고 전문적인 1-2문장, 예: "N-메틸기 추가로 인한 입체장애가 Asp381과의 수소결합을 방해하여 2.3 pKi 단위 활성 감소를 초래"]
        
        상세 분석:
        1. 구조 비교: [SMILES 구조의 정확한 차이점, 원자 번호와 결합 유형 명시]
        2. 물리화학적 영향: [LogP, TPSA, 분자량 변화의 구체적 수치와 의미]
        3. 생체 상호작용 가설: [특정 아미노산 잔기와의 상호작용 변화, 결합 에너지 추정]
        4. 활성 변화 연결: [정량적 구조-활성 관계 설명]
        5. 추가 실험 제안: [구체적 프로토콜과 예상 결과]
        
        분자 설계 제안: [후속 화합물의 구체적 구조 변경 전략]
        
        **중요: 모든 설명은 구체적 수치, 특정 분자 부위, 명확한 메커니즘을 포함해야 하며, '~일 것이다', '~로 추정된다' 같은 모호한 표현보다는 과학적 근거에 기반한 확정적 분석을 제시하세요.**
        """
    
    def _extract_confidence_from_text(self, hypothesis: str) -> float:
        """가설 텍스트에서 실제 신뢰도 값을 추출"""
        import re
        
        # "신뢰도: XX%" 패턴 찾기
        confidence_match = re.search(r'신뢰도:.*?(\d+)%', hypothesis)
        if confidence_match:
            confidence_value = int(confidence_match.group(1))
            return confidence_value / 100.0
        
        # 영어 패턴도 확인
        confidence_match = re.search(r'confidence:.*?(\d+)%', hypothesis, re.IGNORECASE)
        if confidence_match:
            confidence_value = int(confidence_match.group(1))
            return confidence_value / 100.0
        
        # 추출 실패 시 키워드 기반 계산으로 fallback
        return self._calculate_confidence_by_keywords(hypothesis)
    
    def _calculate_confidence_by_keywords(self, hypothesis: str) -> float:
        """가설의 신뢰도 계산 (단순 휴리스틱)"""
        confidence_indicators = [
            ('구체적인 메커니즘' in hypothesis or 'mechanism' in hypothesis.lower(), 0.2),
            ('실험' in hypothesis or 'experiment' in hypothesis.lower(), 0.15),
            ('문헌' in hypothesis or 'literature' in hypothesis.lower(), 0.15),
            ('SMILES' in hypothesis or 'smiles' in hypothesis.lower(), 0.1),
            ('수소결합' in hypothesis or 'hydrogen bond' in hypothesis.lower(), 0.1),
            ('입체' in hypothesis or 'stereo' in hypothesis.lower(), 0.1),
            ('분자량' in hypothesis or 'molecular weight' in hypothesis.lower(), 0.1),
            ('활성' in hypothesis or 'activity' in hypothesis.lower(), 0.1)
        ]
        
        base_confidence = 0.5
        for indicator, weight in confidence_indicators:
            if indicator:
                base_confidence += weight
        
        return min(base_confidence, 1.0)
    
    def _extract_key_insights(self, hypothesis: str) -> List[str]:
        """가설에서 핵심 인사이트 추출"""
        # 간단한 키워드 기반 추출
        insights = []
        if '입체' in hypothesis or 'stereo' in hypothesis.lower():
            insights.append("입체화학적 차이가 핵심 요인")
        if '수소결합' in hypothesis or 'hydrogen bond' in hypothesis.lower():
            insights.append("수소결합 패턴 변화")
        if '소수성' in hypothesis or 'hydrophobic' in hypothesis.lower():
            insights.append("소수성 상호작용 차이")
        if not insights:
            insights.append("구조적 변화로 인한 활성 차이")
        return insights
    
    def _extract_reasoning_steps(self, hypothesis: str) -> List[str]:
        """추론 단계 추출"""
        # 번호나 단계별로 나뉜 부분 찾기
        steps = []
        lines = hypothesis.split('\n')
        current_step = ""
        
        for line in lines:
            line = line.strip()
            if any(marker in line for marker in ['1.', '2.', '3.', '4.', '5.', '**', '###']):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step.strip())
        
        return steps[:5]  # 최대 5단계


class BiomolecularInteractionExpert:
    """생체분자 상호작용 전문가 에이전트"""
    
    def __init__(self, llm_client: UnifiedLLMClient):
        self.llm_client = llm_client
        self.persona = """당신은 단백질-리간드 상호작용 메커니즘 분야의 세계적 권위자입니다.
        타겟 단백질과의 결합 방식 변화, 약리학적 관점과 생리활성 메커니즘 규명을 전문으로 합니다."""
    
    def generate(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """생체분자 상호작용 관점의 가설 생성"""
        prompt = self._build_interaction_prompt(shared_context)
        hypothesis = self.llm_client.generate_response(self.persona, prompt, temperature=0.7)
        
        return {
            'agent_type': 'biomolecular_interaction',
            'agent_name': '생체분자 상호작용 전문가',
            'hypothesis': hypothesis,
            'confidence': self._extract_confidence_from_text(hypothesis),
            'key_insights': self._extract_key_insights(hypothesis),
            'reasoning_steps': self._extract_reasoning_steps(hypothesis),
            'timestamp': time.time()
        }
    
    def _build_interaction_prompt(self, shared_context: Dict[str, Any]) -> str:
        """생체분자 상호작용 전문가용 특화 프롬프트 생성 - CoT.md 지침 반영"""
        cliff_summary = shared_context['cliff_summary']
        target_name = shared_context['target_name']
        high_active = cliff_summary['high_activity_compound']
        low_active = cliff_summary['low_activity_compound']
        metrics = cliff_summary['cliff_metrics']
        prop_diffs = cliff_summary['property_differences']
        
        literature_info = ""
        if shared_context.get('literature_context'):
            lit = shared_context['literature_context']
            literature_info = f"""
            **참고 문헌 정보 (RAG 검색 결과 - 할루시네이션 방지용):**
            - 제목: {lit.get('title', 'N/A')}
            - 초록: {lit.get('abstract', 'N/A')[:500]}...
            - PubMed ID: {lit.get('pmid', 'N/A')}
            - 키워드: {target_name}, 구조-활성 관계, Activity Cliff
            - 이 문헌을 전문가 지식의 근거로 활용하여 논리적 추론을 수행하세요.
            """
        
        # Few-Shot 예시 (단백질-리간드 상호작용 사례)
        few_shot_example = """
        **Few-Shot 예시 - 전문가 분석 과정 참조:**
        
        [예시] EGFR 키나제 억제제 Activity Cliff 분석:
        화합물 A: 게피티니브 (pKi: 7.8) vs 화합물 B: 엘로티니브 (pKi: 8.5)
        
        1. 단백질-리간드 결합: 퀴나졸린 코어의 6,7위치 치환기 차이가 ATP 결합 포켓과의 상호작용 패턴 변화
        2. 상호작용 패턴: 엘로티니브의 아세틸렌 링커가 Cys797과 새로운 소수성 접촉 형성
        3. 결합 기하학: 추가 아로마틱 고리가 DFG 루프와의 π-π 스택킹 개선
        4. 약리학적 메커니즘: 향상된 결합 기하학으로 0.7 pKi 단위 친화도 증가
        5. ADMET 영향: CYP3A4 대사 안정성 개선, 반감기 연장
        
        [귀하의 분석 과제]
        """
        
        return f"""
        당신은 단백질-리간드 상호작용 메커니즘 분야의 세계적 권위자입니다. 타겟 단백질과의 결합 방식 변화, 약리학적 관점과 생리활성 메커니즘 규명을 전문으로 하는 선임 연구자로서, 실제 신약 개발에서 사용되는 체계적 분석을 수행해주세요.
        
        {few_shot_example}
        
        **Activity Cliff 분석 대상:**
        
        **타겟 단백질:** {target_name}
        
        **화합물 정보:**
        - 고활성 화합물: {high_active['id']} (pKi: {high_active['pki']:.2f})
          SMILES: {high_active['smiles']}
        - 저활성 화합물: {low_active['id']} (pKi: {low_active['pki']:.2f})
          SMILES: {low_active['smiles']}
        
        **In-Context 생화학적 특성 (할루시네이션 방지용):**
        - 활성도 차이: {metrics['activity_difference']:.2f} pKi 단위
        - 구조 유사도: {metrics['similarity']:.3f} (Tanimoto)
        - 분자량 차이: {prop_diffs['mw_diff']:.2f} Da
        - LogP 차이: {prop_diffs['logp_diff']:.2f}
        - TPSA 차이: {prop_diffs.get('tpsa_diff', 0):.2f} Ų
        - 수소결합 공여자/수용자 변화 예상
        
        {literature_info}
        
        **단계별 Chain-of-Thought 분석 수행:**
        실제 구조생물학자/약리학자가 사용하는 분석 절차를 따라 다음 5단계로 체계적으로 분석하세요:
        
        1. **단백질-리간드 결합**: {target_name} 활성 부위와의 결합 방식 차이를 구체적으로 추론하세요. 알려진 단백질 구조 정보와 결합 포켓 특성을 고려하여 분석하세요.
        
        2. **상호작용 패턴**: 수소결합, 소수성 상호작용, π-π 스택킹, 반데르발스 힘 등의 변화가 어떻게 활성에 영향을 미치는지 설명하세요.
        
        3. **결합 기하학**: 분자 형태 변화가 단백질 포켓과의 입체적 적합성(shape complementarity)에 미치는 영향을 3차원 관점에서 분석하세요.
        
        4. **약리학적 메커니즘**: 결합 친화도 변화가 어떻게 기능적 활성 변화로 이어지는지, 알로스테릭 효과나 결합 동역학적 요인을 포함하여 설명하세요.
        
        5. **ADMET 영향**: 구조 변화가 대사 안정성, 선택성, 투과성 등에 미치는 영향과 전체적인 약물성에 대한 함의를 분석하세요.
        
        **필수 요구사항 - 구조생물학 전문가 수준:**
        1. 특정 아미노산 잔기 번호 명시 (Asp123, Phe456 등)
        2. 결합 친화도 값 계산 (Kd, Ki 값 또는 비율)
        3. 상호작용 에너지 정량화 (-5.2 kcal/mol 등)
        4. 도킹 스코어 비교와 RMSD 값
        5. 선택성 비율 예측 (vs off-target)
        
        **금지 사항 - 일반적 메커니즘 설명 금지:**
        - "수소결합이 중요하다" → "구체적 수소결합 길이와 위치"
        - "소수성 상호작용" → "특정 소수성 잔기와의 접촉 면적"
        - "활성이 감소한다" → "IC50 값 15배 증가" 등
        
        **실제 구조생물학 연구에서 사용할 수 있는 구체적 데이터와 메커니즘을 제시하세요.**
        
        **결과 형식 (반드시 이 형식을 정확히 따르세요):**
        
        신뢰도: [구체적 수치와 근거, 예: 78% - 도킹 스코어 차이와 결합 친화도 예측 기반]
        
        핵심 가설: [구체적이고 전문적인 메커니즘, 예: "Phe256과의 π-π 스택킹 상실로 인한 결합 친화도 15배 감소가 주요 원인"]
        
        상세 분석:
        1. 단백질-리간드 결합: [특정 결합 포켓, 잔기 번호, 상호작용 유형 명시]
        2. 상호작용 패턴: [수소결합 길이, 소수성 접촉 면적의 구체적 변화]
        3. 결합 기하학: [RMSD, 결합각, 비틀림각의 정량적 분석]
        4. 약리학적 메커니즘: [Ki/Kd 값 예측, 선택성 비율 계산]
        5. ADMET 영향: [CYP 대사, 혈장 단백질 결합률의 구체적 예측]
        
        분자 설계 제안: [특정 치환기 도입 전략과 예상 친화도 개선]
        
        **중요: 결합 친화도, 상호작용 에너지, 특정 아미노산 잔기 번호를 포함한 정량적 분석을 제시하고, 실제 구조생물학 데이터에 기반한 구체적 메커니즘을 설명하세요.**
        """
    
    def _extract_confidence_from_text(self, hypothesis: str) -> float:
        """가설 텍스트에서 실제 신뢰도 값을 추출"""
        import re
        
        # "신뢰도: XX%" 패턴 찾기
        confidence_match = re.search(r'신뢰도:.*?(\d+)%', hypothesis)
        if confidence_match:
            confidence_value = int(confidence_match.group(1))
            return confidence_value / 100.0
        
        # 영어 패턴도 확인
        confidence_match = re.search(r'confidence:.*?(\d+)%', hypothesis, re.IGNORECASE)
        if confidence_match:
            confidence_value = int(confidence_match.group(1))
            return confidence_value / 100.0
        
        # 추출 실패 시 키워드 기반 계산으로 fallback
        return self._calculate_confidence_by_keywords(hypothesis)
    
    def _calculate_confidence_by_keywords(self, hypothesis: str) -> float:
        """가설의 신뢰도 계산"""
        confidence_indicators = [
            ('결합' in hypothesis or 'binding' in hypothesis.lower(), 0.2),
            ('단백질' in hypothesis or 'protein' in hypothesis.lower(), 0.15),
            ('활성부위' in hypothesis or 'active site' in hypothesis.lower(), 0.15),
            ('상호작용' in hypothesis or 'interaction' in hypothesis.lower(), 0.1),
            ('친화도' in hypothesis or 'affinity' in hypothesis.lower(), 0.1),
            ('선택성' in hypothesis or 'selectivity' in hypothesis.lower(), 0.1),
            ('대사' in hypothesis or 'metabolism' in hypothesis.lower(), 0.1),
            ('도킹' in hypothesis or 'docking' in hypothesis.lower(), 0.1)
        ]
        
        base_confidence = 0.5
        for indicator, weight in confidence_indicators:
            if indicator:
                base_confidence += weight
        
        return min(base_confidence, 1.0)
    
    def _extract_key_insights(self, hypothesis: str) -> List[str]:
        """핵심 인사이트 추출"""
        insights = []
        if '결합' in hypothesis or 'binding' in hypothesis.lower():
            insights.append("단백질-리간드 결합 차이")
        if '활성부위' in hypothesis or 'active site' in hypothesis.lower():
            insights.append("활성부위 상호작용 변화")
        if '선택성' in hypothesis or 'selectivity' in hypothesis.lower():
            insights.append("선택성 차이")
        if not insights:
            insights.append("생체분자 상호작용 변화")
        return insights
    
    def _extract_reasoning_steps(self, hypothesis: str) -> List[str]:
        """추론 단계 추출"""
        steps = []
        lines = hypothesis.split('\n')
        current_step = ""
        
        for line in lines:
            line = line.strip()
            if any(marker in line for marker in ['1.', '2.', '3.', '4.', '5.', '**', '###']):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step.strip())
        
        return steps[:5]


class SARIntegrationExpert:
    """SAR 통합 전문가 에이전트"""
    
    def __init__(self, llm_client: UnifiedLLMClient):
        self.llm_client = llm_client
        self.persona = """당신은 화학정보학과 신약 개발 파이프라인의 실무 전문가입니다.
        SAR 분석 최적화, 신약 개발 전략 제시, 최신 화학정보학 기법 통합이 전문 분야입니다."""
    
    def generate(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """SAR 통합 관점의 가설 생성"""
        prompt = self._build_sar_prompt(shared_context)
        hypothesis = self.llm_client.generate_response(self.persona, prompt, temperature=0.7)
        
        return {
            'agent_type': 'sar_integration',
            'agent_name': 'SAR 통합 전문가',
            'hypothesis': hypothesis,
            'confidence': self._extract_confidence_from_text(hypothesis),
            'key_insights': self._extract_key_insights(hypothesis),
            'reasoning_steps': self._extract_reasoning_steps(hypothesis),
            'timestamp': time.time()
        }
    
    def _build_sar_prompt(self, shared_context: Dict[str, Any]) -> str:
        """SAR 통합 전문가용 특화 프롬프트 생성 - CoT.md 지침 반영"""
        cliff_summary = shared_context['cliff_summary']
        target_name = shared_context['target_name']
        high_active = cliff_summary['high_activity_compound']
        low_active = cliff_summary['low_activity_compound']
        metrics = cliff_summary['cliff_metrics']
        prop_diffs = cliff_summary['property_differences']
        
        literature_info = ""
        if shared_context.get('literature_context'):
            lit = shared_context['literature_context']
            literature_info = f"""
            **참고 문헌 정보 (RAG 검색 결과 - 할루시네이션 방지용):**
            - 제목: {lit.get('title', 'N/A')}
            - 초록: {lit.get('abstract', 'N/A')[:500]}...
            - PubMed ID: {lit.get('pmid', 'N/A')}
            - 키워드: {target_name}, 구조-활성 관계, Activity Cliff
            - 이 문헌을 전문가 지식의 근거로 활용하여 논리적 추론을 수행하세요.
            """
        
        # Few-Shot 예시 (SAR 분석 사례)
        few_shot_example = """
        **Few-Shot 예시 - 전문가 분석 과정 참조:**
        
        [예시] ACE 억제제 계열 SAR 분석:
        시리즈: 캅토프릴 → 에날라프릴 (pKi: 6.5 → 8.2)
        
        1. SAR 패턴: 티올기 → 카르복실기 변경으로 1.7 pKi 단위 활성 증가
        2. 화학정보학 인사이트: 낮은 Tanimoto 유사도(0.4)에도 큰 활성 차이는 약물발견의 전환점
        3. 신약 개발 전략: 프로드러그 전략 도입으로 ADMET 특성 개선
        4. 최적화 방향: 아연 결합 모티프 최적화가 핵심, 주변 치환기는 선택성 조절
        5. 예측 모델링: 금속 배위 결합을 고려한 3D-QSAR 모델 필요
        
        [귀하의 분석 과제]
        """
        
        return f"""
        당신은 화학정보학과 신약 개발 파이프라인의 실무 전문가입니다. SAR 분석 최적화, 신약 개발 전략 제시, 최신 화학정보학 기법 통합이 전문 분야인 선임 연구자로서, 실제 제약회사에서 사용되는 체계적 SAR 분석을 수행해주세요.
        
        {few_shot_example}
        
        **Activity Cliff 분석 대상:**
        
        **타겟 단백질:** {target_name}
        
        **화합물 정보:**
        - 고활성: {high_active['id']} (pKi: {high_active['pki']:.2f})
          SMILES: {high_active['smiles']}
        - 저활성: {low_active['id']} (pKi: {low_active['pki']:.2f})
          SMILES: {low_active['smiles']}
        
        **In-Context SAR 메트릭 (할루시네이션 방지용):**
        - Cliff 점수: {metrics.get('cliff_score', 0):.3f}
        - 구조 유사도: {metrics['similarity']:.3f} (Tanimoto)
        - 활성 차이: {metrics['activity_difference']:.2f} pKi 단위
        - 같은 스캐폴드: {metrics.get('same_scaffold', 'Unknown')}
        - 구조적 차이: {metrics['structural_difference_type']}
        
        **물리화학적 특성 차이:**
        - 분자량: {prop_diffs['mw_diff']:.2f} Da
        - LogP: {prop_diffs['logp_diff']:.2f} (지용성 변화)
        - TPSA: {prop_diffs.get('tpsa_diff', 0):.2f} Ų (극성 표면적 변화)
        
        {literature_info}
        
        **단계별 Chain-of-Thought 분석 수행:**
        실제 화학정보학자/신약개발자가 사용하는 분석 절차를 따라 다음 5단계로 체계적으로 분석하세요:
        
        1. **SAR 패턴 분석**: 이 Activity Cliff가 보여주는 구조-활성 관계의 핵심 트렌드를 식별하세요. 어떤 구조적 변화가 활성에 가장 큰 영향을 미치는지 정량적으로 분석하세요.
        
        2. **화학정보학 인사이트**: Tanimoto 유사도 {metrics['similarity']:.3f}와 {metrics['activity_difference']:.2f} pKi 단위 활성 차이의 조합이 갖는 화학정보학적 의미를 해석하세요. 이것이 SAR 공간에서 의미하는 바를 설명하세요.
        
        3. **신약 개발 전략**: 이 결과가 후속 화합물 설계와 최적화 전략에 주는 구체적인 시사점을 제시하세요. Lead optimization 관점에서 우선순위를 제안하세요.
        
        4. **최적화 방향**: 활성 개선을 위한 구조 변경 전략을 분자 설계 관점에서 제안하세요. 어떤 부분을 고정하고 어떤 부분을 변경해야 하는지 구체적으로 설명하세요.
        
        5. **예측 모델링**: QSAR/ML 모델 구축 시 이 Activity Cliff 데이터가 주는 교훈과 모델 개선 방향을 제안하세요. 피처 선택과 알고리즘 선택에 대한 가이드라인을 제시하세요.
        
        **필수 요구사항 - 신약개발 전문가 수준:**
        1. 정량적 QSAR 모델 제시 (R² 값, 방정식 등)
        2. 후속 화합물 3-5개의 구체적 구조와 예상 활성값
        3. 합성 가능성과 비용 추정 (FTE, 비용 등)
        4. 치환기별 기여도 순위 (Hammett 상수 활용)
        5. 특허 회피 전략과 경쟁사 분석
        
        **금지 사항 - 추상적 전략 금지:**
        - "최적화가 필요하다" → "구체적 최적화 단계와 타겟 구조"
        - "비슷한 화합물" → "완전한 SMILES 구조와 예상 pKi 값"
        - "개선이 기대된다" → "정량적 개선 예측과 성공 확률"
        
        **실제 제약회사에서 사용할 수 있는 구체적 데이터와 전략을 제시하여 즉시 실행 가능한 액션 플랜을 작성하세요.**
        
        **결과 형식 (반드시 이 형식을 정확히 따르세요):**
        
        신뢰도: [구체적 수치와 근거, 예: 92% - QSAR 모델 예측값과 구조적 유사체 데이터 일치]
        
        핵심 가설: [구체적 SAR 관계, 예: "R2 위치 전자끌기 치환기 도입 시 0.5 log 단위당 1.2 pKi 증가의 선형 관계"]
        
        상세 분석:
        1. SAR 패턴 분석: [Hammett 상수, 입체 매개변수의 정량적 상관관계]
        2. 화학정보학 인사이트: [Tanimoto 계수와 활성 차이의 수학적 모델링]
        3. 신약 개발 전략: [리드 최적화 단계별 우선순위와 성공 확률]
        4. 최적화 방향: [특정 치환기의 정량적 기여도와 다음 합성 타겟]
        5. 예측 모델링: [Random Forest/SVM 모델의 예측 정확도와 신뢰구간]
        
        분자 설계 제안: [구체적 구조식과 예상 활성값을 포함한 차세대 화합물 3-5개]
        
        실험 제안: [합성 경로, 활성 측정 프로토콜, 예상 비용과 기간]
        
        **중요: 정량적 QSAR 관계식, 구체적 치환기 효과, 예측 활성값을 포함하여 실제 제약회사에서 사용할 수 있는 수준의 구체적 전략을 제시하세요.**
        """
    
    def _extract_confidence_from_text(self, hypothesis: str) -> float:
        """가설 텍스트에서 실제 신뢰도 값을 추출"""
        import re
        
        # "신뢰도: XX%" 패턴 찾기
        confidence_match = re.search(r'신뢰도:.*?(\d+)%', hypothesis)
        if confidence_match:
            confidence_value = int(confidence_match.group(1))
            return confidence_value / 100.0
        
        # 영어 패턴도 확인
        confidence_match = re.search(r'confidence:.*?(\d+)%', hypothesis, re.IGNORECASE)
        if confidence_match:
            confidence_value = int(confidence_match.group(1))
            return confidence_value / 100.0
        
        # 추출 실패 시 키워드 기반 계산으로 fallback
        return self._calculate_confidence_by_keywords(hypothesis)
    
    def _calculate_confidence_by_keywords(self, hypothesis: str) -> float:
        """가설의 신뢰도 계산"""
        confidence_indicators = [
            ('SAR' in hypothesis or 'sar' in hypothesis.lower(), 0.2),
            ('최적화' in hypothesis or 'optimization' in hypothesis.lower(), 0.15),
            ('설계' in hypothesis or 'design' in hypothesis.lower(), 0.15),
            ('예측' in hypothesis or 'prediction' in hypothesis.lower(), 0.1),
            ('모델' in hypothesis or 'model' in hypothesis.lower(), 0.1),
            ('전략' in hypothesis or 'strategy' in hypothesis.lower(), 0.1),
            ('패턴' in hypothesis or 'pattern' in hypothesis.lower(), 0.1),
            ('트렌드' in hypothesis or 'trend' in hypothesis.lower(), 0.1)
        ]
        
        base_confidence = 0.5
        for indicator, weight in confidence_indicators:
            if indicator:
                base_confidence += weight
        
        return min(base_confidence, 1.0)
    
    def _extract_key_insights(self, hypothesis: str) -> List[str]:
        """핵심 인사이트 추출"""
        insights = []
        if 'SAR' in hypothesis or 'sar' in hypothesis.lower():
            insights.append("SAR 패턴 식별")
        if '최적화' in hypothesis or 'optimization' in hypothesis.lower():
            insights.append("구조 최적화 전략")
        if '예측' in hypothesis or 'prediction' in hypothesis.lower():
            insights.append("예측 모델 개선점")
        if not insights:
            insights.append("SAR 트렌드 분석")
        return insights
    
    def _extract_reasoning_steps(self, hypothesis: str) -> List[str]:
        """추론 단계 추출"""
        steps = []
        lines = hypothesis.split('\n')
        current_step = ""
        
        for line in lines:
            line = line.strip()
            if any(marker in line for marker in ['1.', '2.', '3.', '4.', '5.', '**', '###']):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            else:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step.strip())
        
        return steps[:5]


class ReflectionAgent:
    """가설 타당성 평가 에이전트"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        
    def evaluate_hypotheses(self, domain_hypotheses: List[Dict], shared_context: Dict) -> List[Dict]:
        """각 가설의 타당성을 종합적으로 평가"""
        
        st.info("🤔 **Phase 3: Reflection** - 가설 타당성 평가 및 피드백 생성중...")
        
        evaluation_results = []
        
        for i, hypothesis in enumerate(domain_hypotheses):
            with st.spinner(f"{hypothesis['agent_name']} 가설 평가 중..."):
                evaluation_prompt = self._build_evaluation_prompt(hypothesis, shared_context)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "당신은 과학적 가설 평가 전문가입니다. 객관적이고 건설적인 평가를 제공합니다."},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0.3
                )
                
                evaluation_text = response.choices[0].message.content
                
                # 평가 점수 파싱
                scores = self._parse_evaluation_scores(evaluation_text)
                
                result = {
                    'hypothesis_id': i,
                    'agent_name': hypothesis['agent_name'],
                    'original_hypothesis': hypothesis,
                    'evaluation_text': evaluation_text,
                    'scores': scores,
                    'feedback': self._extract_feedback(evaluation_text),
                    'strengths': self._extract_strengths(evaluation_text),
                    'weaknesses': self._extract_weaknesses(evaluation_text),
                    'timestamp': time.time()
                }
                
                evaluation_results.append(result)
                
                # 평가 결과 즉시 표시
                self._display_reflection_result(result)
        
        return evaluation_results
    
    def _build_evaluation_prompt(self, hypothesis: Dict, shared_context: Dict) -> str:
        """평가용 프롬프트 구성"""
        return f"""
        **가설 평가 요청:**
        
        **전문가:** {hypothesis['agent_name']}
        **가설 내용:**
        {hypothesis['hypothesis']}
        
        **원본 신뢰도:** {hypothesis['confidence']:.0%}
        
        **평가 요청:**
        다음 기준으로 이 가설을 객관적으로 평가해주세요:
        
        1. **과학적 엄밀성** (Scientific Rigor): 논리적 일관성, 과학적 근거
        2. **증거 통합** (Evidence Integration): 데이터와 문헌 활용도
        3. **실용성** (Practical Applicability): 실제 적용 가능성
        4. **혁신성** (Innovation): 새로운 인사이트 제공
        
        **평가 형식:**
        점수: [각 기준별 0-100점]
        강점: [2-3개 항목]
        약점: [1-2개 항목] 
        개선 제안: [구체적 피드백]
        총평: [종합 평가]
        
        객관적이고 건설적인 평가를 부탁드립니다.
        """
    
    def _parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, float]:
        """평가 텍스트에서 점수 추출 - 개선된 파싱 로직"""
        # 기본값을 합리적인 범위로 설정
        scores = {
            'scientific_rigor': 75.0,
            'evidence_integration': 75.0,
            'practical_applicability': 75.0,
            'innovation': 75.0
        }
        
        # 더 정확한 점수 추출을 위한 개선된 로직
        lines = evaluation_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            # 점수 패턴 찾기: "엄밀성: 85점" 또는 "Scientific Rigor: 85"
            if any(keyword in line_lower for keyword in ['엄밀성', 'rigor', '과학적']):
                score = self._extract_score_from_line(line)
                if score is not None and 0 <= score <= 100:
                    scores['scientific_rigor'] = score
            elif any(keyword in line_lower for keyword in ['증거', 'evidence', '통합']):
                score = self._extract_score_from_line(line)
                if score is not None and 0 <= score <= 100:
                    scores['evidence_integration'] = score
            elif any(keyword in line_lower for keyword in ['실용', 'practical', '적용']):
                score = self._extract_score_from_line(line)
                if score is not None and 0 <= score <= 100:
                    scores['practical_applicability'] = score
            elif any(keyword in line_lower for keyword in ['혁신', 'innovation', '창의']):
                score = self._extract_score_from_line(line)
                if score is not None and 0 <= score <= 100:
                    scores['innovation'] = score
        
        # 모든 점수가 유효한 범위 내에 있는지 확인
        for key, value in scores.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 100:
                scores[key] = 75.0  # 안전한 기본값
        
        return scores
    
    def _extract_score_from_line(self, line: str) -> Optional[float]:
        """라인에서 점수 추출"""
        import re
        # 0-100 범위의 숫자 찾기
        matches = re.findall(r'\b(\d{1,3})\b', line)
        for match in matches:
            score = float(match)
            if 0 <= score <= 100:
                return score
        return None
    
    def _extract_feedback(self, evaluation_text: str) -> List[str]:
        """피드백 추출"""
        feedback = []
        lines = evaluation_text.split('\n')
        in_feedback_section = False
        
        for line in lines:
            line = line.strip()
            if '개선' in line or 'feedback' in line.lower() or '제안' in line:
                in_feedback_section = True
                continue
            elif in_feedback_section and line and not line.startswith('**'):
                feedback.append(line)
            elif in_feedback_section and line.startswith('**'):
                break
        
        return feedback[:3]  # 최대 3개
    
    def _extract_strengths(self, evaluation_text: str) -> List[str]:
        """강점 추출"""
        strengths = []
        lines = evaluation_text.split('\n')
        in_strengths_section = False
        
        for line in lines:
            line = line.strip()
            if '강점' in line or 'strength' in line.lower():
                in_strengths_section = True
                continue
            elif in_strengths_section and line and not line.startswith('**'):
                strengths.append(line)
            elif in_strengths_section and line.startswith('**'):
                break
        
        return strengths[:3]  # 최대 3개
    
    def _extract_weaknesses(self, evaluation_text: str) -> List[str]:
        """약점 추출"""
        weaknesses = []
        lines = evaluation_text.split('\n')
        in_weaknesses_section = False
        
        for line in lines:
            line = line.strip()
            if '약점' in line or 'weakness' in line.lower():
                in_weaknesses_section = True
                continue
            elif in_weaknesses_section and line and not line.startswith('**'):
                weaknesses.append(line)
            elif in_weaknesses_section and line.startswith('**'):
                break
        
        return weaknesses[:2]  # 최대 2개
    
    def _display_reflection_result(self, result: Dict):
        """평가 결과 표시"""
        with st.expander(f"📝 {result['agent_name']} 평가 결과", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**평가 요약:**")
                st.write(result['evaluation_text'][:200] + "..." if len(result['evaluation_text']) > 200 else result['evaluation_text'])
                
                if result['strengths']:
                    st.write("**주요 강점:**")
                    for strength in result['strengths']:
                        st.write(f"• {strength}")
                        
                if result['weaknesses']:
                    st.write("**개선점:**")
                    for weakness in result['weaknesses']:
                        st.write(f"• {weakness}")
            
            with col2:
                st.write("**평가 점수:**")
                avg_score = sum(result['scores'].values()) / len(result['scores'])
                st.metric("종합 점수", f"{avg_score:.1f}/100")
                
                for criterion, score in result['scores'].items():
                    criterion_kr = {
                        'scientific_rigor': '과학적 엄밀성',
                        'evidence_integration': '증거 통합',
                        'practical_applicability': '실용성',
                        'innovation': '혁신성'
                    }.get(criterion, criterion)
                    st.metric(criterion_kr, f"{score:.1f}")


class EloRankingAgent:
    """Elo 시스템 기반 순위 매김 에이전트"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        self.initial_elo = 1500  # 초기 Elo 점수
        self.k_factor = 32  # Elo 업데이트 계수
        
    async def perform_elo_comparisons(self, reflection_results: List[Dict], criteria_weights: Dict = None) -> Tuple[List[Dict], float]:
        """Elo 시스템으로 가설 간 상대적 우수성 평가"""
        
        if criteria_weights is None:
            criteria_weights = {
                'logical_consistency': 0.4,
                'research_relevance': 0.3,
                'innovation': 0.3
            }
        
        st.info("🏆 **Phase 4: Ranking** - Elo 시스템으로 가설 순위 매김중...")
        
        # 초기 Elo 점수 설정
        elo_scores = {i: self.initial_elo for i in range(len(reflection_results))}
        
        comparison_results = []
        total_comparisons = len(reflection_results) * (len(reflection_results) - 1) // 2
        current_comparison = 0
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 모든 가설 쌍에 대해 비교 수행
        for i in range(len(reflection_results)):
            for j in range(i + 1, len(reflection_results)):
                current_comparison += 1
                progress = current_comparison / total_comparisons
                progress_bar.progress(progress)
                status_text.text(f"Elo 비교 진행중... ({current_comparison}/{total_comparisons})")
                
                hypothesis_a = reflection_results[i]
                hypothesis_b = reflection_results[j]
                
                # 쌍별 비교 수행
                comparison_result = await self._compare_hypotheses_pair(
                    hypothesis_a, hypothesis_b, criteria_weights
                )
                
                # Elo 점수 업데이트
                old_elo_a, old_elo_b = elo_scores[i], elo_scores[j]
                new_elo_a, new_elo_b = self._update_elo_scores(
                    old_elo_a, old_elo_b, comparison_result
                )
                
                elo_scores[i] = new_elo_a
                elo_scores[j] = new_elo_b
                
                # 비교 과정 시각화
                self._display_elo_comparison(hypothesis_a, hypothesis_b, comparison_result, 
                                           new_elo_a, new_elo_b, old_elo_a, old_elo_b)
                
                comparison_results.append({
                    'pair': (i, j),
                    'winner': comparison_result['winner'],
                    'reasoning': comparison_result['reasoning'],
                    'confidence': comparison_result['confidence'],
                    'elo_change': (new_elo_a - old_elo_a, new_elo_b - old_elo_b)
                })
        
        # 최종 순위 매김
        ranked_hypotheses = self._rank_by_elo_scores(reflection_results, elo_scores)
        consensus_score = self._calculate_consensus_score(elo_scores)
        
        # 최종 Elo 순위 표시
        self._display_final_elo_ranking(ranked_hypotheses, elo_scores, consensus_score)
        
        return ranked_hypotheses, consensus_score
    
    async def _compare_hypotheses_pair(self, hyp_a: Dict, hyp_b: Dict, criteria_weights: Dict) -> Dict:
        """두 가설을 직접 비교하여 우수한 가설 선정"""
        
        comparison_prompt = f"""
        **가설 비교 요청:**
        
        다음 두 가설을 객관적으로 비교하고 어느 것이 더 우수한지 판단하세요.
        
        **평가 기준 가중치:**
        - 논리적 일관성: {criteria_weights['logical_consistency']:.1f}
        - 기존 연구 연관성: {criteria_weights['research_relevance']:.1f}
        - 혁신성: {criteria_weights['innovation']:.1f}
        
        **가설 A ({hyp_a['agent_name']}):**
        {hyp_a['original_hypothesis']['hypothesis']}
        
        평가 점수: {hyp_a['scores']}
        
        **가설 B ({hyp_b['agent_name']}):**
        {hyp_b['original_hypothesis']['hypothesis']}
        
        평가 점수: {hyp_b['scores']}
        
        **비교 결과를 JSON 형식으로 제공하세요:**
        {{
            "winner": "A" 또는 "B",
            "confidence": 0.5-1.0,
            "reasoning": "구체적인 비교 이유 (100자 이내)",
            "criteria_analysis": {{
                "logical_consistency": "A 또는 B가 우수한 이유",
                "research_relevance": "A 또는 B가 우수한 이유", 
                "innovation": "A 또는 B가 우수한 이유"
            }}
        }}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 과학적 가설 비교 전문가입니다. 객관적이고 일관된 평가를 제공합니다."},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0.2
        )
        
        try:
            response_text = response.choices[0].message.content.strip()
            
            # JSON 블록을 찾아서 추출
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                result = json.loads(json_text)
                
                # 필수 필드 검증 및 보완
                if "winner" not in result:
                    result["winner"] = "A"
                if "confidence" not in result:
                    result["confidence"] = 0.6
                if "reasoning" not in result:
                    result["reasoning"] = "비교 분석 완료"
                if "criteria_analysis" not in result:
                    result["criteria_analysis"] = {}
            else:
                raise ValueError("JSON 형식을 찾을 수 없음")
                
        except Exception as e:
            # JSON 파싱 실패 시 응답 텍스트 기반으로 휴리스틱 분석
            response_text = response.choices[0].message.content.lower()
            
            # A 또는 B 승자 결정
            winner = "B"  # 기본값
            if "가설 a" in response_text and "우수" in response_text:
                winner = "A"
            elif "가설 b" in response_text and "우수" in response_text:
                winner = "B"
            
            # 신뢰도 추정
            confidence = 0.7
            if "확실" in response_text or "명확" in response_text:
                confidence = 0.8
            elif "애매" in response_text or "유사" in response_text:
                confidence = 0.6
            
            result = {
                "winner": winner,
                "confidence": confidence,
                "reasoning": f"응답 기반 분석: {response.choices[0].message.content[:100]}..." if hasattr(response.choices[0].message, 'content') else "분석 완료",
                "criteria_analysis": {
                    "logical_consistency": f"가설 {winner}가 논리적으로 더 일관성 있음",
                    "research_relevance": f"가설 {winner}가 연구와 더 관련성 높음",
                    "innovation": f"가설 {winner}가 더 혁신적 관점 제시"
                }
            }
        
        return result
    
    def _update_elo_scores(self, elo_a: float, elo_b: float, comparison_result: Dict) -> Tuple[float, float]:
        """Elo 점수 업데이트"""
        # 예상 승률 계산
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        expected_b = 1 - expected_a
        
        # 실제 결과 (winner에 따라)
        if comparison_result['winner'] == 'A':
            actual_a, actual_b = 1, 0
        else:
            actual_a, actual_b = 0, 1
        
        # 신뢰도를 반영한 K-factor 조정
        confidence = comparison_result.get('confidence', 0.6)
        adjusted_k = self.k_factor * confidence
        
        # 새로운 Elo 점수 계산
        new_elo_a = elo_a + adjusted_k * (actual_a - expected_a)
        new_elo_b = elo_b + adjusted_k * (actual_b - expected_b)
        
        return new_elo_a, new_elo_b
    
    def _rank_by_elo_scores(self, reflection_results: List[Dict], elo_scores: Dict) -> List[Dict]:
        """Elo 점수를 기준으로 가설 순위 매김"""
        # (index, elo_score) 튜플로 변환 후 정렬
        sorted_indices = sorted(elo_scores.items(), key=lambda x: x[1], reverse=True)
        
        ranked_hypotheses = []
        for rank, (index, elo_score) in enumerate(sorted_indices):
            hypothesis = reflection_results[index].copy()
            hypothesis['rank'] = rank + 1
            hypothesis['elo_score'] = elo_score
            hypothesis['elo_rating'] = self._get_elo_rating(elo_score)
            ranked_hypotheses.append(hypothesis)
        
        return ranked_hypotheses
    
    def _get_elo_rating(self, elo_score: float) -> str:
        """Elo 점수를 등급으로 변환"""
        if elo_score >= 1700:
            return "S급 (탁월)"
        elif elo_score >= 1600:
            return "A급 (우수)"
        elif elo_score >= 1500:
            return "B급 (평균)"
        elif elo_score >= 1400:
            return "C급 (보통)"
        else:
            return "D급 (미흡)"
    
    def _calculate_consensus_score(self, elo_scores: Dict) -> float:
        """Elo 점수 분산을 바탕으로 합의도 계산"""
        scores = list(elo_scores.values())
        if len(scores) <= 1:
            return 1.0
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # 표준편차를 0-1 범위의 합의도로 변환 (낮은 표준편차 = 높은 합의도)
        # 표준편차 100 이상은 합의도 0, 0은 합의도 1로 설정
        consensus_score = max(0, 1 - (std_dev / 100))
        
        return consensus_score
    
    def _display_elo_comparison(self, hyp_a: Dict, hyp_b: Dict, comparison_result: Dict, 
                               elo_a: float, elo_b: float, old_elo_a: float, old_elo_b: float):
        """Elo 비교 과정 표시"""
        with st.expander(f"⚔️ Elo 비교: {hyp_a['agent_name']} vs {hyp_b['agent_name']}", expanded=False):
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.write("**가설 A**")
                st.write(hyp_a['agent_name'])
                elo_change_a = elo_a - old_elo_a
                st.metric("Elo 점수", f"{elo_a:.0f}", f"{elo_change_a:+.0f}")
                
            with col2:
                st.write("**비교 결과**")
                winner_name = hyp_a['agent_name'] if comparison_result['winner'] == 'A' else hyp_b['agent_name']
                st.success(f"🏆 {winner_name}")
                st.metric("신뢰도", f"{comparison_result.get('confidence', 0.6):.0%}")
                
            with col3:
                st.write("**가설 B**")
                st.write(hyp_b['agent_name'])
                elo_change_b = elo_b - old_elo_b
                st.metric("Elo 점수", f"{elo_b:.0f}", f"{elo_change_b:+.0f}")
            
            st.write("**비교 근거:**")
            st.write(comparison_result['reasoning'])
    
    def _display_final_elo_ranking(self, ranked_hypotheses: List[Dict], elo_scores: Dict, consensus_score: float):
        """최종 Elo 순위 표시"""
        with st.container():
            st.markdown("### 🏆 최종 Elo 순위")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                for i, hypothesis in enumerate(ranked_hypotheses):
                    rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
                    st.write(f"{rank_emoji} **{hypothesis['agent_name']}** - Elo: {hypothesis['elo_score']:.0f} ({hypothesis['elo_rating']})")
            
            with col2:
                st.metric("에이전트 간 합의도", f"{consensus_score:.2f}", 
                         "높음" if consensus_score >= 0.8 else "보통" if consensus_score >= 0.6 else "낮음")
                
                avg_elo = sum(elo_scores.values()) / len(elo_scores)
                st.metric("평균 Elo 점수", f"{avg_elo:.0f}")


class EvolutionAgent:
    """Self-Play 가설 개선 에이전트"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        
    async def self_play_improvement(self, ranked_hypotheses: List[Dict], consensus_score: float, shared_context: Dict) -> List[Dict]:
        """Self-Play 논쟁을 통한 가설 개선"""
        
        if consensus_score >= 0.8:
            st.info("✅ **Phase 5A: Evolution** - 에이전트 간 합의도가 높아 Evolution 단계를 생략합니다.")
            st.metric("합의도 점수", f"{consensus_score:.2f}", "높음 (≥0.8)")
            return ranked_hypotheses
            
        st.info("⚔️ **Phase 5A: Evolution** - Self-Play 논쟁을 통한 가설 개선 진행중...")
        st.metric("합의도 점수", f"{consensus_score:.2f}", "낮음 (<0.8)")
        
        improved_hypotheses = []
        
        # 상위 2개 가설에 대해서만 Self-Play 진행 (시간 절약)
        for i, hypothesis in enumerate(ranked_hypotheses[:2]):
            st.markdown(f"### 🥊 가설 {i+1} Self-Play 논쟁")
            
            # 1단계: 대안 가설 생성
            st.write("**1단계: 대안 가설 생성**")
            with st.spinner("대안 가설 생성 중..."):
                alternative = await self._generate_alternative_hypothesis(hypothesis, shared_context)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**원본 가설**")
                st.write(hypothesis['original_hypothesis']['hypothesis'][:200] + "...")
            with col2:
                st.write("**대안 가설**")
                st.write(alternative['hypothesis'][:200] + "...")
            
            # 2단계: 3라운드 논쟁 시뮬레이션
            st.write("**2단계: 논쟁 시뮬레이션 (3라운드)**")
            debate_results = []
            
            for round_num in range(1, 4):
                st.write(f"**라운드 {round_num}**")
                
                with st.spinner(f"라운드 {round_num} 논쟁 진행 중..."):
                    debate_round = await self._simulate_debate_round(
                        hypothesis, alternative, shared_context, round_num
                    )
                
                debate_results.append(debate_round)
                
                # 라운드별 결과 표시
                self._display_debate_round_result(debate_round, round_num)
            
            # 3단계: 개선된 가설 합성
            st.write("**3단계: 개선된 가설 합성**")
            with st.spinner("개선된 가설 합성 중..."):
                improved = await self._synthesize_improved_hypothesis(
                    hypothesis, alternative, debate_results, shared_context
                )
            
            # 개선 결과 표시
            self._display_improvement_result(hypothesis, improved)
            
            improved_hypotheses.append(improved)
        
        # 개선되지 않은 나머지 가설들도 포함
        for hypothesis in ranked_hypotheses[2:]:
            improved_hypotheses.append(hypothesis)
        
        return improved_hypotheses
    
    async def _generate_alternative_hypothesis(self, original_hypothesis: Dict, shared_context: Dict) -> Dict:
        """원본 가설의 대안 생성"""
        
        prompt = f"""
        **대안 가설 생성 요청:**
        
        다음 원본 가설에 대한 건설적인 대안을 제시해주세요:
        
        **원본 가설 ({original_hypothesis['agent_name']}):**
        {original_hypothesis['original_hypothesis']['hypothesis']}
        
        **Activity Cliff 맥락:**
        {shared_context['cliff_summary']}
        
        **대안 생성 지침:**
        1. 원본 가설의 핵심 아이디어는 유지하되, 다른 관점이나 메커니즘 제시
        2. 동일한 데이터를 다르게 해석할 수 있는 과학적 근거 제공
        3. 원본보다 더 구체적이거나 포괄적인 설명 시도
        4. 실험적 검증 방법도 함께 제안
        
        **결과 형식:**
        - 대안의 핵심 차이점: [원본과의 주요 차이]
        - 대안 가설: [상세한 대안 설명]
        - 우수성 주장: [왜 이 대안이 고려될 만한가]
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 창의적이고 비판적 사고력을 가진 과학자입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        return {
            'hypothesis': response.choices[0].message.content,
            'type': 'alternative',
            'source': 'evolution_agent',
            'timestamp': time.time()
        }
    
    async def _simulate_debate_round(self, original: Dict, alternative: Dict, shared_context: Dict, round_num: int) -> Dict:
        """한 라운드의 논쟁 시뮬레이션"""
        
        debate_prompt = f"""
        **라운드 {round_num} 과학적 논쟁 시뮬레이션:**
        
        **원본 가설 입장:**
        {original['original_hypothesis']['hypothesis']}
        
        **대안 가설 입장:**
        {alternative['hypothesis']}
        
        **논쟁 맥락:**
        {shared_context['cliff_summary']}
        
        **논쟁 규칙:**
        1. 각 가설은 상대방의 약점을 지적하고 자신의 강점을 주장
        2. 과학적 근거와 데이터를 바탕으로 논증
        3. 건설적이고 객관적인 토론 유지
        4. 라운드 {round_num}에 맞는 논쟁 깊이 조절
        
        **결과를 JSON 형식으로 제공:**
        {{
            "original_argument": "원본 가설의 주장 (100자 이내)",
            "alternative_argument": "대안 가설의 주장 (100자 이내)",
            "round_winner": "original" 또는 "alternative",
            "key_points": ["핵심 논점 1", "핵심 논점 2"],
            "evidence_cited": ["인용된 증거 1", "인용된 증거 2"],
            "next_round_focus": "다음 라운드에서 집중할 주제"
        }}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 과학적 논쟁의 공정한 중재자입니다."},
                {"role": "user", "content": debate_prompt}
            ],
            temperature=0.6
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
        except:
            # JSON 파싱 실패 시 기본값
            result = {
                "original_argument": "원본 가설 주장",
                "alternative_argument": "대안 가설 주장",
                "round_winner": "original",
                "key_points": ["논쟁 진행"],
                "evidence_cited": ["데이터 기반 논증"],
                "next_round_focus": "심화 논의"
            }
        
        return result
    
    async def _synthesize_improved_hypothesis(self, original: Dict, alternative: Dict, debate_results: List[Dict], shared_context: Dict) -> Dict:
        """논쟁 결과를 바탕으로 개선된 가설 합성"""
        
        # 논쟁 결과 요약
        debate_summary = "\n".join([
            f"라운드 {i+1}: {result['key_points']}" for i, result in enumerate(debate_results)
        ])
        
        synthesis_prompt = f"""
        **개선된 가설 합성 요청:**
        
        **원본 가설:**
        {original['original_hypothesis']['hypothesis']}
        
        **대안 가설:**
        {alternative['hypothesis']}
        
        **3라운드 논쟁 결과:**
        {debate_summary}
        
        **합성 지침:**
        1. 논쟁에서 나온 최고의 아이디어들을 통합
        2. 각 가설의 강점을 결합하고 약점을 보완
        3. 논쟁 과정에서 발견된 새로운 인사이트 반영
        4. 더 강력하고 포괄적인 가설로 발전
        
        **결과 형식:**
        - 개선 요약: [어떤 점이 개선되었는가]
        - 개선된 가설: [최종 통합 가설]
        - 신뢰도 향상: [왜 더 신뢰할 만한가]
        - 검증 방법: [제안된 실험적 검증]
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 과학적 통합과 종합의 전문가입니다."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.5
        )
        
        improved_hypothesis = original.copy()
        improved_content = response.choices[0].message.content
        
        # final_hypothesis 필드 업데이트 (display_final_results 호환성)
        improved_hypothesis['final_hypothesis'] = improved_content
        improved_hypothesis['improved_hypothesis'] = improved_content
        improved_hypothesis['evolution_applied'] = True
        improved_hypothesis['debate_results'] = debate_results
        improved_hypothesis['alternative_hypothesis'] = alternative
        improved_hypothesis['improvement_timestamp'] = time.time()
        
        # 점수도 약간 향상시킴
        if 'final_score' in improved_hypothesis:
            improved_hypothesis['final_score'] = min(improved_hypothesis['final_score'] + 5, 100)
        
        return improved_hypothesis
    
    def _display_debate_round_result(self, round_result: Dict, round_num: int):
        """논쟁 라운드 결과 표시"""
        with st.expander(f"라운드 {round_num} 상세 결과", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**원본 가설 주장:**")
                st.write(round_result['original_argument'])
                
            with col2:
                st.write("**대안 가설 주장:**")
                st.write(round_result['alternative_argument'])
            
            winner_text = "원본 가설" if round_result['round_winner'] == 'original' else "대안 가설"
            st.info(f"🏆 라운드 {round_num} 승자: {winner_text}")
            
            if round_result.get('key_points'):
                st.write("**핵심 논점:**")
                for point in round_result['key_points']:
                    st.write(f"• {point}")
    
    def _display_improvement_result(self, original: Dict, improved: Dict):
        """개선 결과 표시"""
        with st.container():
            st.markdown("####Self-Play 개선 결과")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**원본 가설:**")
                st.write(original['original_hypothesis']['hypothesis'][:150] + "...")
                
            with col2:
                st.write("**개선된 가설:**")
                improved_text = improved.get('improved_hypothesis', '개선 결과 없음')
                st.write(improved_text[:150] + "...")
            
            if improved.get('evolution_applied'):
                st.success("✨ Self-Play 논쟁을 통해 성공적으로 개선되었습니다!")
            else:
                st.info("논쟁 결과 원본 가설이 충분히 우수하다고 판단되었습니다.")


class MetaReviewAgent:
    """최종 품질 검토 및 종합 리포트 생성 에이전트"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        
    async def compile_final_report(self, final_hypotheses: List[Dict], shared_context: Dict) -> Dict:
        """최종 품질 검토 및 종합 리포트 생성"""
        
        st.info("**Phase 5B: Meta-Review** - 최종 품질 검토 및 리포트 통합중...")
        
        # 각 가설에 대한 최종 품질 평가
        quality_assessments = []
        
        for i, hypothesis in enumerate(final_hypotheses):
            with st.spinner(f"가설 {i+1} 품질 평가 중..."):
                assessment = await self._assess_hypothesis_quality(hypothesis, shared_context)
                quality_assessments.append(assessment)
                
                # 품질 평가 결과 표시
                self._display_quality_assessment(hypothesis, assessment, i+1)
        
        # 종합 리포트 생성
        with st.spinner("종합 리포트 생성 중..."):
            comprehensive_report = await self._generate_comprehensive_report(
                final_hypotheses, quality_assessments, shared_context
            )
        
        return comprehensive_report
    
    async def _assess_hypothesis_quality(self, hypothesis: Dict, shared_context: Dict) -> Dict:
        """개별 가설의 품질 평가"""
        
        # 개선된 가설이 있으면 그것을 평가, 없으면 원본 평가
        hypothesis_text = hypothesis.get('improved_hypothesis', 
                                       hypothesis.get('original_hypothesis', {}).get('hypothesis', ''))
        
        quality_prompt = f"""
        **가설 품질 종합 평가:**
        
        **평가 대상 가설:**
        {hypothesis_text}
        
        **맥락 정보:**
        - 에이전트: {hypothesis.get('agent_name', 'Unknown')}
        - Evolution 적용: {hypothesis.get('evolution_applied', False)}
        - Elo 순위: {hypothesis.get('rank', 'N/A')}
        
        **평가 기준:**
        1. **과학적 엄밀성** (Scientific Rigor): 논리적 일관성, 과학적 근거의 타당성
        2. **논리적 일관성** (Logical Coherence): 추론 과정의 체계성과 명확성
        3. **증거 통합** (Evidence Integration): 데이터와 문헌 정보의 효과적 활용
        4. **실용적 적용가능성** (Practical Applicability): 실제 연구/개발에의 적용 가능성
        
        **평가 결과를 JSON 형식으로 제공:**
        {{
            "overall_score": 0-100,
            "criteria_scores": {{
                "scientific_rigor": 0-100,
                "logical_coherence": 0-100,
                "evidence_integration": 0-100,
                "practical_applicability": 0-100
            }},
            "strengths": ["강점1", "강점2", "강점3"],
            "weaknesses": ["약점1", "약점2"],
            "recommendations": ["개선제안1", "개선제안2"],
            "confidence_level": "높음/보통/낮음",
            "research_impact": "높은 영향/보통 영향/낮은 영향"
        }}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 과학적 품질 평가의 최고 전문가입니다. 객관적이고 건설적인 평가를 제공합니다."},
                {"role": "user", "content": quality_prompt}
            ],
            temperature=0.2
        )
        
        try:
            assessment = json.loads(response.choices[0].message.content)
        except:
            # JSON 파싱 실패 시 기본값
            assessment = {
                "overall_score": 75,
                "criteria_scores": {
                    "scientific_rigor": 75,
                    "logical_coherence": 75,
                    "evidence_integration": 75,
                    "practical_applicability": 75
                },
                "strengths": ["과학적 근거 제시", "논리적 설명", "실용적 접근"],
                "weaknesses": ["추가 검증 필요"],
                "recommendations": ["실험적 검증 수행"],
                "confidence_level": "보통",
                "research_impact": "보통 영향"
            }
        
        return assessment
    
    async def _generate_comprehensive_report(self, final_hypotheses: List[Dict], quality_assessments: List[Dict], shared_context: Dict) -> Dict:
        """종합 리포트 생성"""
        
        # 최고 품질 가설들 선별 (상위 3개)
        top_hypotheses = []
        for i, (hypothesis, assessment) in enumerate(zip(final_hypotheses[:3], quality_assessments[:3])):
            final_hypothesis_text = hypothesis.get('improved_hypothesis', 
                                                 hypothesis.get('original_hypothesis', {}).get('hypothesis', ''))
            
            top_hypotheses.append({
                'rank': i + 1,
                'agent_name': hypothesis.get('agent_name', 'Unknown'),
                'final_hypothesis': final_hypothesis_text,
                'final_score': assessment['overall_score'],
                'quality_scores': assessment['criteria_scores'],
                'evolution_applied': hypothesis.get('evolution_applied', False),
                'elo_score': hypothesis.get('elo_score', 1500),
                'strengths': assessment['strengths'],
                'weaknesses': assessment['weaknesses'],
                'confidence_level': assessment['confidence_level'],
                'research_impact': assessment['research_impact']
            })
        
        # 프로세스 메타데이터
        total_time = time.time() - shared_context.get('timestamp', time.time())
        evolution_count = sum(1 for h in final_hypotheses if h.get('evolution_applied', False))
        
        comprehensive_report = {
            'ranked_hypotheses': top_hypotheses,
            'process_metadata': {
                'total_time': total_time,
                'evolution_applied': f"{evolution_count}개 가설 개선됨" if evolution_count > 0 else "생략됨",
                'total_agents': len(final_hypotheses),
                'elo_consensus': final_hypotheses[0].get('consensus_score', 0) if final_hypotheses else 0
            },
            'literature_context': shared_context.get('literature_context'),
            'cliff_context': shared_context.get('cliff_summary'),
            'generation_timestamp': datetime.now().isoformat(),
            'system_version': '온라인 토론 시스템 v1.0'
        }
        
        return comprehensive_report
    
    def _display_quality_assessment(self, hypothesis: Dict, assessment: Dict, rank: int):
        """품질 평가 결과 표시"""
        with st.expander(f"📊 가설 {rank} 품질 평가 - {hypothesis.get('agent_name', 'Unknown')}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**종합 평가:**")
                overall_score = assessment['overall_score']
                score_color = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
                st.write(f"{score_color} 종합 점수: {overall_score}/100")
                
                if assessment.get('strengths'):
                    st.write("**주요 강점:**")
                    for strength in assessment['strengths'][:3]:
                        st.write(f"• {strength}")
                        
                if assessment.get('weaknesses'):
                    st.write("**개선점:**")
                    for weakness in assessment['weaknesses'][:2]:
                        st.write(f"• {weakness}")
            
            with col2:
                st.write("**세부 점수:**")
                criteria_names = {
                    'scientific_rigor': '과학적 엄밀성',
                    'logical_coherence': '논리적 일관성', 
                    'evidence_integration': '증거 통합',
                    'practical_applicability': '실용성'
                }
                
                for criterion, score in assessment['criteria_scores'].items():
                    criterion_kr = criteria_names.get(criterion, criterion)
                    st.metric(criterion_kr, f"{score}/100")
                
                st.metric("신뢰도", assessment.get('confidence_level', '보통'))
                st.metric("연구 영향", assessment.get('research_impact', '보통 영향'))


class HypothesisEvaluationExpert:
    """가설 평가 전문가 에이전트 - shared_context를 완전히 활용한 맥락 기반 평가"""
    
    def __init__(self, llm_client: UnifiedLLMClient):
        self.llm_client = llm_client
        self.persona = """당신은 15년 경력의 SAR 분석 평가 전문가입니다.
        Activity Cliff 분석, 가설 검증, 과학적 엄밀성 평가에 특화되어 있으며,
        실제 데이터와 문헌 근거를 바탕으로 객관적이고 일관된 평가를 수행합니다."""
    
    def evaluate(self, hypothesis: Dict, shared_context: Dict) -> Dict:
        """맥락 기반 가설 품질 평가 - 기존 evaluate_hypothesis_quality 로직을 클래스로 이전"""
        
        # shared_context에서 핵심 정보 추출
        cliff_summary = shared_context.get('cliff_summary', {})
        literature_context = shared_context.get('literature_context', {})
        target_name = shared_context.get('target_name', '알 수 없음')
        
        # Activity Cliff 정보 구성
        cliff_info = ""
        if cliff_summary:
            high_comp = cliff_summary.get('high_activity_compound', {})
            low_comp = cliff_summary.get('low_activity_compound', {})
            metrics = cliff_summary.get('cliff_metrics', {})
            
            cliff_info = f"""
    **Activity Cliff 분석 대상:**
    - 타겟: {target_name}
    - 고활성 화합물: {high_comp.get('id', 'N/A')} (pKi: {high_comp.get('pki', 'N/A')})
    - 저활성 화합물: {low_comp.get('id', 'N/A')} (pKi: {low_comp.get('pki', 'N/A')})
    - 활성도 차이: {metrics.get('activity_difference', 'N/A')}
    - 구조 유사도: {metrics.get('structural_similarity', 'N/A')}"""
        
        # 문헌 정보 구성
        literature_info = ""
        if literature_context and isinstance(literature_context, dict):
            title = literature_context.get('title', '')
            abstract = literature_context.get('abstract', '')
            if title:
                literature_info = f"""
    **관련 문헌 근거:**
    - 제목: {title[:100]}...
    - 요약: {abstract[:200] if abstract else '요약 없음'}...
    - 관련성: {literature_context.get('relevance_score', 'Medium')}"""
        
        # 맥락 기반 평가 프롬프트
        evaluation_prompt = f"""
    다음 SAR 분석 가설을 **실제 Activity Cliff 데이터와 문헌 근거를 바탕으로** 0-100점 척도로 평가해주세요:
    
    **평가할 가설:**
    {hypothesis.get('hypothesis', '')[:800]}
    {cliff_info}
    {literature_info}
    
    **맥락 기반 평가 기준:**
    1. **과학적 엄밀성 (0-100)**: 가설이 과학적으로 타당하고 검증 가능한가?
    2. **논리적 일관성 (0-100)**: 가설 내 논리가 일관되고 모순이 없는가?
    3. **증거 활용도 (0-100)**: Activity Cliff 데이터와 문헌 근거를 얼마나 잘 활용했는가?
    4. **실용성 (0-100)**: {target_name} 타겟에 대한 신약 개발에 실질적으로 도움이 되는가?
    5. **데이터 부합성 (0-100)**: 실제 Activity Cliff 관찰 결과와 얼마나 일치하는가?
    
    **중요**: 가설이 실제 데이터(pKi 값, 구조 유사도, 활성도 차이)와 얼마나 부합하는지 반드시 고려하세요.
    
    **결과를 JSON 형식으로 제공:**
    {{
        "scientific_rigor": [점수],
        "logical_coherence": [점수],
        "evidence_integration": [점수],
        "practical_applicability": [점수],
        "data_consistency": [점수],
        "overall_score": [5개 점수의 평균],
        "strengths": ["강점1", "강점2", "강점3"],
        "weaknesses": ["약점1", "약점2"],
        "context_relevance": "가설이 Activity Cliff 데이터와 얼마나 관련있는지 설명"
    }}
    """
        
        try:
            response_text = self.llm_client.generate_response(
                self.persona, 
                evaluation_prompt, 
                temperature=0.3
            ).strip()
            
            # JSON 추출
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                evaluation = json.loads(json_text)
                
                # 필수 필드 확인 및 기본값 설정 (5개 평가 기준 포함)
                scores = {
                    'scientific_rigor': evaluation.get('scientific_rigor', 75),
                    'logical_coherence': evaluation.get('logical_coherence', 75),
                    'evidence_integration': evaluation.get('evidence_integration', 75),
                    'practical_applicability': evaluation.get('practical_applicability', 75),
                    'data_consistency': evaluation.get('data_consistency', 75)
                }
                
                overall_score = evaluation.get('overall_score', sum(scores.values()) / len(scores))
                
                return {
                    'scores': scores,
                    'overall_score': overall_score,
                    'strengths': evaluation.get('strengths', ['체계적 분석', 'Activity Cliff 고려']),
                    'weaknesses': evaluation.get('weaknesses', ['개선 여지 있음']),
                    'context_relevance': evaluation.get('context_relevance', 'Activity Cliff 데이터와 연관성 분석됨')
                }
                
        except Exception:
            # 평가 실패 시 기본값 반환
            pass
        
        # 기본 평가 점수 (5개 기준 포함)
        return {
            'scores': {
                'scientific_rigor': 75,
                'logical_coherence': 75,
                'evidence_integration': 75,
                'practical_applicability': 75,
                'data_consistency': 75
            },
            'overall_score': 75,
            'strengths': ['전문가 분석 수행', 'Activity Cliff 데이터 고려'],
            'weaknesses': ['추가 검증 필요'],
            'context_relevance': 'Activity Cliff 맥락에서 기본 평가 수행됨'
        }


# 시각적 표시 함수들
def display_expert_result(result: Dict):
    """각 전문가 결과 표시"""
    with st.expander(f"{result['agent_name']} 결과", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("**생성된 가설:**")
            st.write(result['hypothesis'][:300] + "..." if len(result['hypothesis']) > 300 else result['hypothesis'])
            
        with col2:
            st.metric("신뢰도", f"{result['confidence']:.0%}")
            
            st.write("**핵심 인사이트:**")
            for insight in result['key_insights'][:3]:
                st.write(f"• {insight}")


# 기존 복잡한 display_final_results 함수는 display_simplified_results로 대체됨


# 메인 온라인 토론 시스템 함수
def run_online_discussion_system(selected_cliff: Dict, target_name: str, api_key: str, llm_provider: str = "OpenAI") -> Dict:
    """단순화된 Co-Scientist 방법론 기반 가설 생성 시스템"""
    
    start_time = time.time()
    
    # 통합 LLM 클라이언트 생성
    llm_client = UnifiedLLMClient(api_key, llm_provider)
    
    st.markdown("**Co-Scientist 방법론 기반 SAR 분석**")
    st.markdown(f"3명의 전문가 Agent가 독립적으로 분석한 후 상호 평가를 통해 최고 품질의 가설을 생성합니다.")
    
    # Phase 1: 데이터 준비 + RAG 통합
    st.info("**Phase 1: 데이터 준비** - RAG 통합 컨텍스트 구성")
    shared_context = prepare_shared_context(selected_cliff, target_name)
    
    # 컨텍스트 정보 표시
    with st.expander("분석 대상 정보", expanded=False):
        cliff_summary = shared_context['cliff_summary']
        st.write(f"**고활성 화합물:** {cliff_summary['high_activity_compound']['id']} (pKi: {cliff_summary['high_activity_compound']['pki']:.2f})")
        st.write(f"**저활성 화합물:** {cliff_summary['low_activity_compound']['id']} (pKi: {cliff_summary['low_activity_compound']['pki']:.2f})")
        st.write(f"**활성도 차이:** {cliff_summary['cliff_metrics']['activity_difference']:.2f}")
    
    # RAG 문헌 검색 결과 표시
    if shared_context.get('literature_context'):
        rag_context = shared_context['literature_context']
        with st.expander("검색된 참고 문헌", expanded=False):
            st.markdown(f"**제목:** {rag_context.get('title', 'N/A')}")
            abstract = rag_context.get('abstract', '')
            if abstract:
                display_abstract = abstract[:300] + "..." if len(abstract) > 300 else abstract
                st.markdown(f"**요약:** {display_abstract}")
    
    # Phase 2: Generation - 3개 전문가 독립 분석
    st.markdown("---")
    st.info("**Phase 2: Generation** - 3명의 전문가 Agent가 각자의 관점에서 독립적으로 가설을 생성합니다")
    domain_hypotheses = generation_phase(shared_context, llm_client)
    
    # Phase 3: 전문가 기반 평가 및 순위 매김
    st.markdown("---")
    st.info("**Phase 3: 전문가 평가** - 평가 전문 Agent가 Activity Cliff 데이터와 문헌 근거를 바탕으로 가설을 평가합니다")
    
    # 평가 전문가 에이전트 초기화
    evaluator = HypothesisEvaluationExpert(llm_client)
    evaluated_hypotheses = []
    
    progress_bar = st.progress(0)
    for i, hypothesis in enumerate(domain_hypotheses):
        progress_bar.progress((i + 1) / len(domain_hypotheses))
        
        # 평가 전문가를 통한 가설 품질 평가
        agent_name = hypothesis.get('agent_name', f'전문가 {i+1}')
        with st.spinner(f"평가 전문가가 {agent_name}의 가설을 Activity Cliff 데이터 기반으로 평가 중..."):
            quality_score = evaluator.evaluate(hypothesis, shared_context)
        
        evaluated_hypothesis = {
            'rank': i + 1,
            'agent_name': hypothesis.get('agent_name', f'전문가 {i+1}'),
            'hypothesis': hypothesis.get('hypothesis', ''),
            'confidence': hypothesis.get('confidence', 0.7),
            'quality_scores': quality_score['scores'],
            'overall_score': quality_score['overall_score'],
            'strengths': quality_score['strengths'],
            'weaknesses': quality_score['weaknesses'],
            'context_relevance': quality_score.get('context_relevance', '')
        }
        
        evaluated_hypotheses.append(evaluated_hypothesis)
    
    progress_bar.empty()
    
    # 점수순 정렬
    evaluated_hypotheses.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # 순위 재배정
    for i, hyp in enumerate(evaluated_hypotheses):
        hyp['rank'] = i + 1
    
    # 최종 리포트 생성
    final_report = {
        'ranked_hypotheses': evaluated_hypotheses,
        'process_metadata': {
            'total_time': time.time() - start_time,
            'total_agents': len(domain_hypotheses),
            'analysis_method': 'Co-Scientist 단순화 버전',
            'quality_assessment': True
        },
        'literature_context': shared_context.get('literature_context'),
        'cliff_context': shared_context.get('cliff_summary')
    }
    
    st.markdown("---")
    st.info("**Phase 4: 가설 리포트 생성 및 분석 결과**")
    st.success(f"**총 소요 시간:** {final_report['process_metadata']['total_time']:.1f}초")
    
    # 최종 결과 표시
    display_simplified_results(final_report)
    
    return final_report


# 기존 함수 - HypothesisEvaluationExpert 클래스로 대체됨
def evaluate_hypothesis_quality(hypothesis: Dict, shared_context: Dict, api_key: str) -> Dict:
    """DEPRECATED: HypothesisEvaluationExpert 클래스를 사용하세요"""
    # 호환성을 위해 새 클래스로 리다이렉트
    evaluator = HypothesisEvaluationExpert(api_key)
    return evaluator.evaluate(hypothesis, shared_context)


def display_simplified_results(final_report: Dict):
    """단순화된 최종 결과 표시"""
    
    # 프로세스 요약 (표시 생략)
    
    # 상위 3개 가설 표시
    hypotheses = final_report.get('ranked_hypotheses', [])[:3]
    
    for i, hypothesis in enumerate(hypotheses):
        st.markdown("<br>", unsafe_allow_html=True)
        
        rank_emoji = ["🥇", "🥈", "🥉"][i]
        agent_name = hypothesis.get('agent_name', f'전문가 {i+1}')
        overall_score = hypothesis.get('overall_score', 0)
        
        st.markdown(f"### {rank_emoji} **{agent_name}** (종합점수: {overall_score:.0f}/100)")
        
        # 2열 레이아웃
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 가설 내용
            hypothesis_text = hypothesis.get('hypothesis', '가설을 찾을 수 없습니다.')
            if hypothesis_text:
                st.markdown(hypothesis_text)
            else:
                st.warning("가설 내용을 표시할 수 없습니다.")
                
        with col2:
            st.markdown("**품질 평가**")
            
            # 품질 점수 표시
            quality_scores = hypothesis.get('quality_scores', {})
            
            st.metric("과학적 엄밀성", f"{quality_scores.get('scientific_rigor', 0):.0f}/100")
            st.metric("논리적 일관성", f"{quality_scores.get('logical_coherence', 0):.0f}/100")
            st.metric("증거 활용도", f"{quality_scores.get('evidence_integration', 0):.0f}/100")
            st.metric("실용성", f"{quality_scores.get('practical_applicability', 0):.0f}/100")
            st.metric("데이터 부합성", f"{quality_scores.get('data_consistency', 0):.0f}/100")
            
            # 신뢰도
            confidence = hypothesis.get('confidence', 0.7)
            st.metric("신뢰도", f"{confidence:.0%}")
        
        # 강점과 약점 + 평가 전문가의 상세 분석
        with st.expander(f"{agent_name} 상세 평가 (평가 전문가 분석)", expanded=False):
            # 평가 전문가의 5개 평가 기준 상세 표시
            st.markdown("**평가 전문가의 세부 점수:**")
            score_cols = st.columns(5)
            
            criterion_names = [
                ('scientific_rigor', '과학적 엄밀성'),
                ('logical_coherence', '논리적 일관성'), 
                ('evidence_integration', '증거 활용도'),
                ('practical_applicability', '실용성'),
                ('data_consistency', '데이터 부합성')
            ]
            
            for idx, (key, name) in enumerate(criterion_names):
                with score_cols[idx]:
                    score = quality_scores.get(key, 0)
                    st.metric(name, f"{score:.0f}")
            
            st.markdown("---")
            
            # 맥락 연관성 표시 (컴팩트하게)
            context_relevance = hypothesis.get('context_relevance', '')
            if context_relevance:
                st.markdown("**Activity Cliff 데이터 연관성:**")
                st.write(context_relevance)
                st.markdown("---")
            
            # 강점과 약점을 컴팩트하게 배치
            col_strength, col_weakness = st.columns(2)
            
            with col_strength:
                st.markdown("**🟢 주요 강점:**")
                strengths = hypothesis.get('strengths', [])
                for strength in strengths:
                    st.write(f"• {strength}")
                    
            with col_weakness:
                st.markdown("**🟡 개선 포인트:**")
                weaknesses = hypothesis.get('weaknesses', [])
                for weakness in weaknesses:
                    st.write(f"• {weakness}")
        
        # 가설 간 시각적 구분을 위한 여백
        # st.markdown("<br>", unsafe_allow_html=True)
        # st.markdown("---")
        # st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")


def prepare_shared_context(selected_cliff: Dict, target_name: str) -> Dict:
    """기존 RAG 시스템을 활용한 컨텍스트 준비 - 강화된 문헌 근거 제공"""
    
    # 기존 함수들 재사용
    rag_context = search_pubmed_for_context(
        selected_cliff['mol_1']['SMILES'], 
        selected_cliff['mol_2']['SMILES'], 
        target_name
    )
    cliff_summary = get_activity_cliff_summary(selected_cliff)
    
    # RAG 컨텍스트 품질 향상
    if rag_context and isinstance(rag_context, dict):
        # 문헌 정보 강화
        enhanced_rag = rag_context.copy()
        enhanced_rag['relevance_score'] = 'High' if target_name.lower() in rag_context.get('title', '').lower() else 'Medium'
        enhanced_rag['context_type'] = 'SAR Analysis Reference'
        enhanced_rag['usage_instruction'] = f"이 문헌을 {target_name} 타겟에 대한 Activity Cliff 분석의 과학적 근거로 활용하세요"
        rag_context = enhanced_rag
    
    # 모든 에이전트가 공유할 통합 컨텍스트
    shared_context = {
        'cliff_data': selected_cliff,
        'cliff_summary': cliff_summary,
        'literature_context': rag_context,  # 강화된 PubMed 검색 결과
        'target_name': target_name,
        'timestamp': time.time(),
        'context_quality': 'Enhanced' if rag_context else 'Basic',
        'evidence_level': 'Literature-backed' if rag_context else 'Data-only'
    }
    
    return shared_context


def generation_phase(shared_context: Dict, llm_client: UnifiedLLMClient) -> List[Dict]:
    """3개 도메인 전문가 순차 실행 (간소화 버전)"""
    experts = [
        StructuralChemistryExpert(llm_client),
        BiomolecularInteractionExpert(llm_client),
        SARIntegrationExpert(llm_client)
    ]
    
    domain_hypotheses = []
    progress_bar = st.progress(0)
    
    for i, expert in enumerate(experts):
        try:
            with st.spinner(f"{expert.__class__.__name__} 가설 생성 중..."):
                result = expert.generate(shared_context)
                domain_hypotheses.append(result)
                
                # 진행 상황 업데이트
                progress = (i + 1) / len(experts)
                progress_bar.progress(progress)
                
                # 각 전문가 결과 즉시 표시
                display_expert_result(result)
        except Exception as e:
            st.error(f"{expert.__class__.__name__} 생성 중 오류: {str(e)}")
            # 기본 결과 생성
            result = {
                'agent_type': 'error',
                'agent_name': f"❌ {expert.__class__.__name__}",
                'hypothesis': f"가설 생성 중 오류 발생: {str(e)}",
                'confidence': 0.5,
                'key_insights': ['오류 발생'],
                'reasoning_steps': ['오류로 인한 중단'],
                'timestamp': time.time()
            }
            domain_hypotheses.append(result)
    
    progress_bar.empty()  # Phase 2 진행바 숨기기
    return domain_hypotheses