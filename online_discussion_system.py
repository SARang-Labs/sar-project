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
from llm_debate.tools.docking_simulator import DockingSimulator

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


# ========================================
# 할루시네이션 방지 안전 가이드라인
# ========================================
ANTI_HALLUCINATION_GUIDELINES = """
**전체 전문가 Agent 공통 할루시네이션 방지 가이드라인:**

1. **금지된 통계적 모델링:**
   - 실제 계산하지 않은 회귀방정식, R², RMSE, Q², p-value 생성 금지
   - 가짜 머신러닝 모델 성능 지표 (정확도, F1-score, ROC-AUC) 생성 금지
   - 존재하지 않는 QSAR 모델이나 예측 모델 결과 생성 금지

2. **금지된 정량적 예측:**
   - 구체적 pIC50, Ki, IC50 예측값 생성 금지 ("예측 pIC50: 7.2" 등)
   - 구체적 결합 친화도 수치 예측 금지 ("ΔG = -8.5 kcal/mol" 등)
   - 구체적 선택성 비율 예측 금지 ("100배 향상된 선택성" 등)

3. **허용된 정성적 분석:**
   - 제공된 실제 데이터(SMILES, pIC50, 물성)만을 근거로 한 해석
   - 문헌 기반 메커니즘 추론과 화학적 논리 제시
   - 구조적 차이와 활성 변화의 연관성 해석

4. **도킹 시뮬레이션 결과 활용:**
   - 도킹 결과가 제공된 경우, 이를 추가적인 근거로 활용
   - 도킹 스코어와 상호작용 패턴을 구조-활성 관계 해석에 참고
   - 실험 데이터와 도킹 결과를 종합적으로 고려

5. **검증 기준:**
   - 모든 수치는 제공된 입력 데이터에서만 인용
   - 모든 메커니즘은 일반적 화학 지식에 근거
   - 도킹 결과가 있으면 보조적 근거로 활용
   - 모든 예측은 정성적 방향성만 제시 ("증가할 것으로 예상", "감소 가능성")

이 가이드라인을 위반하는 내용이 감지되면 즉시 수정하고 정성적 분석으로 대체해야 함.
"""

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
        구조 A: 클로르디아제폭시드 (pIC50: 7.2) vs 구조 B: 디아제팜 (pIC50: 8.9)
        
        1. 구조 비교: A는 N-옥사이드 형태, B는 7번 위치에 염소 치환
        2. 물리화학적 영향: N-옥사이드 제거로 전자밀도 증가, 지용성 향상 (LogP +0.8)
        3. 생체 상호작용: GABA 수용체와의 결합 기하학 개선, π-π 스택킹 강화
        4. 활성 변화 연결: 개선된 단백질 적합성으로 1.7 pIC50 단위 활성 증가
        5. 추가 실험: 분자 도킹 시뮬레이션으로 결합 모드 확인, ADMET 예측
        
        [귀하의 분석 과제]
        """
        
        return f"""
        당신은 20년 경력의 선임 약화학자입니다. SAR과 Activity Cliff 분석에서 분자 구조와 전자적 특성 변화 분석의 전문가로서, 실제 신약 개발 현장에서 사용하는 체계적 분석 절차를 따라 정확하고 신뢰할 수 있는 가설을 생성해주세요.
        
        {few_shot_example}
        
        **Activity Cliff 분석 대상:**
        
        **화합물 정보:**
        - 고활성 화합물: {high_active['id']} (pIC50: {high_active['pic50']})
          SMILES: {high_active['smiles']}
        - 저활성 화합물: {low_active['id']} (pIC50: {low_active['pic50']})
          SMILES: {low_active['smiles']}
        
        **In-Context 구조적 특성 (할루시네이션 방지용):**
        - Tanimoto 유사도: {metrics['similarity']:.3f}
        - 활성도 차이: {metrics['activity_difference']}
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
        
        4. **활성 변화 연결**: 이 가설이 관찰된 Activity Cliff ({metrics['activity_difference']} pIC50 단위 차이)를 어떻게 설명하는지 논리적으로 연결하세요.
        
        5. **추가 실험 제안**: 검증을 위한 분자 도킹, ADMET 예측, 계산화학 실험 등 후속 실험을 구체적으로 제안하세요.
        
        **필수 요구사항 - 전문가 수준의 분석:**
        1. 구체적 수치 데이터 포함 (LogP, MW, TPSA 등)
        2. 원자 단위 구조 차이 명시 (C-N 결합 → C-O 결합 등)
        3. 정성적 활성 변화 해석 ("활성 감소 예상", "결합 친화도 향상 가능성" 등)
        4. 구체적 실험 프로토콜 ("AutoDock4로 100회 도킹" 등)
        5. 특정 분자 대상 제시 ("메틸에스터 치환체" 등)
        
        **금지 사항 - 다음과 같은 모호한 표현 금지:**
        - "~일 것으로 예상된다", "~로 추정된다"
        - "가능성이 있다", "보인다", "생각된다"
        - "일반적으로", "대개", "보통"
        
        **실제 제약회사 수준의 분석을 수행하여 즉시 합성 대상으로 사용할 수 있는 구체적 가설을 제시하세요.**
        
        **결과 형식 (반드시 이 형식을 정확히 따르세요):**
        
        핵심 가설: [구체적이고 전문적인 1-2문장, 예: "N-메틸기 추가로 인한 입체장애가 Asp381과의 수소결합을 방해하여 활성 감소를 초래"]
        
        상세 분석:
        1. 구조 비교: [SMILES 구조의 정확한 차이점, 원자 번호와 결합 유형 명시]
        2. 물리화학적 영향: [LogP, TPSA, 분자량 변화의 구체적 수치와 의미]
        3. 생체 상호작용 가설: [특정 아미노산 잔기와의 상호작용 변화, 결합 에너지 추정]
        4. 활성 변화 연결: [정량적 구조-활성 관계 설명]
        5. 추가 실험 제안: [구체적 프로토콜과 예상 결과]
        
        분자 설계 제안: [후속 화합물의 구체적 구조 변경 전략 - pIC50 예측값 언급 금지]
        
        {ANTI_HALLUCINATION_GUIDELINES}
        - 제공된 실제 데이터(SMILES, pIC50, 물성)만을 근거로 정성적 해석
        
        **중요: 전문가가 알고 있을 뻔한 기본 내용은 피하고, 실질적이고 깊이 있는 구조생물학적 통찰을 제공하세요. 구체적 수치, 특정 분자 부위, 명확한 메커니즘을 포함하되 예상 pIC50 값은 언급하지 마세요.**
        """
    
    
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
        self.docking_simulator = DockingSimulator()
    
    def generate(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """생체분자 상호작용 관점의 가설 생성 (도킹 시뮬레이션 포함)"""
        
        # 도킹 시뮬레이션 수행
        docking_results = self._perform_docking_analysis(shared_context)
        
        # 도킹 결과를 포함한 프롬프트 생성
        prompt = self._build_interaction_prompt(shared_context, docking_results)
        hypothesis = self.llm_client.generate_response(self.persona, prompt, temperature=0.7)
        
        return {
            'agent_type': 'biomolecular_interaction',
            'agent_name': '생체분자 상호작용 전문가',
            'hypothesis': hypothesis,
            'key_insights': self._extract_key_insights(hypothesis),
            'reasoning_steps': self._extract_reasoning_steps(hypothesis),
            'docking_analysis': docking_results,  # 도킹 분석 결과 포함
            'timestamp': time.time()
        }
    
    def _perform_docking_analysis(self, shared_context: Dict[str, Any]) -> Dict:
        """도킹 시뮬레이션 수행"""
        cliff_summary = shared_context.get('cliff_summary', {})
        target_name = shared_context.get('target_name', 'EGFR')
        
        high_active = cliff_summary.get('high_activity_compound', {})
        low_active = cliff_summary.get('low_activity_compound', {})
        
        results = {}
        
        # 고활성 화합물 도킹
        if high_active.get('smiles'):
            docking_result = self.docking_simulator.perform_docking(
                high_active['smiles'],
                target_name=target_name
            )
            results['high_active_docking'] = {
                'binding_affinity': docking_result.binding_affinity,
                'ki_estimate': 10 ** (-docking_result.binding_affinity / 1.36) * 1000,  # nM 단위 (μM에서 변환)
                'interactions': docking_result.interactions,
                'rmsd': docking_result.rmsd_lb
            }
        
        # 저활성 화합물 도킹
        if low_active.get('smiles'):
            docking_result = self.docking_simulator.perform_docking(
                low_active['smiles'],
                target_name=target_name
            )
            results['low_active_docking'] = {
                'binding_affinity': docking_result.binding_affinity,
                'ki_estimate': 10 ** (-docking_result.binding_affinity / 1.36) * 1000,  # nM 단위 (μM에서 변환)
                'interactions': docking_result.interactions,
                'rmsd': docking_result.rmsd_lb
            }
        
        # 도킹 결과 비교 분석
        if 'high_active_docking' in results and 'low_active_docking' in results:
            high_score = results['high_active_docking']['binding_affinity']
            low_score = results['low_active_docking']['binding_affinity']
            results['comparative_analysis'] = {
                'affinity_difference': high_score - low_score,
                'predicted_activity_ratio': abs(high_score / low_score) if low_score != 0 else None,
                'supports_activity_cliff': high_score < low_score  # 낮은 값이 더 강한 결합
            }
        
        return results
    
    def _build_interaction_prompt(self, shared_context: Dict[str, Any], docking_results: Dict = None) -> str:
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
        화합물 A: 게피티니브 (pIC50: 7.8) vs 화합물 B: 엘로티니브 (pIC50: 8.5)
        
        1. 단백질-리간드 결합: 퀴나졸린 코어의 6,7위치 치환기 차이가 ATP 결합 포켓과의 상호작용 패턴 변화
        2. 상호작용 패턴: 엘로티니브의 아세틸렌 링커가 Cys797과 새로운 소수성 접촉 형성
        3. 결합 기하학: 추가 아로마틱 고리가 DFG 루프와의 π-π 스택킹 개선
        4. 약리학적 메커니즘: 향상된 결합 기하학으로 0.7 pIC50 단위 친화도 증가
        5. ADMET 영향: CYP3A4 대사 안정성 개선, 반감기 연장
        
        [귀하의 분석 과제]
        """
        
        return f"""
        당신은 단백질-리간드 상호작용 메커니즘 분야의 세계적 권위자입니다. 타겟 단백질과의 결합 방식 변화, 약리학적 관점과 생리활성 메커니즘 규명을 전문으로 하는 선임 연구자로서, 실제 신약 개발에서 사용되는 체계적 분석을 수행해주세요.
        
        {few_shot_example}
        
        **Activity Cliff 분석 대상:**
        
        **타겟 단백질:** {target_name}
        
        **화합물 정보:**
        - 고활성 화합물: {high_active['id']} (pIC50: {high_active['pic50']})
          SMILES: {high_active['smiles']}
        - 저활성 화합물: {low_active['id']} (pIC50: {low_active['pic50']})
          SMILES: {low_active['smiles']}
        
        **In-Context 생화학적 특성 (할루시네이션 방지용):**
        - 활성도 차이: {metrics['activity_difference']} pIC50 단위
        - 구조 유사도: {metrics['similarity']:.3f} (Tanimoto)
        - 분자량 차이: {prop_diffs['mw_diff']:.2f} Da
        - LogP 차이: {prop_diffs['logp_diff']:.2f}
        - TPSA 차이: {prop_diffs.get('tpsa_diff', 0):.2f} Ų
        - 수소결합 공여자/수용자 변화 예상
        
        {literature_info}
        
        {self._format_docking_results(docking_results) if docking_results else ""}
        
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
        
        핵심 가설: [구체적이고 전문적인 메커니즘, 예: "Phe256과의 π-π 스택킹 상실로 인한 결합 친화도 감소가 주요 원인"]
        
        상세 분석:
        1. 단백질-리간드 결합: [특정 결합 포켓, 잔기 번호, 상호작용 유형 명시]
        2. 상호작용 패턴: [수소결합 길이, 소수성 접촉 면적의 구체적 변화]
        3. 결합 기하학: [RMSD, 결합각, 비틀림각의 정량적 분석]
        4. 약리학적 메커니즘: [결합 기전 해석, 선택성 차이 원인 분석]
        5. ADMET 영향: [대사 패턴 차이, 약동학적 특성 변화 해석]
        
        분자 설계 제안: [특정 치환기 도입 전략 - pIC50 예측값 언급 금지]
        
        {ANTI_HALLUCINATION_GUIDELINES}
        - 제공된 실제 데이터만을 근거로 정성적 메커니즘 해석

        **중요: 신약개발 전문가가 이미 알고 있는 뻔한 내용은 제외하고, 깊이 있는 약물화학적 분석에 집중하세요. 결합 패턴, 상호작용 유형, 특정 아미노산 잔기와의 관계를 포함한 정성적 분석을 제시하되 예상 pIC50 값은 언급하지 마세요.**
        """
    
    def _format_docking_results(self, docking_results: Dict) -> str:
        """도킹 시뮬레이션 결과를 프롬프트용 텍스트로 포맷팅"""
        if not docking_results:
            return ""
        
        formatted = "\n**도킹 시뮬레이션 결과:**\n"
        
        if 'high_active_docking' in docking_results:
            high = docking_results['high_active_docking']
            formatted += f"- 고활성 화합물: 결합 친화도 {high['binding_affinity']:.1f} kcal/mol, Ki 추정값 {high['ki_estimate']:.2f} nM\n"
        
        if 'low_active_docking' in docking_results:
            low = docking_results['low_active_docking']
            formatted += f"- 저활성 화합물: 결합 친화도 {low['binding_affinity']:.1f} kcal/mol, Ki 추정값 {low['ki_estimate']:.2f} nM\n"
        
        if 'comparative_analysis' in docking_results:
            comp = docking_results['comparative_analysis']
            formatted += f"- 친화도 차이: {comp['affinity_difference']:.1f} kcal/mol\n"
            if comp['supports_activity_cliff']:
                formatted += "- 도킹 결과는 실험적 Activity Cliff를 지지합니다.\n"
            else:
                formatted += "- 도킹 결과는 실험 데이터와 상이한 경향을 보입니다 (추가 분석 필요).\n"
        
        return formatted
    
    
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


class QSARExpert:
    """QSAR 전문가 에이전트"""
    
    def __init__(self, llm_client: UnifiedLLMClient):
        self.llm_client = llm_client
        self.persona = """당신은 구조-활성 관계(SAR) 분석의 세계적 전문가입니다.
        제공된 실제 데이터(SMILES, pIC50, 물리화학적 특성)를 바탕으로 분자 특성 변화와 활성 차이 간의 
        정성적 관계를 분석하고, 문헌 근거와 화학적 지식을 토대로 구조-활성 관계 패턴을 해석하는 것이 전문입니다.
        
        **중요**: 실제 계산하지 않은 QSAR 모델, 회귀방정식, 통계값(R², RMSE 등)을 절대 생성하지 마세요."""
    
    def generate(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """QSAR 관점의 가설 생성"""
        prompt = self._build_qsar_prompt(shared_context)
        hypothesis = self.llm_client.generate_response(self.persona, prompt, temperature=0.7)
        
        return {
            'agent_type': 'qsar',
            'agent_name': 'QSAR 전문가',
            'hypothesis': hypothesis,
            'key_insights': self._extract_key_insights(hypothesis),
            'reasoning_steps': self._extract_reasoning_steps(hypothesis),
            'timestamp': time.time()
        }
    
    def _build_qsar_prompt(self, shared_context: Dict[str, Any]) -> str:
        """QSAR 전문가용 특화 프롬프트 생성 - CoT.md 지침 반영"""
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
        
        # Few-Shot 예시 (Activity Cliff 활성 차이 원인 분석 사례)
        few_shot_example = """
        **Few-Shot 예시 - 전문가 분석 과정 참조:**
        
        [예시] ACE 억제제 계열 Activity Cliff 활성 차이 원인 분석:
        시리즈: 캅토프릴 → 에날라프릴 (pIC50: 6.5 → 8.2)
        
        1. 활성 차이 패턴: 티올기 → 카르복실기 변경으로 1.7 pIC50 단위 활성 증가
        2. 화학정보학 인사이트: 낮은 Tanimoto 유사도(0.4)에도 큰 활성 차이는 약물발견의 전환점
        3. 신약 개발 전략: 프로드러그 전략 도입으로 ADMET 특성 개선
        4. 최적화 방향: 아연 결합 모티프 최적화가 핵심, 주변 치환기는 선택성 조절
        5. 메커니즘 해석: 금속 배위 결합 강화가 효소-억제제 친화도를 크게 향상시킴
        
        [귀하의 분석 과제]
        """
        
        return f"""
        당신은 Activity Cliff 활성 차이 원인 분석의 세계적 전문가입니다. 제공된 실제 데이터를 바탕으로 분자 특성 차이와 활성 변화를 화학적 논리로 해석하는 것이 전문 분야이며, Activity Cliff 현상을 실제 구조 변화와 물리화학적 특성으로 설명하는 것이 특기입니다.
        
        {few_shot_example}
        
        **Activity Cliff 분석 대상:**
        
        **타겟 단백질:** {target_name}
        
        **화합물 정보:**
        - 고활성: {high_active['id']} (pIC50: {high_active['pic50']})
          SMILES: {high_active['smiles']}
        - 저활성: {low_active['id']} (pIC50: {low_active['pic50']})
          SMILES: {low_active['smiles']}
        
        **In-Context Activity Cliff 메트릭 (할루시네이션 방지용):**
        - Cliff 점수: {metrics.get('cliff_score', 0):.3f}
        - 구조 유사도: {metrics['similarity']:.3f} (Tanimoto)
        - 활성 차이: {metrics['activity_difference']} pIC50 단위
        - 같은 스캐폴드: {metrics.get('same_scaffold', 'Unknown')}
        - 구조적 차이: {metrics['structural_difference_type']}
        
        **물리화학적 특성 차이:**
        - 분자량: {prop_diffs['mw_diff']:.2f} Da
        - LogP: {prop_diffs['logp_diff']:.2f} (지용성 변화)
        - TPSA: {prop_diffs.get('tpsa_diff', 0):.2f} Ų (극성 표면적 변화)
        
        {literature_info}
        
        **단계별 Activity Cliff 활성 차이 원인 분석 수행:**
        구조-활성 관계 전문가로서 제공된 실제 데이터만을 바탕으로 다음 4단계로 분석하세요:
        
        1. **실제 분자 특성 비교**: 제공된 물리화학적 특성 차이(분자량: {prop_diffs['mw_diff']:.2f} Da, LogP: {prop_diffs['logp_diff']:.2f}, TPSA: {prop_diffs.get('tpsa_diff', 0):.2f} Ų)와 SMILES 구조 차이를 분석하세요. 
           어떤 특성 변화가 {metrics['activity_difference']} pIC50 차이와 관련될 수 있는지 해석하세요.
        
        2. **구조-특성 관계 해석**: 두 화합물의 구조적 차이가 물리화학적 특성에 미치는 영향을 분석하세요. 
           지용성, 극성, 분자 크기 변화가 활성에 어떻게 기여할 수 있는지 설명하세요.
        
        3. **문헌 기반 유사 사례**: 제공된 문헌 정보와 일반적인 Activity Cliff 지식을 바탕으로 유사한 구조 변화 사례를 언급하세요.
           {target_name} 타겟에서 알려진 구조-활성 관계 패턴과 비교하세요.
        
        4. **Activity Cliff 해석**: 높은 구조 유사도({metrics['similarity']:.3f})에도 불구하고 큰 활성 차이가 발생하는 
           구조적 원인을 실제 분자 특성 변화 관점에서 해석하세요.
        
        {ANTI_HALLUCINATION_GUIDELINES}
        
        **전문가 수준의 Activity Cliff 활성 차이 원인 분석을 제공하되, 실제 존재하지 않는 데이터는 절대 생성하지 마세요.**
        
        **정성적 Activity Cliff 활성 차이 원인 분석에 집중하세요:**
        - 제공된 실제 데이터(SMILES, pIC50, 물성)만을 근거로 분석
        - 구조 변화와 활성 차이 간의 화학적 논리 제시
        - 문헌 정보를 바탕으로 한 메커니즘 해석
        - 가짜 통계값이나 모델 결과는 절대 생성 금지
        
        **Activity Cliff 전문가의 화학적 직관과 문헌 지식을 바탕으로 과학적으로 타당한 해석을 제시하세요.**
        
        **결과 형식 (반드시 이 형식을 정확히 따르세요):**
        
        핵심 가설: [제공된 구조적 차이와 물리화학적 특성 변화를 바탕으로 한 메커니즘 가설]
        
        상세 분석:
        1. 실제 분자 특성 비교: [제공된 MW, LogP, TPSA 차이와 활성 차이의 연관성 해석]
        2. 구조-특성 관계: [SMILES 비교를 통한 구조적 차이가 특성에 미치는 영향]
        3. 문헌 기반 해석: [제공된 문헌 정보와 일반적인 Activity Cliff 지식 활용]
        4. Activity Cliff 메커니즘: [구조 유사도 대비 큰 활성 차이의 원인 분석]
        
        구조 최적화 방향: [Activity Cliff 활성 차이 원인 분석을 바탕으로 한 일반적인 구조 개선 방향 제안]
        
        **중요: 실제 제공된 데이터만을 근거로 분석하고, 존재하지 않는 QSAR 모델이나 통계값은 절대 생성하지 마세요.**
        """
    
    
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


class HypothesisEvaluationExpert:
    """가설 평가 전문가 에이전트 - shared_context를 완전히 활용한 맥락 기반 평가"""
    
    def __init__(self, llm_client: UnifiedLLMClient):
        self.llm_client = llm_client
        self.persona = """당신은 15년 경력의 SAR 분석 평가 전문가입니다.
        Activity Cliff 구조 활성 차이 원인 분석, 가설 검증, 과학적 엄밀성 평가에 특화되어 있으며,
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
    - 고활성 화합물: {high_comp.get('id', 'N/A')} (pIC50: {high_comp.get('pki', 'N/A')})
    - 저활성 화합물: {low_comp.get('id', 'N/A')} (pIC50: {low_comp.get('pki', 'N/A')})
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
    
    **평가 기준 (가중치 적용):**
    1. **MMP 재검증 (40%)**: 가설에서 언급한 수치가 실제 데이터와 일치하는지, SMILES/pIC50/구조 유사도가 정확한지, 물성 계산이 올바른지 평가
    2. **SAR 분석 (40%)**: 구조 변화 → 메커니즘 → 활성 변화의 논리가 타당한지, 제시한 메커니즘이 화학적으로 합리적인지, 구체적 분석인지 평가  
    3. **타겟 키워드 (20%)**: {target_name} 타겟 특이적 언급, kinase domain/binding site 등 전문 용어 사용, 타겟의 알려진 특성 반영 정도 평가
    
    **핵심 질문**: 구조적으로 유사한 두 화합물이 {metrics.get('activity_difference', 'N/A')} pIC50 차이를 보이는 구조 활성 차이의 원인은 무엇인가?
    
    **결과를 JSON 형식으로 제공:**
    {{
        "mmp_validation": [점수],
        "sar_analysis": [점수],
        "target_keywords": [점수],
        "overall_score": [가중평균: (mmp_validation*0.4 + sar_analysis*0.4 + target_keywords*0.2)],
        "strengths": ["강점1", "강점2", "강점3"],
        "weaknesses": ["약점1", "약점2"],
        "evaluation_rationale": "이 가설이 Activity Cliff 구조 활성 차이 원인을 얼마나 잘 규명하는지 평가 근거",
        "utilization_plan": {{
            "role": "이 가설의 역할 (예: 주요 베이스, 구조 분석 보완, 메커니즘 설명 보강 등)",
            "utilization_level": "활용 수준 (예: 전체 활용, 핵심 부분만 활용, 특정 통찰만 활용)",
            "core_utilization_parts": ["활용할 구체적 부분1", "활용할 구체적 부분2", "활용할 구체적 부분3"],
            "complementary_parts": ["추가로 보완할 부분1", "추가로 보완할 부분2"],
            "contribution_to_final": "이 가설이 최종 종합 리포트에서 담당할 구체적 역할과 기여 내용"
        }}
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
                
                # 평가_항목.md 기준 필드 확인 및 기본값 설정
                scores = {
                    'mmp_validation': evaluation.get('mmp_validation', 75),
                    'sar_analysis': evaluation.get('sar_analysis', 75),
                    'target_keywords': evaluation.get('target_keywords', 75)
                }
                
                # 가중평균 계산: MMP(40%) + SAR(40%) + Target(20%)
                overall_score = evaluation.get('overall_score', 
                    scores['mmp_validation'] * 0.4 + scores['sar_analysis'] * 0.4 + scores['target_keywords'] * 0.2)
                
                return {
                    'scores': scores,
                    'overall_score': overall_score,
                    'strengths': evaluation.get('strengths', ['체계적 분석', 'Activity Cliff 고려']),
                    'weaknesses': evaluation.get('weaknesses', ['개선 여지 있음']),
                    'evaluation_rationale': evaluation.get('evaluation_rationale', 'Activity Cliff 구조 활성 차이 원인 분석력 기준으로 평가됨'),
                    'utilization_plan': evaluation.get('utilization_plan', {
                        'role': '구조 분석 보완',
                        'utilization_level': '핵심 부분 활용',
                        'core_utilization_parts': ['구조적 차이 분석', '분자적 메커니즘'],
                        'complementary_parts': ['추가 화학적 통찰'],
                        'contribution_to_final': 'Activity Cliff 구조 활성 차이 원인 분석에 구조적 관점 기여'
                    })
                }
                
        except Exception:
            # 평가 실패 시 기본값 반환
            pass
        
        # 평가_항목.md 기준 기본 평가 점수
        return {
            'scores': {
                'mmp_validation': 75,
                'sar_analysis': 75,
                'target_keywords': 75
            },
            'overall_score': 75,  # 가중평균: 75*0.4 + 75*0.4 + 75*0.2 = 75
            'strengths': ['구조 활성 차이 원인 분석 수행', 'Activity Cliff 데이터 고려'],
            'weaknesses': ['추가 검증 필요'],
            'evaluation_rationale': 'Activity Cliff 구조 활성 차이 원인 분석력 기준으로 평가됨',
            'utilization_plan': {
                'role': '전문 분야 분석 기여',
                'utilization_level': '핵심 부분 활용',
                'core_utilization_parts': ['전문가 관점 분석', '구조적 통찰', '메커니즘 설명'],
                'complementary_parts': ['추가적 화학적 해석'],
                'contribution_to_final': 'Activity Cliff 구조 활성 차이 원인 분석에 전문 분야별 관점으로 기여'
            }
        }
    
    def evaluate_hypotheses(self, domain_hypotheses: List[Dict], shared_context: Dict) -> Dict:
        """모든 가설을 종합 평가하고 최종 리포트 생성 - 평가_시스템_가이드.md 기준"""
        
        st.info("**Phase 3: 종합 평가** - 평가 전문 Agent가 각 가설의 구조 활성 차이 원인 분석력을 정량적으로 평가하고 최종 활용 방안을 결정합니다")
        
        # 1단계: 각 가설을 MMP/SAR/Target 기준으로 개별 평가
        individual_evaluations = []
        
        for i, hypothesis in enumerate(domain_hypotheses):
            with st.spinner(f"{hypothesis['agent_name']} 가설 정량적 평가 중..."):
                # MMP/SAR/Target 기준 정량 평가
                quantitative_evaluation = self.evaluate(hypothesis, shared_context)
                
                result = {
                    'hypothesis_id': i,
                    'agent_name': hypothesis['agent_name'], 
                    'original_hypothesis': hypothesis,
                    'mmp_score': quantitative_evaluation['scores']['mmp_validation'],
                    'sar_score': quantitative_evaluation['scores']['sar_analysis'],
                    'target_score': quantitative_evaluation['scores']['target_keywords'],
                    'overall_score': quantitative_evaluation['overall_score'],
                    'strengths': quantitative_evaluation['strengths'],
                    'weaknesses': quantitative_evaluation['weaknesses'],
                    'evaluation_rationale': quantitative_evaluation['evaluation_rationale'],
                    'utilization_plan': quantitative_evaluation['utilization_plan'],  # 활용 계획 추가
                    'timestamp': time.time()
                }
                
                individual_evaluations.append(result)
                
                # 개별 평가 결과 핵심 3가지 표시
                with st.expander(f"{result['agent_name']} 평가 결과", expanded=False):
                    
                    # 1. 평가 점수 표시: MMP/SAR/타겟 키워드별 점수와 총점
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MMP 점수", f"{result['mmp_score']:.0f}/100")
                    with col2:
                        st.metric("SAR 점수", f"{result['sar_score']:.0f}/100")
                    with col3:
                        st.metric("Target 점수", f"{result['target_score']:.0f}/100")
                    with col4:
                        st.metric("종합 점수", f"{result['overall_score']:.1f}/100")
                    
                    # 2. 평가 근거 표시: 각 점수를 준 구체적 이유
                    st.write("**평가 근거:**")
                    st.write(result['evaluation_rationale'])
                    
                    # 3. 활용 방식 표시: 이 가설의 어떤 부분이 최종 가설에서 어떻게 반영될지
                    utilization = result['utilization_plan']
                    st.write("**최종 가설에서의 활용 방식:**")
                    st.write(f"**기여 내용**: {utilization['contribution_to_final']}")
                    st.write(f"**활용할 핵심 부분**: {', '.join(utilization['core_utilization_parts'])}")
        
        # 2단계: 최우수 가설 선정 (강점 개수 기준) + 최종 리포트 생성
        final_report = self._select_best_and_synthesize(individual_evaluations, shared_context)
        
        return final_report
    
    def _select_best_and_synthesize(self, individual_evaluations: List[Dict], shared_context: Dict) -> Dict:
        """2단계: 강점 개수 기준으로 최우수 가설 선정 후 최종 리포트 생성"""
        
        # 강점 개수 기준으로 정렬 (많은 순), 동점 시 전체 점수로 결정
        sorted_evaluations = sorted(
            individual_evaluations, 
            key=lambda x: (len(x['strengths']), x['overall_score']), 
            reverse=True
        )
        
        best_evaluation = sorted_evaluations[0]
        
        # 모든 가설의 강점을 수집
        all_strengths = []
        all_insights = []
        for eval_result in individual_evaluations:
            all_strengths.extend(eval_result['strengths'])
            # 원본 가설에서 key_insights 추출 (있을 경우)
            original_insights = eval_result['original_hypothesis'].get('key_insights', [])
            all_insights.extend(original_insights)
        
        # 최종 종합 리포트 생성 - 평가 중 결정된 활용 방식 기반
        synthesis_prompt = f"""
        다음은 3명의 전문가가 생성한 Activity Cliff 구조 활성 차이 원인 가설들과 평가 전문 Agent가 결정한 활용 방식입니다.
        각 가설의 결정된 활용 방식에 따라 최종 종합 리포트를 작성해주세요.

        **최우수 가설 (주요 베이스):**
        - 전문가: {best_evaluation['agent_name']}
        - 강점 {len(best_evaluation['strengths'])}개: {', '.join(best_evaluation['strengths'])}
        - 종합 점수: {best_evaluation['overall_score']:.1f}점
        - 활용 방식: {best_evaluation['utilization_plan']['contribution_to_final']}
        - 핵심 활용 부분: {', '.join(best_evaluation['utilization_plan']['core_utilization_parts'])}
        - 가설 내용: {best_evaluation['original_hypothesis']['hypothesis'][:500]}...

        **다른 가설들의 결정된 활용 방식:**
        """
        
        for eval_result in sorted_evaluations[1:]:
            utilization = eval_result['utilization_plan']
            synthesis_prompt += f"""
        - {eval_result['agent_name']} ({utilization['role']}):
          * 활용 부분: {', '.join(utilization['core_utilization_parts'])}
          * 보완 부분: {', '.join(utilization['complementary_parts']) if utilization['complementary_parts'] else '없음'}
          * 기여 방식: {utilization['contribution_to_final']}
        """

        synthesis_prompt += f"""
        
        **Activity Cliff 구조 활성 차이 정보:**
        - 타겟: {shared_context.get('target_name', '알 수 없음')}
        - 문헌 근거: {shared_context.get('literature_context', {}).get('title', '관련 문헌 없음')[:100]}

        **통합 원칙:**
        1. **주요 가설을 핵심 베이스**로 사용하되, 다른 가설들의 우수한 요소들을 체계적으로 통합
        2. **구체적이고 혁신적인 내용만** 포함하세요. 다음과 같은 진부한 내용은 **절대 포함 금지**:
           - 뻔한 결론: "실험적 검증이 필요", "추가 연구가 필요", "한계가 있을 수 있습니다"
           - 일반적 평가: "높은 신뢰도를 가집니다", "이론적 예측을 검증해야 합니다"
           - 구체성 없는 방법론 언급: 단순한 "도킹 시뮬레이션", "ADMET 예측" 나열
        3. **독창적 통찰과 구체적 메커니즘만** 제시하세요 - 일반론이나 당연한 내용은 완전히 배제
        4. 각 분석의 고유한 통찰을 통합하되, **전문가 Agent 이름은 절대 언급하지 마세요** (단, 문헌 인용이나 실험 데이터 출처는 허용)
           - 금지: "구조화학 전문가에 따르면", "QSAR 전문가 분석", "생체분자 상호작용 전문가의 결과" 등

        **작성 형식:**
        ## 최종 가설 제안
        **주요 베이스: 채택된 핵심 분석**
        
        **1. 구조적 차이점 분석**
        [채택된 주요 가설의 구조 분석을 핵심 베이스로 하되, 다른 가설들의 우수한 보완 관점들을 체계적으로 통합하여 완성도 높은 종합 분석을 제시]
        
        **2. 작용 기전 가설**
        [생물학적 메커니즘에 대한 구체적이고 근거 있는 설명]
        
        **3. 실험적 근거 및 검증**
        [기존 실험 데이터나 문헌에서 이 가설을 직접적으로 뒷받침하는 구체적 증거만 제시 - 뻔한 검증 방법론 언급 금지]
        
        **4. 분자 설계 제안**
        [혁신적이고 구체적인 구조 변경 전략만 제시 - 일반적인 최적화 방향 설명 금지]
        [제안 화합물들의 완성된 SMILES 코드 3-5개 포함 - 각각 명확한 설계 근거 제시]
        
        **5. 핵심 통찰**
        [이 분석에서만 얻을 수 있는 독창적 통찰과 발견사항 - 일반적인 신뢰도/한계점 평가 금지]
        
        ### 추가 고려 가설
        
        **대안적 접근법:**
        [보조 가설 1의 독창적 관점과 핵심 통찰을 구체적으로 2-3문장 요약 - 주요 가설과 차별화되는 접근 방식과 근거 포함]
        
        **보완적 관점:**
        [보조 가설 2의 추가적 시각과 보완 요소를 구체적으로 2-3문장 요약 - 전체적 분석의 완성도를 높이는 요소들 포함]
        
        **중요 금지사항**: 
        - 당연하고 뻔한 내용 금지 ("실험적 검증이 필요", "추가 연구가 필요", "한계가 있을 수 있습니다")
        - 일반적인 신뢰도 평가 금지 ("높은 신뢰도를 가집니다", "이론적 예측을 검증")
        - 구체적 제안 없는 방법론 언급 금지 ("도킹 시뮬레이션", "ADMET 예측" 등을 단순 나열)
        **필수 요구사항**: 혁신적 통찰, 구체적 메커니즘, 독창적 발견사항, 구체적 실험 제안만 포함하세요.
        
        **SMILES 코드 요구사항**: 분자 설계 제안에서 반드시 다음을 포함하세요:
        - 제안 화합물 1: [설명] - SMILES: [완성된 SMILES 코드]
        - 제안 화합물 2: [설명] - SMILES: [완성된 SMILES 코드]  
        - 제안 화합물 3: [설명] - SMILES: [완성된 SMILES 코드]
        (필요시 최대 5개까지)
        """
        
        final_hypothesis = self.llm_client.generate_response(
            system_prompt="당신은 Activity Cliff 구조 활성 차이 원인 분석 종합 전문가입니다. 여러 전문가의 의견을 통합하여 최고 품질의 종합 리포트를 작성합니다.",
            user_prompt=synthesis_prompt,
            temperature=0.2
        )
        
        return {
            'final_hypothesis': final_hypothesis,
            'individual_evaluations': individual_evaluations,
            'best_evaluation': best_evaluation,
            'synthesis_metadata': {
                'best_hypothesis_agent': best_evaluation['agent_name'],
                'best_hypothesis_strengths': len(best_evaluation['strengths']),
                'best_hypothesis_score': best_evaluation['overall_score'],
                'total_strengths_considered': len(all_strengths),
                'total_insights_integrated': len(all_insights),
                'synthesis_timestamp': time.time(),
                'selection_criteria': 'strengths_count_first',
                'utilization_based_synthesis': True  # 활용 방식 기반 통합 표시
            }
        }
    
    def _build_individual_evaluation_prompt(self, hypothesis: Dict, shared_context: Dict) -> str:
        """개별 가설 분석용 프롬프트 구성"""
        return f"""
        **가설 분석 요청:**
        
        **전문가:** {hypothesis['agent_name']}
        **가설 내용:**
        {hypothesis['hypothesis']}
        
        **분석 요청:**
        이 가설의 강점, 약점, 핵심 인사이트를 객관적으로 분석해주세요:
        
        **분석 형식:**
        강점: [신뢰할 수 있고 가치 있는 부분들 2-3개]
        약점: [개선이 필요한 부분들 1-2개]
        핵심 인사이트: [이 가설에서 얻을 수 있는 중요한 통찰 1-2개]
        
        객관적이고 구체적인 분석을 부탁드립니다.
        """
    
    def _generate_comprehensive_report(self, individual_evaluations: List[Dict], shared_context: Dict) -> Dict:
        """모든 가설의 강점을 통합하여 최종 종합 리포트 생성"""
        
        # 모든 가설의 강점과 인사이트 수집
        all_strengths = []
        all_insights = []
        all_hypotheses_text = []
        
        for eval_result in individual_evaluations:
            all_strengths.extend(eval_result.get('strengths', []))
            all_insights.extend(eval_result.get('key_insights', []))
            all_hypotheses_text.append(f"**{eval_result['agent_name']}**: {eval_result['original_hypothesis']['hypothesis']}")
        
        # 가장 우수한 가설 선정 (강점이 가장 많은 것)
        best_evaluation = max(individual_evaluations, key=lambda x: len(x.get('strengths', [])))
        remaining_evaluations = [eval_result for eval_result in individual_evaluations if eval_result != best_evaluation]
        
        # 종합 리포트 생성 프롬프트
        synthesis_prompt = f"""
        **최종 종합 리포트 작성 요청:**
        
        다음은 분석을 통해 도출된 가설들입니다:
        1. **주요 가설 (채택됨)**: {best_evaluation['original_hypothesis']['hypothesis'][:500]}...
        2. **보조 가설 1**: {remaining_evaluations[0]['original_hypothesis']['hypothesis'][:200]}...
        3. **보조 가설 2**: {remaining_evaluations[1]['original_hypothesis']['hypothesis'][:200]}...
        
        **주요 가설의 핵심 강점:**
        {chr(10).join([f"• {strength}" for strength in best_evaluation.get('strengths', [])[:4]])}
        
        **다른 가설들의 보완 강점:**
        {chr(10).join([f"• {strength}" for strength in all_strengths[:6] if strength not in best_evaluation.get('strengths', [])])}
        
        **작성 지침:**
        1. **주요 가설을 베이스로 사용**하되, 다른 가설의 우수한 부분으로 보완하세요
        2. **구체적이고 혁신적인 내용만** 포함하세요. 다음과 같은 진부한 내용은 **절대 포함 금지**:
           - 뻔한 결론: "실험적 검증이 필요", "추가 연구가 필요", "한계가 있을 수 있습니다"
           - 일반적 평가: "높은 신뢰도를 가집니다", "이론적 예측을 검증해야 합니다"
           - 구체성 없는 방법론 언급: 단순한 "도킹 시뮬레이션", "ADMET 예측" 나열
        3. **독창적 통찰과 구체적 메커니즘만** 제시하세요 - 일반론이나 당연한 내용은 완전히 배제
        4. 각 분석의 고유한 통찰을 통합하되, **전문가 Agent 이름은 절대 언급하지 마세요** (단, 문헌 인용이나 실험 데이터 출처는 허용)
           - 금지: "구조화학 전문가에 따르면", "QSAR 전문가 분석", "생체분자 상호작용 전문가의 결과" 등
        
        **작성 형식:**
        ## 최종 가설 제안
        **주요 베이스: 채택된 핵심 분석**
        
        **1. 구조적 차이점 분석**
        [채택된 주요 가설의 구조 분석을 핵심 베이스로 하되, 다른 가설들의 우수한 보완 관점들을 체계적으로 통합하여 완성도 높은 종합 분석을 제시]
        
        **2. 작용 기전 가설**
        [생물학적 메커니즘에 대한 구체적이고 근거 있는 설명]
        
        **3. 실험적 근거 및 검증**
        [기존 실험 데이터나 문헌에서 이 가설을 직접적으로 뒷받침하는 구체적 증거만 제시 - 뻔한 검증 방법론 언급 금지]
        
        **4. 분자 설계 제안**
        [혁신적이고 구체적인 구조 변경 전략만 제시 - 일반적인 최적화 방향 설명 금지]
        [제안 화합물들의 완성된 SMILES 코드 3-5개 포함 - 각각 명확한 설계 근거 제시]
        
        **5. 핵심 통찰**
        [이 분석에서만 얻을 수 있는 독창적 통찰과 발견사항 - 일반적인 신뢰도/한계점 평가 금지]
        
        ### 추가 고려 가설
        
        **대안적 접근법:**
        [보조 가설 1의 독창적 관점과 핵심 통찰을 구체적으로 2-3문장 요약 - 주요 가설과 차별화되는 접근 방식과 근거 포함]
        
        **보완적 관점:**
        [보조 가설 2의 추가적 시각과 보완 요소를 구체적으로 2-3문장 요약 - 전체적 분석의 완성도를 높이는 요소들 포함]
        
        **중요 금지사항**: 
        - 당연하고 뻔한 내용 금지 ("실험적 검증이 필요", "추가 연구가 필요", "한계가 있을 수 있습니다")
        - 일반적인 신뢰도 평가 금지 ("높은 신뢰도를 가집니다", "이론적 예측을 검증")
        - 구체적 제안 없는 방법론 언급 금지 ("도킹 시뮬레이션", "ADMET 예측" 등을 단순 나열)
        **필수 요구사항**: 혁신적 통찰, 구체적 메커니즘, 독창적 발견사항, 구체적 실험 제안만 포함하세요.
        
        **SMILES 코드 요구사항**: 분자 설계 제안에서 반드시 다음을 포함하세요:
        - 제안 화합물 1: [설명] - SMILES: [완성된 SMILES 코드]
        - 제안 화합물 2: [설명] - SMILES: [완성된 SMILES 코드]  
        - 제안 화합물 3: [설명] - SMILES: [완성된 SMILES 코드]
        (필요시 최대 5개까지)
        """
        
        final_hypothesis = self.llm_client.generate_response(
            system_prompt="당신은 SAR 분석 종합 전문가입니다. 여러 전문가의 의견을 통합하되 전문가 이름은 언급하지 말고, 최고 품질의 통합 종합 리포트를 작성합니다.",
            user_prompt=synthesis_prompt,
            temperature=0.2
        )
        
        return {
            'final_hypothesis': final_hypothesis,
            'individual_evaluations': individual_evaluations,
            'synthesis_metadata': {
                'total_strengths_considered': len(all_strengths),
                'total_insights_integrated': len(all_insights),
                'synthesis_timestamp': time.time()
            }
        }
    
    def _extract_strengths(self, evaluation_text: str) -> List[str]:
        """평가 텍스트에서 강점 추출"""
        strengths = []
        lines = evaluation_text.split('\n')
        
        in_strengths_section = False
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['강점', 'strength', '장점']):
                in_strengths_section = True
                continue
            elif in_strengths_section and line:
                if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                    strength = line.lstrip('•-*123. ').strip()
                    if strength and len(strength) > 5:
                        strengths.append(strength)
                elif not any(keyword in line.lower() for keyword in ['약점', '단점', 'weakness', '개선']):
                    if len(line) > 5:
                        strengths.append(line)
                else:
                    break
        
        return strengths[:3]  # 최대 3개
    
    def _extract_weaknesses(self, evaluation_text: str) -> List[str]:
        """평가 텍스트에서 약점 추출"""
        weaknesses = []
        lines = evaluation_text.split('\n')
        
        in_weaknesses_section = False
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['약점', 'weakness', '단점', '개선']):
                in_weaknesses_section = True
                continue
            elif in_weaknesses_section and line:
                if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                    weakness = line.lstrip('•-*123. ').strip()
                    if weakness and len(weakness) > 5:
                        weaknesses.append(weakness)
                elif not any(keyword in line.lower() for keyword in ['강점', '장점', 'strength']):
                    if len(line) > 5:
                        weaknesses.append(line)
                else:
                    break
        
        return weaknesses[:2]  # 최대 2개
    
    def _extract_key_insights(self, evaluation_text: str) -> List[str]:
        """평가 텍스트에서 핵심 인사이트 추출"""
        insights = []
        lines = evaluation_text.split('\n')
        
        in_insights_section = False
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['핵심 인사이트', 'key insight', '중요한 통찰', '인사이트']):
                in_insights_section = True
                continue
            elif in_insights_section and line:
                if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                    insight = line.lstrip('•-*123. ').strip()
                    if insight and len(insight) > 10:
                        insights.append(insight)
                elif not any(keyword in line.lower() for keyword in ['강점', '약점', '개선', 'strength', 'weakness']):
                    if len(line) > 10:
                        insights.append(line)
                else:
                    break
        
        return insights[:2]  # 최대 2개


# 시각적 표시 함수들
def display_expert_result(result: Dict):
    """각 전문가 결과 표시"""
    with st.expander(f"{result['agent_name']} 결과", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("**생성된 가설:**")
            st.write(result['hypothesis'][:300] + "..." if len(result['hypothesis']) > 300 else result['hypothesis'])
            
        with col2:
            st.write("**핵심 인사이트:**")
            for insight in result['key_insights'][:3]:
                st.write(f"• {insight}")


# 메인 온라인 토론 시스템 함수
def run_online_discussion_system(selected_cliff: Dict, target_name: str, api_key: str, llm_provider: str = "OpenAI") -> Dict:
    """단순화된 Co-Scientist 방법론 기반 가설 생성 시스템"""
    
    start_time = time.time()
    
    # 통합 LLM 클라이언트 생성
    llm_client = UnifiedLLMClient(api_key, llm_provider)
    
    # st.markdown("**Co-Scientist 방법론 기반 SAR 분석**")
    st.markdown(f"3명의 전문가 Agent가 독립적으로 분석한 후 평가를 통해 최고 품질의 가설을 생성합니다.")
    
    # Phase 1: 데이터 준비 + RAG 통합
    st.info("**Phase 1: 데이터 준비** - RAG 통합 컨텍스트 구성")
    shared_context = prepare_shared_context(selected_cliff, target_name)
    
    # 컨텍스트 정보 표시
    with st.expander("분석 대상 정보", expanded=False):
        cliff_summary = shared_context['cliff_summary']
        st.write(f"**고활성 화합물:** {cliff_summary['high_activity_compound']['id']} (pIC50: {cliff_summary['high_activity_compound']['pic50']})")
        st.write(f"**저활성 화합물:** {cliff_summary['low_activity_compound']['id']} (pIC50: {cliff_summary['low_activity_compound']['pic50']})")
        st.write(f"**활성도 차이:** {cliff_summary['cliff_metrics']['activity_difference']}")
    
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
    
    # Phase 3: 종합 평가 및 최종 리포트 생성
    st.markdown("---")
    # st.info("**Phase 3: 종합 평가** - 평가 전문 Agent가 모든 가설의 장점을 통합하여 최종 리포트를 생성합니다")
    
    # 평가 전문가 에이전트 초기화 (기존 클래스 재사용)
    evaluator = HypothesisEvaluationExpert(llm_client)
    
    # 새로운 종합 평가 방식 사용
    evaluation_report = evaluator.evaluate_hypotheses(domain_hypotheses, shared_context)
    
    # 최종 리포트 생성
    final_report = {
        'final_hypothesis': evaluation_report.get('final_hypothesis', ''),
        'individual_evaluations': evaluation_report.get('individual_evaluations', []),
        'domain_hypotheses': domain_hypotheses,  # 도킹 결과가 포함된 가설들 추가
        'synthesis_metadata': evaluation_report.get('synthesis_metadata', {}),
        'process_metadata': {
            'total_time': time.time() - start_time,
            'total_agents': len(domain_hypotheses),
            'analysis_method': 'Co-Scientist 종합 평가 방법론',
            'synthesis_approach': True
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


def display_simplified_results(final_report: Dict):
    """종합 리포트 형식으로 최종 결과 표시"""
    
    # 최종 종합 가설 표시
    final_hypothesis = final_report.get('final_hypothesis', '')
    if final_hypothesis:
        st.markdown(final_hypothesis)
    else:
        st.warning("최종 종합 가설을 생성할 수 없습니다.")
        
        # 대안으로 개별 분석 결과 표시
        individual_evaluations = final_report.get('individual_evaluations', [])
        if individual_evaluations:
            st.markdown("## 📊 개별 전문가 분석 요약")
            
            for eval_result in individual_evaluations:
                with st.expander(f"📝 {eval_result['agent_name']} 상세 분석", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**원본 가설:**")
                        hypothesis_text = eval_result['original_hypothesis'].get('hypothesis', '')
                        if len(hypothesis_text) > 300:
                            st.write(hypothesis_text[:300] + "...")
                        else:
                            st.write(hypothesis_text)
                    
                    with col2:
                        if eval_result.get('strengths'):
                            st.write("**주요 강점:**")
                            for strength in eval_result['strengths'][:2]:
                                st.write(f"• {strength}")
                        
                        if eval_result.get('key_insights'):
                            st.write("**핵심 인사이트:**")
                            for insight in eval_result['key_insights'][:2]:
                                st.write(f"• {insight}")
    
    # 종합 프로세스 메타데이터 표시
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metadata = final_report.get('process_metadata', {})
    synthesis_metadata = final_report.get('synthesis_metadata', {})
    
    with col1:
        st.metric("총 소요시간", f"{metadata.get('total_time', 0):.1f}초")
    with col2:
        st.metric("참여 전문가", f"{metadata.get('total_agents', 0)}명")
    with col3:
        st.metric("통합 강점", f"{synthesis_metadata.get('total_strengths_considered', 0)}개")
    with col4:
        st.metric("통합 인사이트", f"{synthesis_metadata.get('total_insights_integrated', 0)}개")


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
        QSARExpert(llm_client)
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
                'key_insights': ['오류 발생'],
                'reasoning_steps': ['오류로 인한 중단'],
                'timestamp': time.time()
            }
            domain_hypotheses.append(result)
    
    progress_bar.empty()  # Phase 2 진행바 숨기기
    return domain_hypotheses


def display_docking_results(docking_analysis: dict, agent_name: str):
    """도킹 시뮬레이션 결과를 종합적으로 표시"""
    if not docking_analysis:
        return
    
    # 도킹 결과를 토글(expander) 안에 넣기
    with st.expander("도킹 시뮬레이션 분석 결과 (상세 보기)", expanded=False):
        
        # 전체 결과를 한 화면에 표시
        if 'high_active_docking' in docking_analysis and 'low_active_docking' in docking_analysis:
            high_result = docking_analysis['high_active_docking']
            low_result = docking_analysis['low_active_docking']
            
            # 1. 결합 친화도 및 Ki 값 비교 (작은 폰트로)
            st.markdown("**1) 결합 친화도 분석**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write("고활성 화합물")
                st.write(f"• 결합 친화도: {high_result['binding_affinity']:.1f} kcal/mol")
                st.write(f"• Ki 추정값: {high_result['ki_estimate']:.0f} nM")
            
            with col2:
                st.write("저활성 화합물")  
                st.write(f"• 결합 친화도: {low_result['binding_affinity']:.1f} kcal/mol")
                st.write(f"• Ki 추정값: {low_result['ki_estimate']:.0f} nM")
            
            # 2. 비교 분석 결과
            if 'comparative_analysis' in docking_analysis:
                comp_analysis = docking_analysis['comparative_analysis']
                
                with col3:
                    st.write("친화도 차이")
                    diff_value = comp_analysis['affinity_difference']
                    st.write(f"• 차이: {abs(diff_value):.1f} kcal/mol")
                    st.write(f"• 방향: {'고활성 > 저활성' if diff_value < 0 else '저활성 > 고활성'}")
                
                with col4:
                    st.write("예측 정확도")
                    supports_cliff = comp_analysis.get('supports_activity_cliff', False)
                    activity_ratio = comp_analysis.get('predicted_activity_ratio', 1)
                    st.write(f"• 활성비: {activity_ratio:.1f}배")
                    st.write(f"• 실험 일치: {'예' if supports_cliff else '아니오'}")
            
            # 3. 분자간 상호작용 분석
            st.markdown("**2) 단백질-리간드 상호작용**")
            
            interaction_names = {
                'hydrogen_bonds': '수소결합',
                'hydrophobic': '소수성 상호작용',
                'pi_stacking': 'π-π 적층',
                'electrostatic': '정전기 상호작용'
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("고활성 화합물 상호작용:")
                if high_result.get('interactions'):
                    for interaction_type, residues in high_result['interactions'].items():
                        if residues:
                            interaction_name = interaction_names.get(interaction_type, interaction_type)
                            residue_text = ', '.join(residues[:3])
                            if len(residues) > 3:
                                residue_text += f" 외 {len(residues)-3}개"
                            st.write(f"• {interaction_name}: {residue_text}")
                else:
                    st.write("• 상호작용 데이터 없음")
            
            with col2:
                st.write("저활성 화합물 상호작용:")
                if low_result.get('interactions'):
                    for interaction_type, residues in low_result['interactions'].items():
                        if residues:
                            interaction_name = interaction_names.get(interaction_type, interaction_type)
                            residue_text = ', '.join(residues[:3])
                            if len(residues) > 3:
                                residue_text += f" 외 {len(residues)-3}개"
                            st.write(f"• {interaction_name}: {residue_text}")
                else:
                    st.write("• 상호작용 데이터 없음")
            
            # 4. 종합 해석
            st.markdown("**3) 도킹 분석 종합 해석**")
            
            if 'comparative_analysis' in docking_analysis:
                comp_analysis = docking_analysis['comparative_analysis']
                diff_value = comp_analysis['affinity_difference']
                supports_cliff = comp_analysis.get('supports_activity_cliff', False)
                
                if supports_cliff and diff_value < -1.0:
                    interpretation = "도킹 시뮬레이션 결과가 실험적 활성 차이를 잘 설명합니다. 고활성 화합물이 타겟 단백질과 더 강한 결합을 형성하여 높은 생물학적 활성을 보이는 것으로 예측됩니다."
                elif not supports_cliff and abs(diff_value) < 1.0:
                    interpretation = "도킹 결과만으로는 활성 차이를 완전히 설명하기 어렵습니다. 결합 친화도 외에 단백질 동역학, 알로스테릭 효과, 또는 ADMET 특성 차이가 주요 원인일 가능성이 있습니다."
                elif diff_value > 1.0:
                    interpretation = "도킹 결과가 실험 데이터와 상반됩니다. 저활성 화합물이 더 강한 결합을 보이므로, 결합 후 단백질 기능 조절, 대사 안정성, 또는 세포막 투과성 등 다른 요인의 영향이 클 것으로 예상됩니다."
                else:
                    interpretation = "도킹 시뮬레이션 결과 해석이 불명확합니다. 추가적인 분자동역학 시뮬레이션이나 실험적 검증이 필요합니다."
                
                st.write(interpretation)
            
            # 5. 추가 분석 제안
            st.markdown("**4) 후속 분석 제안**")
            suggestions = [
                "분자동역학(MD) 시뮬레이션을 통한 결합 안정성 분석",
                "자유에너지 섭동(FEP) 계산으로 정밀한 결합 친화도 예측",
                "단백질-리간드 복합체의 결합 모드 상세 분석",
                "ADMET 예측을 통한 약동학적 특성 비교"
            ]
            
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"{i}. {suggestion}")
    