"""
SAR 분석 전문가 에이전트 모듈

이 모듈은 SAR(Structure-Activity Relationship) 분석을 위한 다양한 전문가 에이전트들을 정의합니다.
Co-Scientist 방법론을 기반으로 한 전문가 협업 시스템의 핵심 구성요소입니다.

포함된 전문가 에이전트:
- StructuralChemistryExpert: 구조화학 전문가
- BiomolecularInteractionExpert: 생체분자 상호작용 전문가
- QSARExpert: 정량적 구조-활성 관계 전문가
- HypothesisEvaluationExpert: 가설 평가 전문가

주요 기능:
- LLM 기반 전문가 의견 생성
- 도킹 시뮬레이션 결과 해석
- 할루시네이션 방지 안전장치
- 실시간 사용자 피드백 표시
"""

# === 표준 라이브러리 및 외부 패키지 ===
import json
import time
from typing import Dict, List, Any
import streamlit as st

# === 프로젝트 내부 모듈 ===
from .llm_client import UnifiedLLMClient


# === 할루시네이션 방지 안전 가이드라인 ===
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
    """
    구조화학 전문가 에이전트

    20년 경력의 선임 약화학자 관점에서 분자 구조와 전자적 특성 변화를 분석합니다.
    분자 기하학, SMILES 구조 차이점, 화학적 직관과 구조-기능 관계 규명을 전문으로 합니다.

    주요 기능:
    - Activity Cliff 쌍의 구조적 차이점 분석
    - 분자 기하학적 변화가 활성에 미치는 영향 평가
    - 도킹 시뮬레이션 결과와 구조 분석 통합
    - 화학적 직관 기반 메커니즘 추론

    Attributes:
        llm_client (UnifiedLLMClient): LLM 클라이언트 인스턴스
        persona (str): 전문가 역할 정의 프롬프트
    """

    def __init__(self, llm_client: UnifiedLLMClient):
        """
        구조화학 전문가 초기화

        Args:
            llm_client (UnifiedLLMClient): LLM 응답 생성을 위한 클라이언트
        """
        self.llm_client = llm_client
        self.persona = """당신은 20년 경력의 선임 약화학자로, 분자 구조와 전자적 특성 변화 분석의 전문가입니다.
        특히 분자 기하학, SMILES 구조 차이점, 화학적 직관과 구조-기능 관계 규명에 특화되어 있습니다.

**중요한 응답 규칙:**
1. 인사말, 감사 인사, "기꺼이 도와드리겠습니다" 등의 사족은 절대 포함하지 마세요.
2. 바로 분석 내용으로 시작하세요.
3. 일관된 '-입니다' 체로 작성하되, 가설이므로 확정적 단언은 지양하세요."""

    def generate(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        구조화학 관점의 가설 생성

        Activity Cliff 쌍에 대해 구조화학적 관점에서 활성도 차이를 설명하는
        가설을 생성합니다. 분자 구조의 차이점과 그것이 생물학적 활성에
        미치는 영향을 분석합니다.

        Args:
            shared_context (Dict[str, Any]): 공유 컨텍스트 정보
                - cliff_summary: Activity Cliff 쌍 정보
                - target_name: 타겟 단백질명
                - literature_context: 도킹 시뮬레이션 결과 (선택적)

        Returns:
            Dict[str, Any]: 전문가 분석 결과
                - agent_type: 전문가 유형
                - agent_name: 전문가명
                - hypothesis: 생성된 가설
                - key_insights: 핵심 인사이트
                - reasoning_steps: 추론 단계
                - timestamp: 생성 시간
        """
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

        # 세포주 정보 안전하게 처리
        cell_line_info = ""
        if shared_context.get('cell_line_context'):
            cell_line_name = shared_context['cell_line_context'].get('cell_line_name', 'Unknown')
            cell_line_info = f"- 측정 세포주: {cell_line_name}"
        
        literature_info = ""
        if shared_context.get('literature_context'):
            lit = shared_context['literature_context']
            literature_info = f"""
            **도킹 시뮬레이션 결과 (구조 기반 근거):**
            - 화합물 1 결합 친화도: {lit.get('compound1', {}).get('binding_affinity_kcal_mol', 'N/A')} kcal/mol
            - 화합물 2 결합 친화도: {lit.get('compound2', {}).get('binding_affinity_kcal_mol', 'N/A')} kcal/mol
            - 화합물 1 주요 상호작용: {', '.join([k for k, v in lit.get('compound1', {}).get('interaction_fingerprint', {}).items() if v])}
            - 화합물 2 주요 상호작용: {', '.join([k for k, v in lit.get('compound2', {}).get('interaction_fingerprint', {}).items() if v])}
            - 이 도킹 결과를 근거로 구조적 차이가 활성 차이로 이어지는 메커니즘을 분석하세요.
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

        **실험 조건:**
        - 타겟 단백질: {target_name} (PDB ID)
        {cell_line_info}
        
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
    """
    생체분자 상호작용 전문가 에이전트

    단백질-리간드 상호작용 메커니즘 분야의 세계적 권위자 관점에서 분석을 수행합니다.
    타겟 단백질과의 결합 방식 변화, 약리학적 관점과 생리활성 메커니즘 규명을 전문으로 합니다.

    주요 기능:
    - 단백질-리간드 상호작용 패턴 분석
    - 도킹 시뮬레이션 수행 및 결과 해석
    - 결합 친화도 변화 메커니즘 분석
    - 약리학적 관점에서의 활성도 차이 설명

    Attributes:
        llm_client (UnifiedLLMClient): LLM 클라이언트 인스턴스
        persona (str): 전문가 역할 정의 프롬프트
    """

    def __init__(self, llm_client: UnifiedLLMClient):
        """
        생체분자 상호작용 전문가 초기화

        Args:
            llm_client (UnifiedLLMClient): LLM 응답 생성을 위한 클라이언트
        """
        self.llm_client = llm_client
        self.persona = """당신은 단백질-리간드 상호작용 메커니즘 분야의 세계적 권위자입니다.
        타겟 단백질과의 결합 방식 변화, 약리학적 관점과 생리활성 메커니즘 규명을 전문으로 합니다.

**중요한 응답 규칙:**
1. 인사말, 감사 인사, "기꺼이 도와드리겠습니다" 등의 사족은 절대 포함하지 마세요.
2. 바로 분석 내용으로 시작하세요.
3. 일관된 '-입니다' 체로 작성하되, 가설이므로 확정적 단언은 지양하세요."""

    def generate(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        생체분자 상호작용 관점의 가설 생성 (도킹 시뮬레이션 포함)

        Activity Cliff 쌍에 대해 생체분자 상호작용 관점에서 활성도 차이를 설명하는
        가설을 생성합니다. 도킹 시뮬레이션을 수행하여 결합 방식의 차이를 분석합니다.

        Args:
            shared_context (Dict[str, Any]): 공유 컨텍스트 정보
                - cliff_summary: Activity Cliff 쌍 정보
                - target_name: 타겟 단백질명

        Returns:
            Dict[str, Any]: 전문가 분석 결과
                - agent_type: 전문가 유형
                - agent_name: 전문가명
                - hypothesis: 생성된 가설
                - key_insights: 핵심 인사이트
                - reasoning_steps: 추론 단계
                - docking_analysis: 도킹 분석 결과
                - timestamp: 생성 시간
        """
        
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
        
        if high_active.get('smiles') and low_active.get('smiles'):
            # utils.py의 get_docking_context 함수 사용
            from utils import get_docking_context
            docking_results = get_docking_context(
                high_active['smiles'],
                low_active['smiles'],
                target_name
            )
            
            # compound1과 compound2가 None이 아닌지 확인
            if (docking_results and
                docking_results.get('compound1') is not None and
                docking_results.get('compound2') is not None):

                # 결과를 기존 형식으로 변환
                results = {
                    'high_active_docking': {
                        'binding_affinity': docking_results['compound1']['binding_affinity_kcal_mol'],
                        'interactions': docking_results['compound1']['interaction_fingerprint']
                    },
                    'low_active_docking': {
                        'binding_affinity': docking_results['compound2']['binding_affinity_kcal_mol'],
                        'interactions': docking_results['compound2']['interaction_fingerprint']
                    }
                }

                # 비교 분석
                high_score = results['high_active_docking']['binding_affinity']
                low_score = results['low_active_docking']['binding_affinity']
                results['comparative_analysis'] = {
                    'affinity_difference': high_score - low_score,
                    'supports_activity_cliff': high_score < low_score  # 낮은 값이 더 강한 결합
                }

                return results
            else:
                # 도킹 결과가 없는 경우
                return None
        
        return {}
    
    def _build_interaction_prompt(self, shared_context: Dict[str, Any], docking_results: Dict = None) -> str:
        """생체분자 상호작용 전문가용 특화 프롬프트 생성 - CoT.md 지침 반영"""
        cliff_summary = shared_context['cliff_summary']
        target_name = shared_context['target_name']
        high_active = cliff_summary['high_activity_compound']
        low_active = cliff_summary['low_activity_compound']
        metrics = cliff_summary['cliff_metrics']
        prop_diffs = cliff_summary['property_differences']

        # 세포주 정보 안전하게 처리
        cell_line_info = ""
        if shared_context.get('cell_line_context'):
            cell_line_name = shared_context['cell_line_context'].get('cell_line_name', 'Unknown')
            cell_line_info = f"- 측정 세포주: {cell_line_name}"
        
        literature_info = ""
        if shared_context.get('literature_context'):
            lit = shared_context['literature_context']
            literature_info = f"""
            **도킹 시뮬레이션 결과 (구조 기반 근거):**
            - 화합물 1 결합 친화도: {lit.get('compound1', {}).get('binding_affinity_kcal_mol', 'N/A')} kcal/mol
            - 화합물 2 결합 친화도: {lit.get('compound2', {}).get('binding_affinity_kcal_mol', 'N/A')} kcal/mol
            - 화합물 1 주요 상호작용: {', '.join([k for k, v in lit.get('compound1', {}).get('interaction_fingerprint', {}).items() if v])}
            - 화합물 2 주요 상호작용: {', '.join([k for k, v in lit.get('compound2', {}).get('interaction_fingerprint', {}).items() if v])}
            - 이 도킹 결과를 근거로 구조적 차이가 활성 차이로 이어지는 메커니즘을 분석하세요.
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

        **실험 조건:**
        - 타겟 단백질: {target_name} (PDB ID)
        {cell_line_info}
        
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
            formatted += f"- 고활성 화합물: 결합 친화도 {high['binding_affinity']:.1f} kcal/mol\n"
        
        if 'low_active_docking' in docking_results:
            low = docking_results['low_active_docking']
            formatted += f"- 저활성 화합물: 결합 친화도 {low['binding_affinity']:.1f} kcal/mol\n"
        
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
    """
    정량적 구조-활성 관계(QSAR) 전문가 에이전트

    구조-활성 관계(SAR) 분석의 세계적 전문가 관점에서 분석을 수행합니다.
    제공된 실제 데이터를 바탕으로 분자 특성 변화와 활성 차이 간의 정성적 관계를 분석하고,
    문헌 근거와 화학적 지식을 토대로 구조-활성 관계 패턴을 해석합니다.

    주요 기능:
    - 분자 물리화학적 특성과 활성도 관계 분석
    - QSAR 패턴 해석 (정성적 접근)
    - 분자 디스크립터 변화와 활성 차이 연관성 분석
    - 구조-활성 관계 패턴 예측

    중요 제약사항:
    - 실제 계산하지 않은 QSAR 모델 생성 금지
    - 회귀방정식, 통계값(R², RMSE 등) 생성 금지
    - 정성적 분석 중심의 해석 제공

    Attributes:
        llm_client (UnifiedLLMClient): LLM 클라이언트 인스턴스
        persona (str): 전문가 역할 정의 프롬프트
    """

    def __init__(self, llm_client: UnifiedLLMClient):
        """
        QSAR 전문가 초기화

        Args:
            llm_client (UnifiedLLMClient): LLM 응답 생성을 위한 클라이언트
        """
        self.llm_client = llm_client
        self.persona = """당신은 구조-활성 관계(SAR) 분석의 세계적 전문가입니다.
        제공된 실제 데이터(SMILES, pIC50, 물리화학적 특성)를 바탕으로 분자 특성 변화와 활성 차이 간의
        정성적 관계를 분석하고, 문헌 근거와 화학적 지식을 토대로 구조-활성 관계 패턴을 해석하는 것이 전문입니다.

        **중요**: 실제 계산하지 않은 QSAR 모델, 회귀방정식, 통계값(R², RMSE 등)을 절대 생성하지 마세요.

**중요한 응답 규칙:**
1. 인사말, 감사 인사, "기꺼이 도와드리겠습니다" 등의 사족은 절대 포함하지 마세요.
2. 바로 분석 내용으로 시작하세요.
3. 일관된 '-입니다' 체로 작성하되, 가설이므로 확정적 단언은 지양하세요."""

    def generate(self, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        QSAR 관점의 가설 생성

        Activity Cliff 쌍에 대해 정량적 구조-활성 관계 관점에서 활성도 차이를
        설명하는 가설을 생성합니다. 분자 물리화학적 특성의 변화와 활성도 차이의
        정성적 관계를 분석합니다.

        Args:
            shared_context (Dict[str, Any]): 공유 컨텍스트 정보
                - cliff_summary: Activity Cliff 쌍 정보
                - target_name: 타겟 단백질명

        Returns:
            Dict[str, Any]: 전문가 분석 결과
                - agent_type: 전문가 유형
                - agent_name: 전문가명
                - hypothesis: 생성된 가설
                - key_insights: 핵심 인사이트
                - reasoning_steps: 추론 단계
                - timestamp: 생성 시간
        """
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

        # 세포주 정보 안전하게 처리
        cell_line_info = ""
        if shared_context.get('cell_line_context'):
            cell_line_name = shared_context['cell_line_context'].get('cell_line_name', 'Unknown')
            cell_line_info = f"- 측정 세포주: {cell_line_name}"
        
        literature_info = ""
        if shared_context.get('literature_context'):
            lit = shared_context['literature_context']
            literature_info = f"""
            **도킹 시뮬레이션 결과 (구조 기반 근거):**
            - 화합물 1 결합 친화도: {lit.get('compound1', {}).get('binding_affinity_kcal_mol', 'N/A')} kcal/mol
            - 화합물 2 결합 친화도: {lit.get('compound2', {}).get('binding_affinity_kcal_mol', 'N/A')} kcal/mol
            - 화합물 1 주요 상호작용: {', '.join([k for k, v in lit.get('compound1', {}).get('interaction_fingerprint', {}).items() if v])}
            - 화합물 2 주요 상호작용: {', '.join([k for k, v in lit.get('compound2', {}).get('interaction_fingerprint', {}).items() if v])}
            - 이 도킹 결과를 근거로 구조적 차이가 활성 차이로 이어지는 메커니즘을 분석하세요.
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

        **실험 조건:**
        - 타겟 단백질: {target_name} (PDB ID)
        {cell_line_info}
        
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
    """
    가설 평가 전문가 에이전트

    15년 경력의 SAR 분석 평가 전문가 관점에서 가설을 평가합니다.
    Activity Cliff 구조 활성 차이 원인 분석, 가설 검증, 과학적 엄밀성 평가에 특화되어 있으며,
    실제 데이터와 문헌 근거를 바탕으로 객관적이고 일관된 평가를 수행합니다.

    주요 기능:
    - 전문가 가설의 과학적 타당성 평가
    - 실험 데이터와 가설의 일관성 검증
    - 도킹 시뮬레이션 결과와 가설의 부합성 분석
    - 종합적인 신뢰도 점수 산정

    평가 기준:
    - 과학적 근거의 충실성
    - 실험 데이터와의 일관성
    - 화학적 논리의 타당성
    - 도킹 결과와의 부합성

    Attributes:
        llm_client (UnifiedLLMClient): LLM 클라이언트 인스턴스
        persona (str): 전문가 역할 정의 프롬프트
    """

    def __init__(self, llm_client: UnifiedLLMClient):
        """
        가설 평가 전문가 초기화

        Args:
            llm_client (UnifiedLLMClient): LLM 응답 생성을 위한 클라이언트
        """
        self.llm_client = llm_client
        self.persona = """당신은 15년 경력의 SAR 분석 평가 전문가입니다.
        Activity Cliff 구조 활성 차이 원인 분석, 가설 검증, 과학적 엄밀성 평가에 특화되어 있으며,
        실제 데이터와 문헌 근거를 바탕으로 객관적이고 일관된 평가를 수행합니다.

**중요한 응답 규칙:**
1. 인사말, 감사 인사, "기꺼이 도와드리겠습니다" 등의 사족은 절대 포함하지 마세요.
2. 바로 분석 내용으로 시작하세요.
3. 일관된 '-입니다' 체로 작성하되, 가설이므로 확정적 단언은 지양하세요."""

    def evaluate(self, hypothesis: Dict, shared_context: Dict) -> Dict:
        """
        맥락 기반 가설 품질 평가

        각 전문가들이 생성한 가설을 종합적으로 평가하여 과학적 타당성,
        실험 데이터와의 일관성, 화학적 논리성을 검증합니다.

        Args:
            hypothesis (Dict): 평가할 가설 정보
                - hypothesis: 가설 내용
                - agent_type: 전문가 유형
                - key_insights: 핵심 인사이트
            shared_context (Dict): 공유 컨텍스트 정보
                - cliff_summary: Activity Cliff 쌍 정보
                - literature_context: 도킹 시뮬레이션 결과
                - target_name: 타겟 단백질명

        Returns:
            Dict: 평가 결과
                - credibility_score: 신뢰도 점수 (0-10)
                - evaluation_summary: 평가 요약
                - strengths: 강점 목록
                - weaknesses: 약점 목록
                - consistency_with_data: 데이터 일관성
        """
        
        # shared_context에서 핵심 정보 추출
        cliff_summary = shared_context.get('cliff_summary', {})
        literature_context = shared_context.get('literature_context', {})
        target_name = shared_context.get('target_name', '알 수 없음')
        cell_line_context = shared_context.get('cell_line_context', {})
        
        # Activity Cliff 정보 구성
        cliff_info = ""
        if cliff_summary:
            high_comp = cliff_summary.get('high_activity_compound', {})
            low_comp = cliff_summary.get('low_activity_compound', {})
            metrics = cliff_summary.get('cliff_metrics', {})
            
            cell_line_info = ""
            if cell_line_context and cell_line_context.get('cell_line_name'):
                cell_line_info = f"\n    - 측정 세포주: {cell_line_context.get('cell_line_name')}"
            
            cliff_info = f"""
    **Activity Cliff 분석 대상:**
    - 타겟 단백질: {target_name} (PDB ID){cell_line_info}
    - 고활성 화합물: {high_comp.get('id', 'N/A')} (pIC50: {high_comp.get('pic50', 'N/A')})
    - 저활성 화합물: {low_comp.get('id', 'N/A')} (pIC50: {low_comp.get('pic50', 'N/A')})
    - 활성도 차이: {metrics.get('activity_difference', 'N/A')}
    - 구조 유사도: {metrics.get('similarity', 'N/A')}"""
        
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
    3. **타겟 특이성 (20%)**: {target_name} 타겟 단백질 특이적 언급{f", {cell_line_context.get('cell_line_name')} 세포주 특성 고려" if cell_line_context.get('cell_line_name') else ""}, kinase domain/binding site 등 전문 용어 사용, 타겟의 알려진 특성 반영 정도 평가
    
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
        - 문헌 근거: {(shared_context.get('literature_context') or {}).get('title', '관련 문헌 없음')[:100] if shared_context.get('literature_context') else '도킹 데이터 활용'}

        **통합 원칙:**
        1. **주요 가설을 핵심 베이스**로 사용하되, 다른 가설들의 우수한 요소들을 체계적으로 통합
           - 각 전문가가 제시한 구체적 분석 내용과 독창적 통찰을 반드시 활용하세요
           - 전문가들의 상세한 메커니즘 분석과 구체적 근거를 일반론으로 축약하지 마세요
        2. **구체적이고 혁신적인 내용만** 포함하세요. 다음과 같은 진부한 내용은 **절대 포함 금지**:
           - 뻔한 결론: "실험적 검증이 필요", "추가 연구가 필요", "한계가 있을 수 있습니다"
           - 일반적 평가: "높은 신뢰도를 가집니다", "이론적 예측을 검증해야 합니다"
           - 구체성 없는 방법론 언급: "AutoDock4를 통한 도킹 실험", "ADMET 예측", "분자 동역학 시뮬레이션" 등 단순 나열
           - 당연한 도킹 설명: "도킹 시뮬레이션을 통해 확인할 수 있습니다", "결합 친화도를 예측할 수 있습니다"
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
            system_prompt="""당신은 Activity Cliff 구조 활성 차이 원인 분석 종합 전문가입니다. 여러 전문가의 의견을 통합하여 최고 품질의 종합 리포트를 작성합니다.

**중요한 응답 규칙:**
1. 인사말, 감사 인사, "기꺼이 도와드리겠습니다" 등의 사족은 절대 포함하지 마세요.
2. 바로 분석 내용으로 시작하세요.
3. 일관된 '-입니다' 체로 작성하되, 가설이므로 확정적 단언은 지양하세요.""",
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
        
        # shared_context에서 실험 조건 정보 추출
        target_name = shared_context.get('target_name', '알 수 없음')
        cell_line_context = shared_context.get('cell_line_context', {})
        cell_line_name = cell_line_context.get('cell_line_name', '알 수 없음')
        
        # 종합 리포트 생성 프롬프트
        synthesis_prompt = f"""
        **최종 종합 리포트 작성 요청:**
        
        **실험 조건:**
        - 타겟 단백질: {target_name} (PDB ID)
        - 측정 세포주: {cell_line_name}
        
        다음은 분석을 통해 도출된 가설들입니다:
        1. **주요 가설 (채택됨)**: {best_evaluation['original_hypothesis']['hypothesis'][:500]}...
        2. **보조 가설 1**: {remaining_evaluations[0]['original_hypothesis']['hypothesis'][:200]}...
        3. **보조 가설 2**: {remaining_evaluations[1]['original_hypothesis']['hypothesis'][:200]}...
        
        **주요 가설의 핵심 강점:**
        {chr(10).join([f"• {strength}" for strength in best_evaluation.get('strengths', [])[:4]])}
        
        **다른 가설들의 보완 강점:**
        {chr(10).join([f"• {strength}" for strength in all_strengths[:6] if strength not in best_evaluation.get('strengths', [])])}
        
        **작성 지침:**
        1. **실험 조건 고려**: {target_name} 단백질과 {cell_line_name} 세포주의 특성을 반영하여 가설을 작성하세요
        2. **주요 가설을 베이스로 사용**하되, 다른 가설의 우수한 부분으로 보완하세요
        3. **구체적이고 혁신적인 내용만** 포함하세요. 다음과 같은 진부한 내용은 **절대 포함 금지**:
           - 뻔한 결론: "실험적 검증이 필요", "추가 연구가 필요", "한계가 있을 수 있습니다"
           - 일반적 평가: "높은 신뢰도를 가집니다", "이론적 예측을 검증해야 합니다"
           - 구체성 없는 방법론 언급: 단순한 "도킹 시뮬레이션", "ADMET 예측" 나열
        4. **독창적 통찰과 구체적 메커니즘만** 제시하세요 - 일반론이나 당연한 내용은 완전히 배제
        5. 각 분석의 고유한 통찰을 통합하되, **전문가 Agent 이름은 절대 언급하지 마세요** (단, 문헌 인용이나 실험 데이터 출처는 허용)
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
            system_prompt="""당신은 SAR 분석 종합 전문가입니다. 여러 전문가의 의견을 통합하되 전문가 이름은 언급하지 말고, 최고 품질의 통합 종합 리포트를 작성합니다.

**중요한 응답 규칙:**
1. 인사말, 감사 인사, "기꺼이 도와드리겠습니다" 등의 사족은 절대 포함하지 마세요.
2. 바로 분석 내용으로 시작하세요.
3. 일관된 '-입니다' 체로 작성하되, 가설이므로 확정적 단언은 지양하세요.""",
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


