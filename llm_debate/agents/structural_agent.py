"""
구조화학 전문가 에이전트

분자 구조와 화학적 상호작용 관점에서 Activity Cliff를 분석합니다.
원자 수준의 전자 효과, 공간적 효과, 분자 궤도 이론 등을 중심으로 분석합니다.
"""

from .base_agent import BaseAgent
from typing import Dict, Optional

class StructuralAgent(BaseAgent):
    """
    구조화학 전문가 에이전트
    
    분석 초점:
    - 원자 수준에서의 전자 효과와 공간적 효과 (크기, 모양의 영향)
    - 분자 궤도 이론 관점에서의 결합 특성
    - 분자 형태 변화와 유연성이 활성에 미치는 영향
    - 치환기가 분자 전체에 미치는 전자적, 공간적 효과의 정확한 메커니즘
    """
    
    def _get_expertise(self) -> str:
        """전문 분야 반환"""
        return "구조화학"
    
    def _get_system_prompt(self) -> str:
        """구조화학 전문가용 시스템 프롬프트"""
        return """
        당신은 분자 구조와 화학적 상호작용 전문가입니다.
        Activity Cliff 분석에서 구조적 관점을 중심으로 분석해주세요.
        
        분석 강조 포인트:
        - 원자 수준에서의 전자 효과와 공간적 효과 (크기, 모양의 영향)
        - 분자 궤도 이론 관점에서의 결합 특성
        - 분자 형태 변화와 유연성이 활성에 미치는 영향
        - 치환기가 분자 전체에 미치는 전자적, 공간적 효과의 정확한 메커니즘
        
        다음 5단계에 따라 체계적으로 분석하세요:
        1. 구조 비교 분석
        2. 물리화학적 특성 변화 예측
        3. 전자적/입체적 효과 분석
        4. 활성 차이 논리적 연결
        5. 검증 실험 제안
        
        각 단계에서 분자 구조 관점의 전문적 견해를 제시하고,
        구체적인 화학적 근거와 예상 수치를 포함해주세요.
        """
    
    def _build_analysis_prompt(self, activity_cliff: Dict, context_info: Dict = None) -> str:
        """구조화학 분석용 프롬프트 구성"""
        mol1_info = activity_cliff['mol_1']
        mol2_info = activity_cliff['mol_2']
        
        # 낮은 활성과 높은 활성 화합물 구분
        low_activity = mol1_info if mol1_info['pKi'] < mol2_info['pKi'] else mol2_info
        high_activity = mol2_info if mol1_info['pKi'] < mol2_info['pKi'] else mol1_info
        
        prompt = f"""
        **분석 대상 Activity Cliff - 구조화학 관점 분석**

        [화합물 A (낮은 활성)]
        - ID: {low_activity['ID']}
        - SMILES: {low_activity['SMILES']}
        - 활성도 (pKi): {low_activity['pKi']:.2f}

        [화합물 B (높은 활성)]
        - ID: {high_activity['ID']}
        - SMILES: {high_activity['SMILES']}
        - 활성도 (pKi): {high_activity['pKi']:.2f}

        [Activity Cliff 특성]
        - Tanimoto 유사도: {activity_cliff['similarity']:.3f}
        - 활성도 차이: {activity_cliff['activity_diff']:.2f} (약 {10**activity_cliff['activity_diff']:.1f}배)
        """
        
        # RAG 정보가 있으면 추가
        if context_info:
            prompt += f"""
        
        [참고 문헌 정보]
        - 제목: {context_info.get('title', 'N/A')}
        - 초록: {context_info.get('abstract', 'N/A')}
        """
        
        prompt += """

        다음 5단계에 따라 구조화학 관점에서 체계적으로 분석해주세요:

        ## 1단계: 구조 비교 분석
        SMILES를 기반으로 두 화합물의 구조적 차이점을 정밀 분석하세요:
        - 변경된 원자/그룹의 정확한 위치와 특성
        - 치환기 크기, 전자적 성질의 차이 (예: 전자끌기/전자밀기 효과)
        - 입체화학적 변화 (키랄 중심, E/Z 이성질체 등)
        - 방향족성, 공명 효과의 변화

        ## 2단계: 물리화학적 특성 변화 예측
        구조 변화가 다음 특성에 미치는 영향을 정량적으로 예측하세요:
        - logP 변화: 예상 차이값과 방향 (소수성/친수성 변화)
        - 분자 표면의 극성 영역 크기 변화 (TPSA 예상값)
        - 수소결합 공여/수용 능력 변화 (HBD/HBA 개수)
        - 분자 크기 및 모양 변화 (분자량, 회전 가능한 결합 수)
        - 분자 유연성 변화

        ## 3단계: 전자적/입체적 효과 분석
        치환기 변화가 분자 전체에 미치는 효과를 분석하세요:
        - 유도 효과 (inductive effect): 전자밀도 변화
        - 공명 효과 (resonance effect): π 전자계의 변화
        - 입체 효과 (steric effect): 공간적 충돌이나 적합성
        - 오르토 효과: 인접 치환기 간의 상호작용
        - 분자 내 수소결합이나 비공유 상호작용의 변화

        ## 4단계: 활성 차이 논리적 연결
        위 분석들을 종합하여 {activity_cliff['activity_diff']:.2f} pKi 단위의 활성 차이를 설명하세요:
        - 각 구조적 요인이 활성 증가/감소에 기여하는 정도
        - 주요 원인과 부차적 원인의 구분
        - 상승 효과나 상쇄 효과가 있는 경우 그 분석
        - 구조-활성 관계의 논리적 연결고리

        ## 5단계: 검증 실험 제안
        가설을 뒷받침할 수 있는 구체적 실험을 제안하세요:
        - 분자 도킹: 예상되는 결합 에너지 차이값 (kcal/mol)
        - 물성 예측: logP, TPSA, HBD/HBA 계산 예상값
        - 중간체 화합물 합성 제안 (구조적 변화의 단계적 확인)
        - 분광학적 분석 (NMR, IR 등으로 전자 환경 변화 확인)
        - 계산화학 연구 (DFT 계산, 분자궤도 분석)

        각 단계에서 구체적인 화학적 근거와 예상 수치를 제시하고,
        분자 구조 전문가로서의 전문적 견해를 포함해주세요.
        """
        
        return prompt