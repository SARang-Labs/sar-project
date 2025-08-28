"""
생체분자 상호작용 전문가 에이전트

단백질-리간드 상호작용과 결합 부위 분석 관점에서 Activity Cliff를 분석합니다.
단백질 활성 부위의 아미노산과 리간드 간의 구체적 상호작용을 중심으로 분석합니다.
"""

from .base_agent import BaseAgent
from typing import Dict, Optional

class BiologicalAgent(BaseAgent):
    """
    생체분자 상호작용 전문가 에이전트
    
    분석 초점:
    - 단백질 활성 부위의 아미노산과 리간드 간의 구체적 상호작용
    - 수소결합, 소수성 상호작용, π-π 적층 등의 구체적 메커니즘
    - 단백질 구조 변화나 동적 효과가 결합에 미치는 영향
    - 타겟 특이성과 선택성을 결정하는 구조적 요인
    """
    
    def _get_expertise(self) -> str:
        """전문 분야 반환"""
        return "생체분자 상호작용"
    
    def _get_system_prompt(self) -> str:
        """생체분자 상호작용 전문가용 시스템 프롬프트"""
        return """
        당신은 단백질-리간드 상호작용과 결합 부위 분석 전문가입니다.
        Activity Cliff 분석에서 생체분자 상호작용 관점을 중심으로 분석해주세요.
        
        분석 강조 포인트:
        - 단백질 활성 부위의 아미노산과 리간드 간의 구체적 상호작용
        - 수소결합, 소수성 상호작용, π-π 적층 등의 구체적 메커니즘
        - 단백질 구조 변화나 동적 효과가 결합에 미치는 영향
        - 타겟 특이성과 선택성을 결정하는 구조적 요인
        
        다음 5단계에 따라 체계적으로 분석하세요:
        1. 구조 비교 분석 (생체분자 관점)
        2. 단백질-리간드 상호작용 변화 예측
        3. 타겟 특화 상호작용 가설
        4. 활성 차이 논리적 연결
        5. 생화학적 검증 실험 제안
        
        각 단계에서 단백질-리간드 상호작용 전문가로서의 견해를 제시하고,
        구체적인 생화학적 근거와 실험적 검증 방법을 포함해주세요.
        """
    
    def _build_analysis_prompt(self, activity_cliff: Dict, context_info: Dict = None) -> str:
        """생체분자 상호작용 분석용 프롬프트 구성"""
        mol1_info = activity_cliff['mol_1']
        mol2_info = activity_cliff['mol_2']
        
        # 낮은 활성과 높은 활성 화합물 구분
        low_activity = mol1_info if mol1_info['pKi'] < mol2_info['pKi'] else mol2_info
        high_activity = mol2_info if mol1_info['pKi'] < mol2_info['pKi'] else mol1_info
        
        prompt = f"""
        **분석 대상 Activity Cliff - 생체분자 상호작용 관점 분석**

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
        
        [참고 문헌 정보 - 타겟 특화 정보]
        - 제목: {context_info.get('title', 'N/A')}
        - 초록: {context_info.get('abstract', 'N/A')}
        
        위 문헌 정보를 바탕으로 타겟 단백질의 특성을 고려한 분석을 수행해주세요.
        """
        
        prompt += """

        다음 5단계에 따라 생체분자 상호작용 관점에서 체계적으로 분석해주세요:

        ## 1단계: 구조 비교 분석 (생체분자 관점)
        두 화합물의 구조적 차이가 단백질 결합에 미칠 영향을 분석하세요:
        - 리간드 결합에 중요한 작용기의 변화
        - 수소결합 공여/수용체 그룹의 위치 변화
        - 소수성 패치(hydrophobic patch)의 크기와 위치 변화
        - 전하 분포의 변화 (양이온성/음이온성 그룹)
        - 방향족 고리의 π-시스템 변화

        ## 2단계: 단백질-리간드 상호작용 변화 예측
        구조 변화가 단백질과의 상호작용에 미치는 영향을 예측하세요:
        - 수소결합 네트워크의 변화 (형성/파괴되는 수소결합)
        - 소수성 상호작용의 변화 (van der Waals 접촉면 변화)
        - π-π 적층, π-cation, cation-π 상호작용의 변화
        - 염교 결합(salt bridge) 형성 가능성
        - 금속 배위 결합의 변화 (해당하는 경우)

        ## 3단계: 타겟 특화 상호작용 가설
        참고 문헌 정보를 바탕으로 타겟 단백질 특성을 고려한 분석을 수행하세요:
        - 타겟 단백질의 활성 부위 특성 (알려진 중요 잔기)
        - 리간드 결합 포켓의 크기와 모양 제약
        - 타겟별 특이적 상호작용 패턴
        - 결합 부위의 유연성과 유도 적합 효과
        - 알로스테릭 효과나 단백질 구조 변화 가능성

        ## 4단계: 활성 차이 논리적 연결
        상호작용 변화를 활성도 차이와 연결하여 설명하세요:
        - 각 상호작용 변화가 결합 친화도에 미치는 기여도
        - 엔탈피(결합 강도) vs 엔트로피(구조 유연성) 효과
        - 결합 동역학(kinetics)에 미치는 영향 (kon/koff)
        - 단백질-리간드 복합체의 안정성 변화
        - 경쟁적 저해제와의 상대적 친화도 변화

        ## 5단계: 생화학적 검증 실험 제안
        가설을 검증할 수 있는 구체적 생화학 실험을 제안하세요:
        - 결합 친화도 측정 (SPR, ITC, 형광 편광 등)
        - 결합 동역학 분석 (association/dissociation rate)
        - 구조 생물학 연구 (X-ray, cryo-EM, NMR)
        - 돌연변이 연구 (key 아미노산 변이 효과)
        - 선택성 패널 (관련 타겟들과의 교차 반응성)
        - 열역학 분석 (ITC를 통한 ΔH, ΔS 측정)

        각 단계에서 단백질-리간드 상호작용 전문가로서의 견해를 제시하고,
        구체적인 생화학적 메커니즘과 실험적 근거를 포함해주세요.
        특히 참고 문헌의 타겟 관련 정보를 적극 활용하여 분석하세요.
        """
        
        return prompt