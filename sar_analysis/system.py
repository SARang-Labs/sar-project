"""
SAR 분석 시스템 메인 로직

이 모듈은 Co-Scientist 방법론을 사용한 전문가 협업 시스템의 핵심 로직을 구현합니다.
Activity Cliff 쌍에 대해 여러 전문가 에이전트가 협업하여 구조-활성 관계 가설을 생성하고 평가합니다.

주요 구성요소:
- 온라인 토론 시스템: run_online_discussion_system()
- 결과 표시 함수: display_simplified_results(), display_docking_results()
- 공유 컨텍스트 준비: prepare_shared_context()
- 전문가 생성 단계: generation_phase()

Co-Scientist 워크플로우:
1. 공유 컨텍스트 준비 (실험 데이터, 도킹 결과)
2. 다학제 전문가 가설 생성 (구조화학, 생체분자상호작용, QSAR)
3. 가설 평가 및 종합
4. 결과 시각화 및 표시
"""

# === 표준 라이브러리 및 외부 패키지 ===
import sys
import os
import time
from typing import Dict, List, Any
import streamlit as st

# === 프로젝트 내부 모듈 ===
# utils에서 필요한 함수들 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_activity_cliff_summary

from .llm_client import UnifiedLLMClient
from .experts import (
    StructuralChemistryExpert,
    BiomolecularInteractionExpert,
    QSARExpert,
    HypothesisEvaluationExpert
)

# === 시각적 표시 함수들 ===
def display_expert_result(result: Dict):
    """
    각 전문가 결과 표시

    개별 전문가 에이전트의 분석 결과를 Streamlit UI에 표시합니다.
    생성된 가설과 핵심 인사이트를 확장 가능한 형태로 제공합니다.

    Args:
        result (Dict): 전문가 분석 결과
            - agent_name: 전문가명
            - hypothesis: 생성된 가설
            - key_insights: 핵심 인사이트 목록
    """
    with st.expander(f"{result['agent_name']} 결과", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("**생성된 가설:**")
            st.write(result['hypothesis'][:300] + "..." if len(result['hypothesis']) > 300 else result['hypothesis'])
            
        with col2:
            st.write("**핵심 인사이트:**")
            for insight in result['key_insights'][:3]:
                st.write(f"• {insight}")


# === 메인 온라인 토론 시스템 함수 ===
def run_online_discussion_system(selected_cliff: Dict, target_name: str, api_key: str, llm_provider: str = "OpenAI", cell_line: str = None) -> Dict:
    """
    Co-Scientist 방법론 기반 SAR 분석 시스템

    Activity Cliff 쌍에 대해 다학제 전문가 에이전트들이 협업하여
    구조-활성 관계 가설을 생성하고 평가하는 메인 시스템입니다.

    워크플로우:
    1. 공유 컨텍스트 준비 (실험 데이터, 도킹 시뮬레이션)
    2. 전문가 가설 생성 (구조화학, 생체분자상호작용, QSAR)
    3. 가설 평가 및 품질 검증
    4. 최종 결과 종합 및 반환

    Args:
        selected_cliff (Dict): Activity Cliff 쌍 데이터
        target_name (str): 타겟 단백질명
        api_key (str): LLM API 키
        llm_provider (str): LLM 공급자 ("OpenAI" 또는 "Gemini")
        cell_line (str, optional): 세포주 정보

    Returns:
        Dict: 최종 분석 결과
            - best_hypothesis: 최고 품질 가설
            - all_hypotheses: 모든 전문가 가설
            - evaluations: 가설 평가 결과
            - shared_context: 공유 컨텍스트
            - processing_time: 처리 시간
    """
    
    start_time = time.time()
    
    # 통합 LLM 클라이언트 생성
    llm_client = UnifiedLLMClient(api_key, llm_provider)
    
    # st.markdown("**Co-Scientist 방법론 기반 SAR 분석**")
    st.markdown(f"3명의 전문가 Agent가 독립적으로 분석한 후 평가를 통해 최고 품질의 가설을 생성합니다.")
    
    # Phase 1: 데이터 준비 + 도킹 시뮬레이션 통합
    st.info("**Phase 1: 데이터 준비** - 도킹 시뮬레이션 컨텍스트 구성")
    shared_context = prepare_shared_context(selected_cliff, target_name, cell_line)
    
    # 컨텍스트 정보 표시
    with st.expander("분석 대상 정보", expanded=False):
        cliff_summary = shared_context['cliff_summary']
        st.write(f"**고활성 화합물:** {cliff_summary['high_activity_compound']['id']} (pIC50: {cliff_summary['high_activity_compound']['pic50']})")
        st.code(cliff_summary['high_activity_compound']['smiles'], language=None)
        st.write(f"**저활성 화합물:** {cliff_summary['low_activity_compound']['id']} (pIC50: {cliff_summary['low_activity_compound']['pic50']})")
        st.code(cliff_summary['low_activity_compound']['smiles'], language=None)
        st.write(f"**활성도 차이:** {cliff_summary['cliff_metrics']['activity_difference']}")
    
    # 도킹 시뮬레이션 결과 표시
    cliff_summary = shared_context.get('cliff_summary', {})
    if cliff_summary:
        high_compound = cliff_summary.get('high_activity_compound', {})
        low_compound = cliff_summary.get('low_activity_compound', {})
        target_name = shared_context.get('target_name', 'EGFR')
        
        # 도킹 결과 생성 (get_docking_context 함수 사용)
        from utils import get_docking_context
        docking_results = get_docking_context(high_compound.get('smiles'), low_compound.get('smiles'), target_name)
        
        with st.expander("도킹 시뮬레이션 결과", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**화합물 1 (낮은 활성, ID: {low_compound.get('id', 'N/A')})**")
                docking1 = docking_results['compound2']
                st.markdown(f"- **결합 친화도:** {docking1['binding_affinity_kcal_mol']} kcal/mol")
                st.markdown(f"- **수소결합:** {', '.join(docking1['interaction_fingerprint']['Hydrogenbonds']) if docking1['interaction_fingerprint']['Hydrogenbonds'] else '없음'}")
                st.markdown(f"- **소수성 상호작용:** {', '.join(docking1['interaction_fingerprint']['Hydrophobic']) if docking1['interaction_fingerprint']['Hydrophobic'] else '없음'}")
                st.markdown(f"- **할로겐결합:** {', '.join(docking1['interaction_fingerprint']['Halogenbonds']) if docking1['interaction_fingerprint']['Halogenbonds'] else '없음'}")
            
            with col2:
                st.markdown(f"**화합물 2 (높은 활성, ID: {high_compound.get('id', 'N/A')})**")
                docking2 = docking_results['compound1']
                st.markdown(f"- **결합 친화도:** {docking2['binding_affinity_kcal_mol']} kcal/mol")
                st.markdown(f"- **수소결합:** {', '.join(docking2['interaction_fingerprint']['Hydrogenbonds']) if docking2['interaction_fingerprint']['Hydrogenbonds'] else '없음'}")
                st.markdown(f"- **소수성 상호작용:** {', '.join(docking2['interaction_fingerprint']['Hydrophobic']) if docking2['interaction_fingerprint']['Hydrophobic'] else '없음'}")
                st.markdown(f"- **할로겐결합:** {', '.join(docking2['interaction_fingerprint']['Halogenbonds']) if docking2['interaction_fingerprint']['Halogenbonds'] else '없음'}")
    
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
    """
    종합 리포트 형식으로 최종 결과 표시

    Co-Scientist 시스템의 전체 분석 결과를 사용자 친화적인 형태로
    Streamlit UI에 표시합니다. 최종 종합 가설, 개별 전문가 평가,
    도킹 결과 등을 체계적으로 제공합니다.

    Args:
        final_report (Dict): 최종 분석 결과
            - final_hypothesis: 최종 종합 가설
            - individual_evaluations: 개별 전문가 평가
            - domain_hypotheses: 전문가별 가설
            - process_metadata: 처리 메타데이터
    """
    
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


def prepare_shared_context(selected_cliff: Dict, target_name: str, cell_line: str = None) -> Dict:
    """
    공유 컨텍스트 준비

    전문가 에이전트들이 공통으로 사용할 컨텍스트 정보를 준비합니다.
    Activity Cliff 요약 정보와 도킹 시뮬레이션 결과를 포함합니다.

    Args:
        selected_cliff (Dict): Activity Cliff 쌍 데이터
        target_name (str): 타겟 단백질명
        cell_line (str, optional): 세포주 정보

    Returns:
        Dict: 공유 컨텍스트 정보
            - cliff_summary: Activity Cliff 요약
            - target_name: 타겟 단백질명
            - literature_context: 도킹 결과 (선택적)
    """
    """도킹 시뮬레이션을 활용한 컨텍스트 준비 - 강화된 구조 기반 근거 제공"""
    
    # 도킹 시뮬레이션 결과 가져오기
    from utils import get_docking_context
    docking_context = get_docking_context(
        selected_cliff['mol_1']['SMILES'],
        selected_cliff['mol_2']['SMILES'],
        target_name
    )
    cliff_summary = get_activity_cliff_summary(selected_cliff)
    
    # 도킹 컨텍스트 품질 향상
    if docking_context and isinstance(docking_context, dict):
        # 도킹 정보 강화
        enhanced_docking = docking_context.copy()
        enhanced_docking['context_type'] = 'Docking Simulation Result'
        enhanced_docking['usage_instruction'] = f"이 도킹 결과를 {target_name} 타겟에 대한 Activity Cliff 분석의 구조적 근거로 활용하세요"
        docking_context = enhanced_docking
    
    # 세포주 정보 가져오기 (있는 경우)
    cell_line_context = None
    if cell_line:
        try:
            from utils import get_cell_line_info, get_cell_line_context_prompt
            cell_line_info = get_cell_line_info(cell_line)
            cell_line_context = {
                'cell_line_name': cell_line,
                'cell_line_info': cell_line_info,
                'context_prompt': get_cell_line_context_prompt(cell_line_info, target_name)
            }
        except ImportError:
            # utils에 세포주 함수가 없을 경우 기본 정보만
            cell_line_context = {
                'cell_line_name': cell_line,
                'cell_line_info': {'characteristics': f'Cell line: {cell_line}'},
                'context_prompt': f"**세포주 컨텍스트:** 활성도는 {cell_line} 세포주에서 측정됨"
            }
    
    # 모든 에이전트가 공유할 통합 컨텍스트
    shared_context = {
        'cliff_data': selected_cliff,
        'cliff_summary': cliff_summary,
        'literature_context': docking_context,  # 도킹 시뮬레이션 결과
        'target_name': target_name,             # PDB ID (예: 6G6K)
        'cell_line_context': cell_line_context, # 세포주 정보
        'timestamp': time.time(),
        'context_quality': 'Enhanced' if docking_context else 'Basic',
        'evidence_level': 'Docking-backed' if docking_context else 'Data-only'
    }
    
    return shared_context


def generation_phase(shared_context: Dict, llm_client: UnifiedLLMClient) -> List[Dict]:
    """
    전문가 가설 생성 단계

    3명의 전문가 에이전트(구조화학, 생체분자상호작용, QSAR)가
    독립적으로 가설을 생성하는 단계입니다.

    Args:
        shared_context (Dict): 공유 컨텍스트 정보
        llm_client (UnifiedLLMClient): LLM 클라이언트

    Returns:
        List[Dict]: 전문가별 가설 목록
    """
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
    """
    도킹 시뮬레이션 결과 표시

    생체분자 상호작용 전문가의 도킹 분석 결과를
    시각적으로 표시합니다.

    Args:
        docking_analysis (dict): 도킹 분석 결과
        agent_name (str): 전문가명 (현재 미사용)
    """
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
            
            with col2:
                st.write("저활성 화합물")  
                st.write(f"• 결합 친화도: {low_result['binding_affinity']:.1f} kcal/mol")
            
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
