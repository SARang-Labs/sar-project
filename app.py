import streamlit as st
import pandas as pd
from rdkit import Chem
from utils import (
    load_data, find_activity_cliffs, generate_hypothesis, draw_highlighted_pair
)
import plotly.express as px
import os

# Co-Scientist 온라인 토론 시스템 임포트 
try:
    from online_discussion_system import run_online_discussion_system
    ONLINE_DISCUSSION_AVAILABLE = True
    print("Co-Scientist 온라인 토론 시스템 로드 성공")
except ImportError as e:
    ONLINE_DISCUSSION_AVAILABLE = False
    print(f"Co-Scientist 온라인 토론 시스템 로드 실패: {str(e)}")

# 최적 프롬프트 토론 시스템 임포트
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from llm_debate.debate.optimal_prompt_debate_manager import OptimalPromptDebateManager
    from streamlit_components.optimal_prompt_debate_interface import OptimalPromptDebateInterface
    PROMPT_SYSTEM_AVAILABLE = True
    print("최적 프롬프트 토론 시스템 로드 성공")
except ImportError as e:
    PROMPT_SYSTEM_AVAILABLE = False
    print(f"최적 프롬프트 토론 시스템 로드 실패: {str(e)}")

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 기반 SAR 분석 시스템",
    page_icon="🧪",
    layout="wide",
)

# --- 사이드바 UI ---
with st.sidebar:
    st.title("AI SAR 분석 시스템")
    st.info("신약 개발을 위한 AI 기반 구조-활성 관계 분석 및 예측 솔루션입니다.")
    
    st.header("📁 데이터 입력")
    uploaded_file = st.file_uploader("SAR 분석용 CSV 파일을 업로드하세요.", type="csv")
    use_sample_data = st.checkbox("샘플 데이터 사용", value=True)

    st.header("⚙️ AI 모델 설정")
    target_name = st.text_input("분석 대상 타겟 단백질 (예: EGFR)", value="EGFR")
    
    llm_provider = st.selectbox("LLM 공급자 선택:", ("OpenAI", "Google Gemini"))
    
    api_key_placeholder = "OpenAI API 키 (sk-...)" if llm_provider == "OpenAI" else "Gemini API 키"
    api_key = st.text_input("API 키 입력:", type="password", placeholder=api_key_placeholder)
    
    # 세션 상태에 저장 (다른 탭에서 사용하기 위해)
    st.session_state['llm_provider'] = llm_provider
    st.session_state['api_key'] = api_key

st.header("분석 결과 대시보드")

# --- 데이터 로딩 ---
df = None
if use_sample_data:
    # 로컬 경로에 맞게 수정
    sample_path = 'data/large_sar_data.csv'
    if os.path.exists(sample_path):
        df = load_data(sample_path)
    else:
        st.sidebar.error(f"샘플 데이터 파일 '{sample_path}'를 찾을 수 없습니다.")
elif uploaded_file:
    df = load_data(uploaded_file)

# --- 탭 구성 ---
if df is not None:
    # 탭 구성: Co-Scientist SAR 분석, 기존 SAR 분석, 최적 프롬프트 토론
    tabs = []
    
    if ONLINE_DISCUSSION_AVAILABLE:
        tabs.append("SAR 분석 (토론 시스템 적용)")
    
    tabs.append("SAR 분석 (기존)")
    
    if PROMPT_SYSTEM_AVAILABLE:
        tabs.append("최적 프롬프트 토론")
    
    # 탭 생성
    if len(tabs) == 1:
        tab_basic, = st.tabs(tabs)
        tab_advanced, tab_prompt = None, None
    elif len(tabs) == 2:
        if ONLINE_DISCUSSION_AVAILABLE:
            tab_advanced, tab_basic = st.tabs(tabs)
            tab_prompt = None
        else:
            tab_basic, tab_prompt = st.tabs(tabs)
            tab_advanced = None
    else:
        tab_advanced, tab_basic, tab_prompt = st.tabs(tabs)

    # ==================== 고급 SAR 분석 (온라인 토론) 탭 ====================
    if ONLINE_DISCUSSION_AVAILABLE and tab_advanced is not None:
        with tab_advanced:
            st.markdown("### Activity Cliff 자동 분석")
            st.info("기존 SAR 분석 시스템의 가설 생성 부분에 \"5단계 온라인 토론 시스템\"을 적용합니다.")

            # --- SAR 탭 데이터 시각화 (기존과 동일) ---
            with st.expander("현재 데이터 활성도 분포 보기"):
                plot_df = df.copy()
                if 'pKi' in plot_df.columns:
                    plot_df['pKi'] = pd.to_numeric(plot_df['pKi'], errors='coerce')
                    plot_df.dropna(subset=['pKi'], inplace=True)
                    
                    if not plot_df.empty:
                        display_df = plot_df.drop(columns=['mol', 'fp', 'scaffold'], errors='ignore')
                        st.dataframe(display_df.head())
                        fig = px.histogram(plot_df, x='pKi', title='활성도(pKi) 분포', labels={'pKi': 'pKi 값'})
                        st.plotly_chart(fig, use_container_width=True, key='advanced_histogram')
                    else:
                        st.warning("'pKi' 컬럼에 유효한 숫자 데이터가 없어 분포를 표시할 수 없습니다.")
                else:
                    st.warning("'pKi' 컬럼을 찾을 수 없어 분포를 표시할 수 없습니다.")

            col1, col2 = st.columns(2)
            with col1:
                similarity_threshold_adv = st.slider("유사도 임계값 (Tanimoto)", 0.5, 1.0, 0.8, 0.01, key='sim_adv')
            with col2:
                activity_diff_threshold_adv = st.slider("활성도 차이 임계값 (ΔpKi)", 0.5, 5.0, 1.0, 0.1, key='act_adv')

            if st.button("Activity Cliff 분석 시작", key='advanced_analyze'):
                with st.spinner("Activity Cliff 분석 중..."):
                    cliffs = find_activity_cliffs(df, similarity_threshold_adv, activity_diff_threshold_adv)
                    st.session_state['cliffs_advanced'] = cliffs

            if 'cliffs_advanced' in st.session_state:
                cliffs = st.session_state['cliffs_advanced']
                if not cliffs:
                    st.warning("설정된 조건에 맞는 Activity Cliff를 찾을 수 없습니다.")
                else:
                    st.success(f"총 {len(cliffs)}개의 Activity Cliff를 찾았습니다. 분석할 쌍을 선택하세요.")
                    
                    cliff_options = [f"{i+1}. {c['mol_1']['ID']} vs {c['mol_2']['ID']} (ΔpKi: {c['activity_diff']:.2f})" for i, c in enumerate(cliffs)]
                    selected_option = st.selectbox("분석할 Activity Cliff 선택:", cliff_options, key='cliff_select_adv')
                    
                    if selected_option:
                        selected_index = cliff_options.index(selected_option)
                        selected_cliff = cliffs[selected_index]
                        st.session_state['selected_cliff_index_advanced'] = selected_index

                        mol1_info = selected_cliff['mol_1']
                        mol2_info = selected_cliff['mol_2']

                        st.markdown("---")
                        st.markdown(f"#### 선택된 Cliff: **{mol1_info['ID']}** vs **{mol2_info['ID']}**")
                        
                        # Activity Cliff 정보 표시 (기존과 동일)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Tanimoto 유사도", f"{selected_cliff['similarity']:.3f}")
                        c2.metric("pKi 차이 (ΔpKi)", f"{selected_cliff['activity_diff']:.3f}")
                        
                        if 'structural_difference' in selected_cliff:
                            c3.metric("구조적 차이", selected_cliff['structural_difference'])
                        if 'is_stereoisomer' in selected_cliff:
                            c4.metric("입체이성질체", "예" if selected_cliff['is_stereoisomer'] else "아니오")
                        
                        # 물리화학적 특성 비교 (기존과 동일)
                        if 'mol1_properties' in selected_cliff and 'mol2_properties' in selected_cliff:
                            with st.expander("물리화학적 특성 비교"):
                                prop1 = selected_cliff['mol1_properties']
                                prop2 = selected_cliff['mol2_properties']
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"{mol1_info['ID']} 분자량", f"{prop1.get('molecular_weight', 0):.1f} Da")
                                    st.metric(f"{mol1_info['ID']} LogP", f"{prop1.get('logp', 0):.2f}")
                                with col2:
                                    st.metric(f"{mol2_info['ID']} 분자량", f"{prop2.get('molecular_weight', 0):.1f} Da")
                                    st.metric(f"{mol2_info['ID']} LogP", f"{prop2.get('logp', 0):.2f}")
                                with col3:
                                    mw_diff = abs(prop1.get('molecular_weight', 0) - prop2.get('molecular_weight', 0))
                                    logp_diff = abs(prop1.get('logp', 0) - prop2.get('logp', 0))
                                    st.metric("분자량 차이", f"{mw_diff:.1f} Da")
                                    st.metric("LogP 차이", f"{logp_diff:.2f}")
                        
                        # 분자 구조 시각화 (기존과 동일)
                        svg1, svg2 = draw_highlighted_pair(mol1_info['SMILES'], mol2_info['SMILES'])

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**화합물 1: {mol1_info['ID']}** (pKi: {mol1_info['pKi']:.2f})")
                            if svg1:
                                st.image(svg1, use_container_width=True)
                            else:
                                st.warning("이미지를 생성할 수 없습니다.")
                        
                        with col2:
                            st.markdown(f"**화합물 2: {mol2_info['ID']}** (pKi: {mol2_info['pKi']:.2f})")
                            if svg2:
                                st.image(svg2, use_container_width=True)
                            else:
                                st.warning("이미지를 생성할 수 없습니다.")
                        
                        # ===== 여기가 핵심 차이점: 온라인 토론 시스템 사용 =====
                        st.markdown("---")
                        st.markdown("### 가설 리포트 생성")
                        
                        # API 키 확인
                        if not api_key:
                            st.warning(f"⚠️ 사이드바에서 {llm_provider} API 키를 입력해주세요.")
                        else:
                            # 온라인 토론 시작 버튼
                            if st.button("5단계 온라인 토론 시작", type="primary", key="start_online_discussion_adv"):
                                try:
                                    # 비동기 함수 실행 (사이드바 LLM 설정 전달)
                                    final_report = run_online_discussion_system(selected_cliff, target_name, api_key, llm_provider)
                                    st.session_state['discussion_report_advanced'] = final_report
                                except Exception as e:
                                    st.error(f"❌ 온라인 토론 중 오류가 발생했습니다: {str(e)}")
                                    st.exception(e)
                            
                            # 이전 토론 결과가 있는 경우 표시
                            if 'discussion_report_advanced' in st.session_state:
                                st.markdown("### 최종 가설 제안")
                                st.success("**결론:** Co-Scientist 방법론을 통해 3명의 전문가가 각자의 관점에서 체계적으로 분석한 결과입니다. 상위 가설들을 종합적으로 고려하여 후속 연구를 진행하시기 바랍니다.")

                                report = st.session_state['discussion_report_advanced']
                                
                                if report.get('ranked_hypotheses'):
                                    for i, hypothesis in enumerate(report['ranked_hypotheses'][:3]):
                                        rank_emoji = ["🥇", "🥈", "🥉"][i]
                                        score = hypothesis.get('overall_score', hypothesis.get('final_score', 75))
                                        with st.expander(f"{rank_emoji} 가설 {i+1} (점수: {score:.0f}/100)", expanded=i==0):
                                            hypothesis_text = (
                                                hypothesis.get('hypothesis', '') or 
                                                hypothesis.get('final_hypothesis', '') or
                                                '가설 내용을 찾을 수 없습니다.'
                                            )
                                            st.write(hypothesis_text)
                                            if hypothesis.get('evolution_applied'):
                                                st.success("✨ Self-Play 논쟁으로 개선됨")
                                st.markdown("---")

    # ==================== 기존 SAR 분석 탭 ====================  
    if tab_basic is not None:
        with tab_basic:
            st.subheader("Activity Cliff 자동 분석 리포트")

            # --- SAR 탭 데이터 시각화 ---
            with st.expander("현재 데이터 활성도 분포 보기"):
                plot_df = df.copy()
                if 'pKi' in plot_df.columns:
                    plot_df['pKi'] = pd.to_numeric(plot_df['pKi'], errors='coerce')
                    plot_df.dropna(subset=['pKi'], inplace=True)
                    
                    if not plot_df.empty:
                        display_df = plot_df.drop(columns=['mol', 'fp', 'scaffold'], errors='ignore')
                        st.dataframe(display_df.head()) # RDKit 객체가 제거된 display_df를 표시
                        fig = px.histogram(plot_df, x='pKi', title='활성도(pKi) 분포', labels={'pKi': 'pKi 값'})
                        st.plotly_chart(fig, use_container_width=True, key='basic_histogram')
                    else:
                        st.warning("'pKi' 컬럼에 유효한 숫자 데이터가 없어 분포를 표시할 수 없습니다.")
                else:
                    st.warning("'pKi' 컬럼을 찾을 수 없어 분포를 표시할 수 없습니다.")

            col1, col2 = st.columns(2)
            with col1:
                similarity_threshold = st.slider("유사도 임계값 (Tanimoto)", 0.5, 1.0, 0.8, 0.01, key='sim_basic')
            with col2:
                activity_diff_threshold = st.slider("활성도 차이 임계값 (ΔpKi)", 0.5, 5.0, 1.0, 0.1, key='act_basic')

            if st.button("Activity Cliff 분석 시작", key='sar_analyze'):
                with st.spinner("Activity Cliff 분석 중..."):
                    cliffs = find_activity_cliffs(df, similarity_threshold, activity_diff_threshold)
                    st.session_state['cliffs'] = cliffs

            if 'cliffs' in st.session_state:
                cliffs = st.session_state['cliffs']
                if not cliffs:
                    st.warning("설정된 조건에 맞는 Activity Cliff를 찾을 수 없습니다.")
                else:
                    st.success(f"총 {len(cliffs)}개의 Activity Cliff를 찾았습니다. 분석할 쌍을 선택하세요.")
                    
                    cliff_options = [f"{i+1}. {c['mol_1']['ID']} vs {c['mol_2']['ID']} (ΔpKi: {c['activity_diff']:.2f})" for i, c in enumerate(cliffs)]
                    selected_option = st.selectbox("분석할 Activity Cliff 선택:", cliff_options, key='cliff_select')
                    
                    if selected_option:
                        selected_index = cliff_options.index(selected_option)
                        selected_cliff = cliffs[selected_index]
                        st.session_state['selected_cliff_index'] = selected_index  # 다른 탭에서 사용하기 위해 저장

                    mol1_info = selected_cliff['mol_1']
                    mol2_info = selected_cliff['mol_2']

                    st.markdown("---")
                    st.markdown(f"#### 선택된 Cliff: **{mol1_info['ID']}** vs **{mol2_info['ID']}**")
                    
                    # 개선된 Activity Cliff 정보 표시
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Tanimoto 유사도", f"{selected_cliff['similarity']:.3f}")
                    c2.metric("pKi 차이 (ΔpKi)", f"{selected_cliff['activity_diff']:.3f}")
                    
                    # 추가 정보 표시 (가능한 경우)
                    if 'structural_difference' in selected_cliff:
                        c3.metric("구조적 차이", selected_cliff['structural_difference'])
                    if 'is_stereoisomer' in selected_cliff:
                        c4.metric("입체이성질체", "예" if selected_cliff['is_stereoisomer'] else "아니오")
                    
                    # 물리화학적 특성 비교 (가능한 경우)
                    if 'mol1_properties' in selected_cliff and 'mol2_properties' in selected_cliff:
                        with st.expander("물리화학적 특성 비교"):
                            prop1 = selected_cliff['mol1_properties']
                            prop2 = selected_cliff['mol2_properties']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{mol1_info['ID']} 분자량", f"{prop1.get('molecular_weight', 0):.1f} Da")
                                st.metric(f"{mol1_info['ID']} LogP", f"{prop1.get('logp', 0):.2f}")
                            with col2:
                                st.metric(f"{mol2_info['ID']} 분자량", f"{prop2.get('molecular_weight', 0):.1f} Da")
                                st.metric(f"{mol2_info['ID']} LogP", f"{prop2.get('logp', 0):.2f}")
                            with col3:
                                mw_diff = abs(prop1.get('molecular_weight', 0) - prop2.get('molecular_weight', 0))
                                logp_diff = abs(prop1.get('logp', 0) - prop2.get('logp', 0))
                                st.metric("분자량 차이", f"{mw_diff:.1f} Da")
                                st.metric("LogP 차이", f"{logp_diff:.2f}")
                    
                    svg1, svg2 = draw_highlighted_pair(mol1_info['SMILES'], mol2_info['SMILES'])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**화합물 1: {mol1_info['ID']}** (pKi: {mol1_info['pKi']:.2f})")
                        if svg1:
                            st.image(svg1, use_container_width=True)
                        else:
                            st.warning("이미지를 생성할 수 없습니다.")
                    
                    with col2:
                        st.markdown(f"**화합물 2: {mol2_info['ID']}** (pKi: {mol2_info['pKi']:.2f})")
                        if svg2:
                            st.image(svg2, use_container_width=True)
                        else:
                            st.warning("이미지를 생성할 수 없습니다.")
                    
                    # 기존 AI 가설 생성
                    with st.spinner("AI가 참고 문헌을 검색하고 가설을 생성 중입니다..."):
                        hypothesis, source_info = generate_hypothesis(selected_cliff, target_name, api_key, llm_provider)
                    
                    st.markdown("---")
                    st.markdown("#### 기본 AI 가설")
                    st.markdown(hypothesis)

                    if source_info:
                        with st.expander("참고 문헌 정보 (RAG 근거)"):
                            st.markdown(f"**- 제목:** {source_info['title']}")
                            st.markdown(f"**- 링크:** [PubMed]({source_info['link']})")
                            st.markdown(f"**- 초록:** {source_info['abstract']}")
                    
                    # 고급 분석 시스템 안내
                    st.markdown("---")
                    st.markdown("#### 시스템 안내")
                    
                    if ONLINE_DISCUSSION_AVAILABLE:
                        st.info("**온라인 토론 시스템**: 5단계 Co-Scientist 방법론으로 3명의 전문가 Agent가 토론하여 최고 품질의 가설을 생성합니다.")
                    
                    if PROMPT_SYSTEM_AVAILABLE:
                        st.info("**최적 프롬프트 토론**: 전문가 AI가 토론을 통해 최적의 프롬프트와 가설을 생성합니다.")

    # ==================== 최적 프롬프트 토론 시스템 탭 ====================
    if PROMPT_SYSTEM_AVAILABLE and tab_prompt is not None:
        with tab_prompt:
            st.markdown("# 최적 프롬프트 토론 시스템")
            st.markdown("**토론 주제**: 자동화된 지능형 SAR 분석 시스템을 위한 최적 근거 중심 가설 생성 방법론 확립")
            
            
            # Activity Cliff가 선택된 경우에만 토론 가능
            if 'cliffs' in st.session_state and st.session_state.get('cliffs'):
                selected_cliff_index = st.session_state.get('selected_cliff_index', 0)
                if selected_cliff_index < len(st.session_state['cliffs']):
                    selected_cliff = st.session_state['cliffs'][selected_cliff_index]
                    
                    # 최적 프롬프트 토론 인터페이스 표시
                    optimal_interface = OptimalPromptDebateInterface()
                    optimal_interface.show_interface(
                        activity_cliff=selected_cliff,
                        target_name=target_name
                    )
                else:
                    st.warning("먼저 Activity Cliff를 선택해주세요.")
            else:
                st.info("**시작 방법:**")
                st.markdown("""
                1. **첫 번째 탭**에서 Activity Cliff를 선택하세요
                2. **이 탭**에서 3명의 전문가 Agent가 토론을 통해 최적의 프롬프트를 생성합니다
                3. **결과**로 최고 품질의 프롬프트와 가설을 얻게 됩니다
                
                **토론 과정:**
                - **1단계**: 각 전문가가 독립적으로 프롬프트 생성 → 가설 생성
                - **2단계**: 3번의 토론 라운드 (직접 인용 기반 투명한 평가)
                - **3단계**: 토론 결과 종합하여 최종 최적 프롬프트 1개 생성
                """)
