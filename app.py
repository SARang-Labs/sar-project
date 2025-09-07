import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import sqlite3 # DB 연동을 위해 추가
import json

# utils.py로부터 모든 필요한 함수를 임포트합니다.
from utils import (
    load_data,
    find_activity_cliffs,
    generate_hypothesis_cliff,
    generate_hypothesis_quantitative,
    draw_highlighted_pair,
    check_stereoisomers,
    calculate_molecular_properties,
    get_structural_difference_keyword,
    save_results_to_db
)

# --- 외부 시스템 임포트 (원본과 동일) ---
try:
    from online_discussion_system import run_online_discussion_system
    ONLINE_DISCUSSION_AVAILABLE = True
    print("✅ Co-Scientist 온라인 토론 시스템 로드 성공")
except ImportError as e:
    ONLINE_DISCUSSION_AVAILABLE = False
    print(f"❌ Co-Scientist 온라인 토론 시스템 로드 실패: {str(e)}")

try:
    from llm_debate.debate.optimal_prompt_debate_manager import OptimalPromptDebateManager
    from streamlit_components.optimal_prompt_debate_interface import OptimalPromptDebateInterface
    PROMPT_SYSTEM_AVAILABLE = True
    print("✅ 최적 프롬프트 토론 시스템 로드 성공")
except ImportError as e:
    PROMPT_SYSTEM_AVAILABLE = False
    print(f"❌ 최적 프롬프트 토론 시스템 로드 실패: {str(e)}")

# --- 페이지 기본 설정 (원본과 동일) ---
st.set_page_config(page_title="AI 기반 SAR 분석 시스템", page_icon="🧪", layout="wide")


# --- 공통 로직 처리 헬퍼 함수 (원본과 동일) ---
def process_and_display_pair(idx, cliff_data, sim_thresh, activity_col, tab_key, target_name, api_key, llm_provider):
    mol1 = pd.Series(cliff_data['mol_1'])
    mol2 = pd.Series(cliff_data['mol_2'])
    similarity = cliff_data['similarity']
    
    header = f"쌍 #{idx+1} (ID: {mol1.get('ID', 'N/A')} vs {mol2.get('ID', 'N/A')}) | 유사도: {similarity:.3f}"
    
    with st.expander(header, expanded=True):
        real_act_diff = cliff_data['activity_diff']
        structural_diff = cliff_data['structural_difference']
        is_stereoisomer = cliff_data['is_stereoisomer']
        mol1_props = cliff_data['mol1_properties']
        mol2_props = cliff_data['mol2_properties']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tanimoto 유사도", f"{similarity:.3f}")
        c2.metric(f"Δ{activity_col}", f"{real_act_diff:.3f}")
        c3.metric("구조적 차이", structural_diff)
        c4.metric("입체이성질체", "예" if is_stereoisomer else "아니오")

        with st.container():
            sub_c1, sub_c2, sub_c3 = st.columns(3)
            with sub_c1:
                st.metric(f"{mol1.get('ID', 'N/A')} 분자량", f"{mol1_props.get('molecular_weight', 0):.1f} Da")
                st.metric(f"{mol1.get('ID', 'N/A')} LogP", f"{mol1_props.get('logp', 0):.2f}")
            with sub_c2:
                st.metric(f"{mol2.get('ID', 'N/A')} 분자량", f"{mol2_props.get('molecular_weight', 0):.1f} Da")
                st.metric(f"{mol2.get('ID', 'N/A')} LogP", f"{mol2_props.get('logp', 0):.2f}")
            with sub_c3:
                mw_diff = abs(mol1_props.get('molecular_weight', 0) - mol2_props.get('molecular_weight', 0))
                logp_diff = abs(mol1_props.get('logp', 0) - mol2_props.get('logp', 0))
                st.metric("분자량 차이", f"{mw_diff:.1f} Da")
                st.metric("LogP 차이", f"{logp_diff:.2f}")
        st.markdown("---")

        svg1, svg2 = draw_highlighted_pair(mol1['SMILES'], mol2['SMILES'])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**화합물 1: {mol1.get('ID', 'N/A')}**")
            metric_label_1 = f"{activity_col} ({mol1.get('Activity', 'N/A')})"
            metric_value_1 = f"{mol1.get(activity_col, 0):.3f}"
            st.metric(label=metric_label_1, value=metric_value_1)
            st.image(svg1, use_container_width=True)
        with c2:
            st.markdown(f"**화합물 2: {mol2.get('ID', 'N/A')}**")
            metric_label_2 = f"{activity_col} ({mol2.get('Activity', 'N/A')})"
            metric_value_2 = f"{mol2.get(activity_col, 0):.3f}"
            st.metric(label=metric_label_2, value=metric_value_2)
            st.image(svg2, use_container_width=True)
        
        st.markdown("---")

        if tab_key.endswith('basic'):
            if st.button("AI 가설 생성", key=f"gen_hyp_{idx}_{tab_key}"):
                if not api_key: st.warning("사이드바에서 API 키를 입력해주세요.")
                else:
                    with st.spinner("AI 가설 생성 중..."):
                        if tab_key.startswith('quantitative'):
                            hypothesis, context = generate_hypothesis_quantitative(mol1, mol2, similarity, target_name, api_key, llm_provider)
                        else: 
                            hypothesis, context = generate_hypothesis_cliff(cliff_data, target_name, api_key, llm_provider, activity_col)
                        st.markdown(hypothesis)
                        if context:
                            with st.expander("참고 문헌 정보 (RAG)"): st.json(context)

        elif tab_key.endswith('advanced'):
        # --- [수정된 부분 시작] ---
         if st.button("온라인 토론 시작 및 결과 저장", key=f"disc_{idx}_{tab_key}"):
            if not api_key: 
                st.warning("사이드바에서 API 키를 입력해주세요.")
            elif not ONLINE_DISCUSSION_AVAILABLE: 
                st.error("온라인 토론 시스템 모듈을 로드할 수 없습니다.")
            else:
                with st.spinner("AI 전문가들이 토론 후 최종 리포트를 작성합니다..."):
                    # 1. 온라인 토론 시스템 실행하여 최종 리포트 받기
                    final_report = run_online_discussion_system(cliff_data, target_name, api_key, llm_provider)
                    
                    st.markdown("### 전문가 토론 최종 리포트")
                    st.json(final_report)

                    # 2. utils의 함수를 호출하여 DB에 최종 리포트 저장
                    # final_report가 dict 형태일 수 있으므로, json.dumps로 텍스트 변환
                    report_text = json.dumps(final_report, indent=2, ensure_ascii=False)
                    
                    saved_id = save_results_to_db(
                        db_path=db_path,
                        cliff_data=cliff_data,
                        hypothesis_text=report_text, # 최종 리포트를 저장
                        llm_provider="Expert Discussion System", # 에이전트 이름 변경
                        context_info=None # 리포트 자체에 포함된 것으로 간주
                    )

                    if saved_id:
                        st.success(f"토론 리포트가 데이터베이스에 성공적으로 저장되었습니다. (Analysis ID: {saved_id})")
                    else:
                        st.error("데이터베이스 저장에 실패했습니다.")


# --- UI 렌더링 함수 (원본과 동일) ---

def render_quantitative_analysis_ui(df, available_activity_cols, tab_key, target_name, api_key, llm_provider):
    st.info("구조적으로 유사하지만 **활성 분류(Activity)가 다른** 화합물 쌍을 탐색합니다.")
    if 'Activity' not in df.columns:
        st.error("오류: 정량 분석을 실행하려면 데이터에 'Activity' 컬럼이 필요합니다.")
        return
    if not available_activity_cols:
        st.error("오류: 분석에 사용할 유효한 활성 컬럼(pKi/pIC50)이 데이터에 없습니다.")
        return
    ref_activity_col = available_activity_cols[0]

    sim_thresh = st.slider("유사도 임계값", 0.5, 1.0, 0.8, 0.01, key=f'sim_quant_{tab_key}')
    if st.button("정량 분석 실행", key=f'run_quant_{tab_key}'):
        with st.spinner("정량 분석 중..."):
            df_quant = df.dropna(subset=['SMILES', 'Activity', ref_activity_col]).copy()
            df_quant['mol'] = df_quant['SMILES'].apply(Chem.MolFromSmiles)
            df_quant.dropna(subset=['mol'], inplace=True)
            df_quant['scaffold'] = df_quant['mol'].apply(lambda m: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None)
            fpgenerator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            df_quant['fp'] = [fpgenerator.GetFingerprint(m) for m in df_quant['mol']]
            df_quant.reset_index(inplace=True, drop=True)
            pairs = []
            for i in range(len(df_quant)):
                for j in range(i + 1, len(df_quant)):
                    sim = DataStructs.TanimotoSimilarity(df_quant.iloc[i]['fp'], df_quant.iloc[j]['fp'])
                    if sim >= sim_thresh and df_quant.iloc[i]['Activity'] != df_quant.iloc[j]['Activity']:
                        pairs.append({'mol1_index': i, 'mol2_index': j, 'similarity': sim})
            activity_map = {'Highly Active': 4, 'Moderately Active': 3, 'Weakly Active': 2, 'Inactive': 1}
            for pair in pairs:
                activity1 = df_quant.iloc[pair['mol1_index']]['Activity']
                activity2 = df_quant.iloc[pair['mol2_index']]['Activity']
                score1 = activity_map.get(activity1, 0)
                score2 = activity_map.get(activity2, 0)
                pair['activity_category_diff'] = abs(score1 - score2)
            pairs.sort(key=lambda x: x.get('activity_category_diff', 0), reverse=True)
            st.session_state[f'quant_pairs_{tab_key}'] = pairs
            st.session_state[f'quant_data_{tab_key}'] = df_quant

    if f'quant_pairs_{tab_key}' in st.session_state:
        pairs = st.session_state[f'quant_pairs_{tab_key}']
        df_quant_valid = st.session_state[f'quant_data_{tab_key}']
        st.success(f"총 {len(pairs)}개의 유의미한 화합물 쌍을 찾았습니다.")
        if not pairs:
            st.warning("현재 조건에 맞는 화합물 쌍을 찾지 못했습니다. 임계값을 조절해보세요.")
        else:
            st.markdown("#### 상세 분석 목록")
            pair_options = [
                f"{idx+1}. {df_quant_valid.iloc[p['mol1_index']].get('ID', 'N/A')} vs {df_quant_valid.iloc[p['mol2_index']].get('ID', 'N/A')} "
                f"(유사도: {p['similarity']:.2f}, 분류차이 점수: {p.get('activity_category_diff', 0)})" 
                for idx, p in enumerate(pairs)
            ]
            selected_pair_str = st.selectbox("분석할 쌍을 선택하세요:", pair_options, key=f"pair_select_{tab_key}")
            if selected_pair_str:
                selected_idx = pair_options.index(selected_pair_str)
                pair_info = pairs[selected_idx]
                mol1 = df_quant_valid.iloc[pair_info['mol1_index']]
                mol2 = df_quant_valid.iloc[pair_info['mol2_index']]
                cliff_data_quant = {
                    'mol_1': mol1.to_dict(),
                    'mol_2': mol2.to_dict(),
                    'similarity': pair_info['similarity'],
                    'activity_diff': abs(mol1.get(ref_activity_col, 0) - mol2.get(ref_activity_col, 0)),
                    'structural_difference': get_structural_difference_keyword(mol1['SMILES'], mol2['SMILES']),
                    'is_stereoisomer': check_stereoisomers(mol1['SMILES'], mol2['SMILES']),
                    'mol1_properties': calculate_molecular_properties(mol1['mol']),
                    'mol2_properties': calculate_molecular_properties(mol2['mol']),
                    'same_scaffold': mol1.get('scaffold') == mol2.get('scaffold'),
                    'score': (abs(mol1.get(ref_activity_col, 0) - mol2.get(ref_activity_col, 0))) * (pair_info['similarity'] - sim_thresh) * (1 if mol1.get('scaffold') == mol2.get('scaffold') else 0.5)
                }
                process_and_display_pair(
                    idx=selected_idx, cliff_data=cliff_data_quant, sim_thresh=sim_thresh, 
                    activity_col=ref_activity_col, tab_key=f"quantitative_{tab_key}",
                    target_name=target_name, api_key=api_key, llm_provider=llm_provider
                )

def render_cliff_detection_ui(df, available_activity_cols, tab_key, target_name, api_key, llm_provider):
    st.info("구조가 유사하지만 **선택된 활성 값의 차이가 큰** 쌍(Activity Cliff)을 탐색합니다.")
    if not available_activity_cols:
        st.error("오류: 분석 가능한 활성 컬럼(pKi/pIC50)이 없습니다.")
        return
    selected_col = st.selectbox("분석 기준 컬럼 선택:", options=available_activity_cols, key=f'col_{tab_key}')
    with st.expander("현재 데이터 활성도 분포 보기"):
        plot_df_dist = df.copy()
        plot_df_dist[selected_col] = pd.to_numeric(plot_df_dist[selected_col], errors='coerce')
        plot_df_dist.dropna(subset=[selected_col], inplace=True)
        if not plot_df_dist.empty:
            st.metric(label=f"분석에 사용될 유효 데이터 개수", value=f"{len(plot_df_dist)} 개")
            display_cols = ['SMILES', 'Target', selected_col]
            st.dataframe(plot_df_dist[display_cols].head())
            fig_hist = px.histogram(plot_df_dist, x=selected_col, title=f'{selected_col} 값 분포', labels={selected_col: f'{selected_col} 값'})
            st.plotly_chart(fig_hist, use_container_width=True, key=f"histogram_{tab_key}")
        else:
            st.warning(f"'{selected_col}' 컬럼에 유효한 데이터가 없어 분포를 표시할 수 없습니다.")
    c1, c2 = st.columns(2)
    with c1: sim_thresh = st.slider("유사도 임계값", 0.5, 1.0, 0.8, 0.01, key=f'sim_{tab_key}')
    with c2: act_diff_thresh = st.slider(f"Δ{selected_col} 임계값", 0.1, 5.0, 1.0, 0.1, key=f'act_{tab_key}')
    if st.button("활성 절벽 탐지 실행", key=f'run_cliff_{tab_key}'):
        with st.spinner("활성 절벽 분석 중..."):
            cliffs = find_activity_cliffs(df, sim_thresh, act_diff_thresh, selected_col)
            st.session_state[f'cliffs_{tab_key}'] = cliffs
            st.session_state[f'analyzed_col_{tab_key}'] = selected_col

    if f'cliffs_{tab_key}' in st.session_state:
        cliffs = st.session_state[f'cliffs_{tab_key}']
        analyzed_col = st.session_state.get(f'analyzed_col_{tab_key}', selected_col)
        st.success(f"총 {len(cliffs)}개의 활성 절벽 쌍을 찾았습니다.")
        if cliffs:
            plot_df_scatter = pd.DataFrame(cliffs)
            plot_df_scatter['pair_label'] = plot_df_scatter.apply(
                lambda row: f"{row['mol_1'].get('ID', 'N/A')} vs {row['mol_2'].get('ID', 'N/A')}", axis=1
            )
            st.markdown("#### Activity Cliff 분포 시각화")
            fig_scatter = px.scatter(
                plot_df_scatter,
                x='similarity',
                y='activity_diff', 
                title='Activity Cliff 분포 (우측 상단이 가장 유의미한 영역)',
                labels={'similarity': '구조 유사도 (Tanimoto)', 'activity_diff': f'활성도 차이 (Δ{analyzed_col})'}, # <<< 여기도 수정
                hover_data=['pair_label', 'score'],
                color='score',
                color_continuous_scale=px.colors.sequential.Viridis,
                size='activity_diff' # <<< 여기도 수정
            )
            fig_scatter.add_shape(
                type="rect", xref="x", yref="y",
                x0=sim_thresh, y0=act_diff_thresh, x1=1.0, 
                y1=plot_df_scatter['activity_diff'].max() * 1.1, # <<< 여기를 수정
                line=dict(color="Red", width=2, dash="dash"),
                fillcolor="rgba(255,0,0,0.1)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("---")
        if not cliffs:
            st.warning("현재 조건에 맞는 활성 절벽을 찾지 못했습니다. 임계값을 조절해보세요.")
        else:
            st.markdown("#### 상세 분석 목록")
            pair_options = [
                f"{idx+1}. {c['mol_1'].get('ID', 'N/A')} vs {c['mol_2'].get('ID', 'N/A')} "
                f"(유사도: {c['similarity']:.2f}, 활성차이: {c['activity_diff']:.2f})"
                for idx, c in enumerate(cliffs)
            ]
            selected_pair_str = st.selectbox("분석할 쌍을 선택하세요:", pair_options, key=f"pair_select_{tab_key}")
            if selected_pair_str:
                selected_idx = pair_options.index(selected_pair_str)
                cliff = cliffs[selected_idx]
                process_and_display_pair(
                    idx=selected_idx, cliff_data=cliff, sim_thresh=sim_thresh, 
                    activity_col=analyzed_col, tab_key=tab_key,
                    target_name=target_name, api_key=api_key, llm_provider=llm_provider
                )


# --- [수정 시작] DB 연동을 위한 데이터 로딩 함수 ---
db_path = "/Users/lionkim/Downloads/project_archive/sar-project/patent_etl_pipeline/database/patent_data.db" 

@st.cache_data
def get_target_list(database_path):
    """DB의 targets 테이블에서 전체 타겟 이름 목록만 빠르게 가져옵니다."""
    if not os.path.exists(database_path):
        st.sidebar.error(f"DB 파일을 찾을 수 없습니다: {database_path}")
        return []
    try:
        conn = sqlite3.connect(database_path, check_same_thread=False)
        # targets 테이블에서 target_name만 조회
        query = "SELECT target_name FROM targets ORDER BY target_name;"
        df = pd.read_sql_query(query, conn)
        return df['target_name'].tolist()
    except Exception as e:
        st.sidebar.error(f"DB 타겟 목록 로딩 중 오류: {e}")
        return []
    finally:
        if 'conn' in locals() and conn:
            conn.close()

@st.cache_data
def get_data_for_target(database_path, target_name):
    """사용자가 선택한 특정 타겟의 데이터만 DB에서 JOIN하여 로드합니다."""
    if not os.path.exists(database_path): return None
    try:
        conn = sqlite3.connect(database_path, check_same_thread=False)
        # 제공해주신 쿼리에 WHERE 절을 추가하여 특정 타겟 데이터만 선택
        query = """
        SELECT
            c.smiles AS "SMILES",
            t.target_name AS "Target",
            a.pic50 AS "pIC50",
            a.ic50 AS "IC50",
            a.activity_category AS "Activity",
            c.compound_id AS "ID"
        FROM activities a
        JOIN compounds c ON a.compound_id = c.compound_id
        JOIN targets t ON a.target_id = t.target_id
        WHERE t.target_name = ?;
        """
        # SQL Injection 공격 방지를 위해 파라미터를 사용하여 안전하게 쿼리 실행
        df = pd.read_sql_query(query, conn, params=(target_name,))
        return df
    except Exception as e:
        st.error(f"'{target_name}' 데이터 로딩 중 오류: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()

@st.cache_data
def get_data_from_db(database_path):
    """SQLite 데이터베이스에서 데이터를 로드합니다."""
    if not os.path.exists(database_path):
        st.sidebar.error(f"DB 파일을 찾을 수 없습니다: {database_path}")
        return None
    try:
        conn = sqlite3.connect(database_path, check_same_thread=False)
        # 원본 utils.py의 load_data가 처리하는 컬럼들을 모두 가져옵니다.
        # 컬럼명을 원본 load_data 함수가 기대하는 형식과 유사하게 맞춰줍니다.
        query = """
        SELECT
            c.smiles AS "SMILES",
            t.target_name AS "Target",
            a.pic50 AS "pIC50",
            a.ic50 AS "IC50",
            a.activity_category AS "Activity",
            c.compound_id AS "ID"
        FROM activities a
        JOIN compounds c ON a.compound_id = c.compound_id
        JOIN targets t ON a.target_id = t.target_id;
        """
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.sidebar.error(f"DB 로딩 중 오류: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()
# --- [수정 끝] ---


# --- Main App ---
def main():
    with st.sidebar:
        st.title("AI SAR 분석 시스템")
        st.info("AI 기반 구조-활성 관계(SAR) 분석 및 예측 솔루션입니다.")
        
        st.header("📁 데이터 선택")
        
        # 1단계: 전체 타겟 목록만 빠르게 로드하여 Selectbox를 생성합니다.
        target_list = get_target_list(db_path)
        selected_target = None
        
        if target_list:
            selected_target = st.selectbox('분석할 타겟 선택', target_list)
        else:
            st.warning("데이터베이스에서 타겟 목록을 불러올 수 없습니다.")

        st.header("⚙️ AI 모델 설정")
        # target_name_input은 이제 기본값 또는 보조 용도로만 사용됩니다.
        target_name_input = st.text_input("분석 대상 타겟 단백질 (참고용)", value=selected_target or "EGFR")
        llm_provider = st.selectbox("LLM 공급자 선택:", ("OpenAI", "Gemini"))
        api_key = st.text_input("API 키 입력:", type="password", placeholder="OpenAI 또는 Gemini API 키")

    st.header("분석 결과 대시보드")
    df, available_activity_cols = None, []
    
    # 2단계: 사용자가 타겟을 선택한 경우에만 해당 데이터를 DB에서 로드합니다.
    if selected_target:
        with st.spinner(f"'{selected_target}' 데이터 로딩 중..."):
            # 특정 타겟의 데이터만 DB에서 가져옵니다.
            df_from_db = get_data_for_target(db_path, selected_target)
        
        if df_from_db is not None:
            # 3단계: 로드된 데이터를 후처리 함수(utils.py의 load_data)로 전달합니다.
            df_processed, available_activity_cols = load_data(df_from_db)

            if df_processed is not None:
                # --- [수정된 부분 시작] ---
                # 이 단계에서 미리 중복을 제거합니다.
                ref_col = available_activity_cols[0] if available_activity_cols else 'pIC50'
                if ref_col in df_processed.columns:
                    # 1. 활성도가 높은 순으로 정렬
                    df_sorted = df_processed.sort_values(ref_col, ascending=False)
                    # 2. SMILES 기준 중복 제거 (가장 활성도 높은 데이터만 남김)
                    df = df_sorted.drop_duplicates(subset=['SMILES'], keep='first')
                else:
                    df = df_processed.drop_duplicates(subset=['SMILES'], keep='first')
                
                st.sidebar.success(f"총 {len(df_from_db)}개 데이터 중 {len(df)}개의 고유 화합물 로드 완료!")
                # --- [수정된 부분 끝] ---
            
            # Activity 컬럼이 없는 경우, pKi/pIC50 기준으로 자동 생성합니다.
            if df is not None and 'Activity' not in df.columns and any(col in df.columns for col in ['pKi', 'pIC50']):
                ref_col = 'pKi' if 'pKi' in df.columns else 'pIC50'
                conditions = [
                    (df[ref_col] > 7.0),
                    (df[ref_col] > 5.7) & (df[ref_col] <= 7.0),
                    (df[ref_col] > 5.0) & (df[ref_col] <= 5.7),
                    (df[ref_col] <= 5.0) | (df[ref_col].isna())
                ]
                labels = ['Highly Active', 'Moderately Active', 'Weakly Active', 'Inactive']
                df['Activity'] = np.select(conditions, labels, default='Unclassified')
                st.info("Info: pKi/pIC50 값을 기준으로 Activity 컬럼을 새로 생성했습니다.")

    # 4단계: 최종 처리된 데이터(df)가 있을 경우에만 분석 탭들을 렌더링합니다.
    if df is not None:
        st.success(f"'{selected_target}'에 대한 {len(df)}개의 화합물 데이터 분석 준비 완료!")
        
        tabs_to_create = []
        if ONLINE_DISCUSSION_AVAILABLE: tabs_to_create.append("SAR 분석 (토론 시스템 적용)")
        tabs_to_create.append("SAR 분석 (기본)")
        if PROMPT_SYSTEM_AVAILABLE: tabs_to_create.append("최적 프롬프트 토론")
        
        created_tabs = st.tabs(tabs_to_create)
        tab_map = {name: tab for name, tab in zip(tabs_to_create, created_tabs)}
        
        tab_advanced = tab_map.get("SAR 분석 (토론 시스템 적용)")
        tab_basic = tab_map.get("SAR 분석 (기본)")
        tab_prompt = tab_map.get("최적 프롬프트 토론")

        # 분석 함수에 전달할 타겟 이름은 이제 사이드바에서 선택된 값을 사용합니다.
        target_name_to_use = selected_target

        if tab_advanced:
            with tab_advanced:
                st.subheader("구조-활성 관계 분석 (토론 시스템 적용)")
                analysis_type_adv = st.radio("분석 유형 선택:", ("활성 절벽 탐지", "정량 분석"), horizontal=True, key="adv_type")
                st.markdown("---")
                if analysis_type_adv == "정량 분석":
                    render_quantitative_analysis_ui(df, available_activity_cols, 'advanced', target_name_to_use, api_key, llm_provider)
                else:
                    render_cliff_detection_ui(df, available_activity_cols, 'advanced', target_name_to_use, api_key, llm_provider)
        
        if tab_basic:
            with tab_basic:
                st.subheader("구조-활성 관계 분석 (기본)")
                analysis_type_basic = st.radio("분석 유형 선택:", ("활성 절벽 탐지", "정량 분석"), horizontal=True, key="basic_type")
                st.markdown("---")
                if analysis_type_basic == "정량 분석":
                    render_quantitative_analysis_ui(df, available_activity_cols, 'basic', target_name_to_use, api_key, llm_provider)
                else:
                    render_cliff_detection_ui(df, available_activity_cols, 'basic', target_name_to_use, api_key, llm_provider)

        if tab_prompt:
            with tab_prompt:
                st.markdown("# 최적 프롬프트 토론 시스템")
                # (이하 프롬프트 토론 탭 로직은 기존과 동일)
    else:
        st.info("분석을 시작하려면 사이드바에서 분석할 타겟을 선택하세요.")

if __name__ == "__main__":
    main()

