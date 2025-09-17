import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import plotly.express as px
import json
from sqlalchemy.orm import Session
from patent_etl_pipeline.database import (
    SessionLocal, Patent, Compound, Target, Activity,
    DATABASE_PATH, init_db
)
from patent_etl_pipeline.run_etl import run_etl

from utils import (
    load_data,
    find_quantitative_pairs,
    generate_hypothesis_quantitative,
    draw_highlighted_pair,
    check_stereoisomers,
    calculate_molecular_properties,
    get_structural_difference_keyword,
    save_results_to_db,
    get_analysis_history
)

# --- 외부 시스템 임포트 ---
try:
    from online_discussion_system import run_online_discussion_system
    ONLINE_DISCUSSION_AVAILABLE = True
    print("✅ 시스템 로드 성공")
except ImportError as e:
    ONLINE_DISCUSSION_AVAILABLE = False
    print(f"❌ 시스템 로드 실패: {str(e)}")


# --- 페이지 기본 설정 ---
st.set_page_config(page_title="AI 기반 SAR 분석 시스템", page_icon="🧪", layout="wide")


# --- 공통 로직 처리 헬퍼 함수 ---
def process_and_display_pair(idx, cliff_data, activity_col, tab_key, target_name, api_key, llm_provider, selected_patent, cell_line=None):
    mol1 = pd.Series(cliff_data['mol_1'])
    mol2 = pd.Series(cliff_data['mol_2'])
    similarity = cliff_data['similarity']
    
    header = f"쌍 #{idx+1} (ID: {mol1.get('ID', 'N/A')} vs {mol2.get('ID', 'N/A')}) | 유사도: {similarity:.3f}"
    
    with st.expander(header, expanded=True):
        real_act_diff = cliff_data['activity_difference']
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
                        hypothesis, context = generate_hypothesis_quantitative(mol1, mol2, similarity, target_name, api_key, llm_provider)
                        st.markdown(hypothesis)
                        if context:
                            with st.expander("도킹 시뮬레이션 결과"): st.json(context)

        elif tab_key.endswith('advanced'):

         if st.button("분석 시작 및 결과 저장", key=f"disc_{idx}_{tab_key}"):
            if not api_key: 
                st.warning("사이드바에서 API 키를 입력해주세요.")
            elif not ONLINE_DISCUSSION_AVAILABLE: 
                st.error("온라인 다각도 분석 시스템 모듈을 로드할 수 없습니다.")
            else:
                with st.spinner("AI 전문가들이 다각도 분석 후 최종 리포트를 작성합니다..."):
                    # 1. 온라인 다각도 분석 시스템 실행하여 최종 리포트 받기
                    # target_name: PDB ID (도킹용), cell_line: 세포주 (실험조건용)
                    final_report = run_online_discussion_system(cliff_data, target_name, api_key, llm_provider, cell_line)
                    
                    # 도킹 시뮬레이션 결과가 있는 경우 별도 표시
                    if isinstance(final_report, dict) and 'domain_hypotheses' in final_report:
                        for hypothesis in final_report['domain_hypotheses']:
                            if hypothesis.get('agent_name') == '생체분자 상호작용 전문가' and 'docking_analysis' in hypothesis:
                                with st.expander("도킹 시뮬레이션 결과", expanded=False):
                                    try:
                                        from online_discussion_system import display_docking_results
                                        display_docking_results(hypothesis['docking_analysis'], hypothesis['agent_name'])
                                    except ImportError:
                                        st.write("도킹 시뮬레이션 모듈을 로드할 수 없습니다.")
                                    
                                    with st.expander("상세 데이터 (JSON)", expanded=False):
                                        st.json(hypothesis['docking_analysis'])
                                break
                    
                    # JSON 상세 분석을 토글로 표시
                    with st.expander("리포트 상세 결과 (JSON)", expanded=False):
                        st.json(final_report)

                    # 2. utils의 함수를 호출하여 DB에 최종 리포트 저장
                    # final_report가 dict 형태일 수 있으므로, json.dumps로 텍스트 변환
                    report_text = json.dumps(final_report, indent=2, ensure_ascii=False)
                    
                    saved_id = save_results_to_db(
                        patent_number=selected_patent,
                        cliff_data=cliff_data,
                        hypothesis_text=report_text, # 최종 리포트를 저장
                        llm_provider="Expert Discussion System", # 에이전트 이름 변경
                        context_info=None # 리포트 자체에 포함된 것으로 간주
                    )

                    if saved_id:
                        st.success(f"분석 리포트가 데이터베이스에 성공적으로 저장되었습니다. (Analysis ID: {saved_id})")
                    else:
                        st.error("데이터베이스 저장에 실패했습니다.")


# --- UI 렌더링 함수  ---

def render_quantitative_analysis_ui(df, available_activity_cols, tab_key, target_name, api_key, llm_provider, selected_patent, cell_line=None):
    st.info("구조적으로 유사하지만 **활성 분류(Activity)가 다른** 화합물 쌍을 탐색합니다.")
    if 'Activity' not in df.columns or not available_activity_cols:
        st.error("오류: 분석에 필요한 'Activity' 또는 활성 컬럼(pIC50/pKi)이 없습니다.")
        return
    ref_activity_col = available_activity_cols[0]

    sim_thresh = st.slider("유사도 임계값", 0.5, 1.0, 0.8, 0.01, key=f'sim_quant_{tab_key}')
    
    if st.button("정량 분석 실행", key=f'run_quant_{tab_key}'):
        with st.spinner("정량 분석 중..."):
            # --- [수정된 부분] ---
            # 복잡한 분석 로직 대신 utils.py의 함수를 한 줄로 호출합니다.
            pairs, df_quant_processed = find_quantitative_pairs(df, sim_thresh, ref_activity_col)
            # --- [수정된 부분 끝] ---
            
            st.session_state[f'quant_pairs_{tab_key}'] = pairs
            st.session_state[f'quant_data_{tab_key}'] = df_quant_processed

    if f'quant_pairs_{tab_key}' in st.session_state:
        pairs = st.session_state[f'quant_pairs_{tab_key}']
        df_quant_valid = st.session_state[f'quant_data_{tab_key}']
        
        st.success(f"총 {len(pairs)}개의 유의미한 화합물 쌍을 찾았습니다.")
        if not pairs:
            st.warning("현재 조건에 맞는 화합물 쌍을 찾지 못했습니다. 임계값을 조절해보세요.")
        else:
            # Activity Cliff 분포 시각화 추가
            quantitative_plot_data = []
            for p in pairs:
                mol1 = df_quant_valid.iloc[p['mol1_index']]
                mol2 = df_quant_valid.iloc[p['mol2_index']]
                quantitative_plot_data.append({
                    'similarity': p['similarity'],
                    'activity_difference': abs(mol1.get(ref_activity_col, 0) - mol2.get(ref_activity_col, 0)),
                    'pair_label': f"{mol1.get('ID', 'N/A')} vs {mol2.get('ID', 'N/A')}",
                    'score': p.get('activity_category_diff', 0)
                })
            
            if quantitative_plot_data:
                plot_df_quant = pd.DataFrame(quantitative_plot_data)
                st.markdown("#### Activity Cliff 분포 시각화")
                fig_quant_scatter = px.scatter(
                    plot_df_quant,
                    x='similarity',
                    y='activity_difference', 
                    title='Activity Cliff 분포 (우측 상단이 가장 유의미한 영역)',
                    labels={'similarity': '구조 유사도 (Tanimoto)', 'activity_difference': f'활성도 차이 (Δ{ref_activity_col})'}, 
                    hover_data=['pair_label', 'score'],
                    color='score',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    size='activity_difference' 
                )
                fig_quant_scatter.add_shape(
                    type="rect", xref="x", yref="y",
                    x0=sim_thresh, y0=0, x1=1.0, 
                    y1=plot_df_quant['activity_difference'].max() * 1.1,
                    line=dict(color="Red", width=2, dash="dash"),
                    fillcolor="rgba(255,0,0,0.1)"
                )
                st.plotly_chart(fig_quant_scatter, use_container_width=True)
                st.markdown("---")
            
            st.markdown("#### 상세 분석 목록")
            pair_options = [
                f"{idx+1}. {df_quant_valid.iloc[p['mol1_index']].get('ID', 'N/A')} vs {df_quant_valid.iloc[p['mol2_index']].get('ID', 'N/A')} "
                f"(유사도: {p['similarity']:.2f}, 활성차이: {abs(df_quant_valid.iloc[p['mol1_index']].get(ref_activity_col, 0) - df_quant_valid.iloc[p['mol2_index']].get(ref_activity_col, 0)):.2f}, 분류차이 점수: {p.get('activity_category_diff', 0)})"
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
                    'activity_difference': abs(mol1.get(ref_activity_col, 0) - mol2.get(ref_activity_col, 0)),
                    'structural_difference': get_structural_difference_keyword(mol1['SMILES'], mol2['SMILES']),
                    'is_stereoisomer': check_stereoisomers(mol1['SMILES'], mol2['SMILES']),
                    'mol1_properties': calculate_molecular_properties(mol1['mol']),
                    'mol2_properties': calculate_molecular_properties(mol2['mol']),
                    'same_scaffold': mol1.get('scaffold') == mol2.get('scaffold'),
                    'score': (abs(mol1.get(ref_activity_col, 0) - mol2.get(ref_activity_col, 0))) * (pair_info['similarity'] - sim_thresh) * (1 if mol1.get('scaffold') == mol2.get('scaffold') else 0.5)
                }
                process_and_display_pair(
                    idx=selected_idx, cliff_data=cliff_data_quant,
                    activity_col=ref_activity_col, tab_key=f"quantitative_{tab_key}",
                    target_name=target_name, api_key=api_key, llm_provider=llm_provider, selected_patent=selected_patent,
                    cell_line=cell_line
                )



# --- DB 연동을 위한 데이터 로딩 함수 ---
db_path = "patent_etl_pipeline/database/patent_data.db" 

@st.cache_data
def get_patent_list():
    """DB에서 전체 특허 번호 목록만 빠르게 가져옵니다."""
    db: Session = SessionLocal()
    try:
        patents = db.query(Patent.patent_number).order_by(Patent.patent_number.desc()).all()
        return [p[0] for p in patents] # 튜플 리스트를 일반 리스트로 변환
    except Exception as e:
        st.sidebar.error(f"DB 특허 목록 로딩 중 오류: {e}")
        return []
    finally:
        db.close()

@st.cache_data
def get_targets_for_patent(patent_number):
    """입력된 특허 번호에 해당하는 모든 타겟의 이름을 DB에서 찾아 반환합니다."""
    if not patent_number: return []
    db: Session = SessionLocal()
    try:
        targets = db.query(Target.target_name)\
                    .join(Activity, Target.target_id == Activity.target_id)\
                    .join(Patent, Activity.patent_id == Patent.patent_id)\
                    .filter(Patent.patent_number == patent_number)\
                    .distinct().order_by(Target.target_name).all()
        return [t[0] for t in targets]
    except Exception as e:
        st.sidebar.error(f"특허 '{patent_number}'의 타겟 목록 로딩 중 오류: {e}")
        return []
    finally:
        db.close()

@st.cache_data
def get_data_for_patent_and_target(patent_number, target_name):
    """특정 특허와 특정 타겟에 대한 데이터만 DB에서 JOIN하여 가져옵니다."""
    if not patent_number: return None  # 특허는 필수
    db: Session = SessionLocal()
    try:
        # SQLAlchemy의 read_sql_query를 사용하여 DataFrame으로 직접 변환
        query = db.query(
                    Compound.smiles.label("SMILES"),
                    Compound.compound_id.label("ID"),
                    Target.target_name.label("Target"),
                    Patent.patent_number.label("Patent"),
                    Activity.ic50.label("IC50"),
                    Activity.pic50.label("pIC50"),
                    Activity.activity_category.label("Activity")
                ).join(Activity, Compound.compound_id == Activity.compound_id)\
                 .join(Target, Activity.target_id == Target.target_id)\
                 .join(Patent, Activity.patent_id == Patent.patent_id)\
                 .filter(Patent.patent_number == patent_number)
        
        # target_name이 지정된 경우에만 타겟 필터 추가
        if target_name:
            query = query.filter(Target.target_name == target_name)
        
        query = query.statement
        df = pd.read_sql_query(query, db.bind)
        return df
    except Exception as e:
        st.error(f"데이터 로딩 중 오류: {e}")
        return None
    finally:
        db.close()

# --- Main App ---
def main():
    if not os.path.exists(DATABASE_PATH):
        st.title("🚀 SAR 분석 시스템 초기 설정")
        st.info("최초 실행을 위해 데이터베이스를 자동으로 생성합니다. 잠시만 기다려주세요...")
        try:
            init_db() # database.py의 함수 호출
            st.success("데이터베이스가 성공적으로 생성되었습니다!")
            st.info("이제 ETL 스크립트(run_etl.py)를 실행하여 데이터를 채워주시거나, 앱 내 데이터 로드 기능을 이용해 주세요.")
            st.rerun()
        except Exception as e:
            st.error(f"데이터베이스 생성 중 오류 발생: {e}")
        return # DB 생성 후 일단 정지

    with st.sidebar:
        st.title("AI SAR 분석 시스템")
        st.info("AI 기반 구조-활성 관계(SAR) 분석 및 예측 솔루션입니다.")   

        # --- 데이터 로드 UI ---
        with st.expander("📚 데이터 관리 (신규 특허 로드)", expanded=False):
            patent_number_input = st.text_input("특허 번호 입력", placeholder="예: 1020170094694")
            uploaded_file = st.file_uploader("특허 엑셀 파일 업로드", type=["xlsx"])
            
            if st.button("데이터베이스에 저장"):
                if patent_number_input and uploaded_file:
                    with st.spinner("ETL 프로세스 실행 중... 엑셀 파일을 읽고 DB에 저장합니다."):
                        # 업로드된 파일을 임시 저장하지 않고 메모리에서 바로 사용
                        file_buffer = io.BytesIO(uploaded_file.getvalue())
                        
                        # run_etl.py의 함수 호출
                        progress_bar = st.progress(0, text="ETL 시작...")
                        success, message = run_etl(patent_number_input, file_buffer, progress_bar)
                        
                        if success:
                            st.success(message)
                            # 드롭다운 목록을 업데이트하기 위해 get_patent_list 함수의 캐시를 지웁니다.
                            get_patent_list.clear()
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.warning("특허 번호와 엑셀 파일을 모두 입력/업로드해주세요.")
 
        st.header("📁 데이터 선택")
        
        # 1. 특허 번호 입력 (DB에 있는 목록에서 선택하거나 직접 입력)
        patent_list = get_patent_list()
        selected_patent = st.selectbox("1. 분석할 특허 번호를 선택하세요:", options=[""] + patent_list)
        
        # 2. 선택된 특허에 포함된 타겟 목록 표시
        selected_target = None
        if selected_patent:
            target_list = get_targets_for_patent(selected_patent)
            if target_list:
                selected_target = st.selectbox("2. 분석할 타겟을 선택하세요:", options=[""] + target_list)
            else:
                st.warning(f"'{selected_patent}' 특허에 대한 타겟 데이터가 없습니다.")

        st.header("⚙️ AI 모델 설정")
        # 특허별 기본 타겟 단백질 PDB ID 설정
        # 1020170094694 특허의 경우에만 6G6K를 기본값으로 사용
        default_target_pdb = "6G6K" if selected_patent and "1020170094694" in selected_patent else ""
        target_name_input = st.text_input("분석 대상 타겟 단백질 (PDB ID)", value=default_target_pdb, 
                                         help="타겟 단백질의 PDB ID를 입력하세요. 예: 6G6K, 1M17, 4ZAU 등")
        llm_provider = st.selectbox("LLM 공급자 선택:", ("OpenAI", "Gemini"))
        api_key = st.text_input("API 키 입력:", type="password", placeholder="OpenAI 또는 Gemini API 키")

    # --- 탭 구조 정의 ---
    tab_titles = ["실시간 분석", "분석 이력 조회"]

    created_tabs = st.tabs(tab_titles)
    tab_map = {name: tab for name, tab in zip(tab_titles, created_tabs)}

    # --- 탭 1: 실시간 분석 ---
    with tab_map["실시간 분석"]:
        st.header("실시간 분석 대시보드")
        df, available_activity_cols = None, []

        # 특허가 선택되고 타겟(selected_target)이 선택되었을 때 데이터 로드
        # 타겟은 데이터 필터링용, PDB ID는 도킹 시뮬레이션용으로 각각 사용
        if selected_patent and selected_target:
            with st.spinner(f"특허 '{selected_patent}'의 '{selected_target}' 타겟 데이터 로딩 중..."):
                # 1. 특허와 세포주에 맞는 데이터를 DB에서 가져옵니다.
                df_from_db = get_data_for_patent_and_target(selected_patent, selected_target)

            if df_from_db is not None:
                # 2. 가져온 데이터를 utils.py의 load_data 함수로 후처리합니다.
                df_processed, available_activity_cols = load_data(df_from_db)

                if df_processed is not None:
                    # 3. 데이터 로드 직후, 분석 전에 중복 화합물을 제거합니다.
                    ref_col = available_activity_cols[0] if available_activity_cols else 'pIC50'
                    if ref_col in df_processed.columns:
                        # 활성도가 높은 순으로 정렬
                        df_sorted = df_processed.sort_values(ref_col, ascending=False)
                        # SMILES 기준 중복 제거 (가장 활성도 높은 데이터만 남김)
                        df = df_sorted.drop_duplicates(subset=['SMILES'], keep='first')
                    else:
                        # 활성 컬럼이 없을 경우, 그냥 SMILES 기준으로 중복 제거
                        df = df_processed.drop_duplicates(subset=['SMILES'], keep='first')

                    st.sidebar.success(f"총 {len(df_from_db)}개 행 중 {len(df)}개의 고유 화합물 로드 완료!")

                    # 4. Activity 컬럼이 없는 경우, pIC50/pKi 기준으로 자동 생성합니다.
                    if 'Activity' not in df.columns and any(col in df.columns for col in ['pIC50', 'pKi']):
                        ref_col_act = 'pIC50' if 'pIC50' in df.columns else 'pKi'
                        conditions = [
                            (df[ref_col_act] > 7.0),
                            (df[ref_col_act] > 5.7) & (df[ref_col_act] <= 7.0),
                            (df[ref_col_act] > 5.0) & (df[ref_col_act] <= 5.7),
                            (df[ref_col_act] <= 5.0) | (df[ref_col_act].isna())
                        ]
                        labels = ['Highly Active', 'Moderately Active', 'Weakly Active', 'Inactive']
                        df['Activity'] = np.select(conditions, labels, default='Unclassified')
                        st.info("Info: pIC50/pKi 값을 기준으로 Activity 컬럼을 새로 생성했습니다.")

        # 최종 처리된 데이터(df)가 있을 경우에만 분석 UI를 렌더링합니다.
        if df is not None:
            st.success(f"'{selected_target}' 타겟에 대한 {len(df)}개의 화합물 데이터 분석 준비 완료!")

            # PDB ID와 세포주 정보 검증
            if not target_name_input:
                st.warning("타겟 단백질 PDB ID가 입력되지 않았습니다. 도킹 시뮬레이션이 제한될 수 있습니다.")
            
            # 분석 함수에 전달할 정보들
            target_protein_pdb = target_name_input  # 도킹 시뮬레이션용 PDB ID
            cell_line_name = selected_target        # 실험 조건용 세포주

            # SAR 분석 UI (온라인 토론 시스템 사용 가능한 경우만 표시)
            if ONLINE_DISCUSSION_AVAILABLE:
                st.markdown("---")
                st.subheader("구조-활성 관계 분석 (활성 절벽 탐지)")
                render_quantitative_analysis_ui(df, available_activity_cols, 'advanced', target_protein_pdb, api_key, llm_provider, selected_patent, cell_line_name)
            else:
                st.error("온라인 다각도 분석 시스템을 로드할 수 없습니다.")
        else:
            st.info("분석을 시작하려면 사이드바에서 특허와 타겟을 모두 선택하세요.")

    # --- 탭 2: 분석 이력 조회 ---
    with tab_map["분석 이력 조회"]:
        st.header("분석 이력 조회")

        with st.spinner("과거 분석 이력을 불러오는 중..."):
            history_df = get_analysis_history()

        if history_df.empty:
            st.info("저장된 분석 이력이 없습니다. '실시간 분석' 탭에서 분석을 실행하고 결과를 저장해주세요.")
        else:
            st.info(f"총 {len(history_df)}개의 분석 이력을 찾았습니다.")
            
            search_id = st.text_input("검색할 화합물 ID (compound_id_1 또는 compound_id_2):")
            
            display_df = history_df
            if search_id:
                try:
                    search_id_int = int(search_id)
                    display_df = history_df[
                        (history_df['compound_id_1'] == search_id_int) | 
                        (history_df['compound_id_2'] == search_id_int)
                    ]
                except ValueError:
                    st.warning("ID는 숫자로 입력해주세요.")

            # 테이블 표시 전에 시간을 한국 시간으로 변환
            if not display_df.empty and 'analysis_timestamp' in display_df.columns:
                from datetime import timedelta
                display_df_copy = display_df.copy()
                display_df_copy['analysis_timestamp'] = display_df_copy['analysis_timestamp'] + timedelta(hours=9)
                display_df_copy['analysis_timestamp'] = display_df_copy['analysis_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(display_df_copy)
            else:
                st.dataframe(display_df)

            st.markdown("---")
            st.subheader("상세 정보 보기")
            
            # 검색 결과가 있으면 검색 결과 내에서, 없으면 전체 이력 내에서 선택
            detail_options = [""] + display_df['analysis_id'].tolist()
            selected_analysis_id = st.selectbox(
                "상세히 볼 분석 ID를 선택하세요:", 
                options=detail_options
            )

            if selected_analysis_id:
                detail_data = history_df[history_df['analysis_id'] == selected_analysis_id].iloc[0]
                
                # UTC 시간을 한국 시간으로 변환
                from datetime import timedelta
                
                # pandas Timestamp에 9시간 더하기
                kst_time = detail_data['analysis_timestamp'] + timedelta(hours=9)
                formatted_time = kst_time.strftime('%Y-%m-%d %H:%M:%S')
                
                st.json({
                    "분석 ID": detail_data['analysis_id'],
                    "분석 시간": formatted_time,
                    "분석 쌍": f"ID {detail_data['compound_id_1']} vs ID {detail_data['compound_id_2']}",
                    "유사도": f"{detail_data['similarity']:.3f}" if pd.notna(detail_data['similarity']) else "N/A",
                    "활성 차이": f"{detail_data['activity_difference']:.3f}" if pd.notna(detail_data['activity_difference']) else "N/A",
                    "점수": f"{detail_data['score']:.2f}" if pd.notna(detail_data['score']) else "N/A",
                    "분석 에이전트": detail_data['agent_name']
                })

                st.markdown("##### AI 생성 가설/리포트")
                try:
                    report_json = json.loads(detail_data['hypothesis_text'])
                    st.json(report_json)
                except (json.JSONDecodeError, TypeError):
                    st.info(detail_data['hypothesis_text'] or "저장된 가설이 없습니다.")

if __name__ == "__main__":
    main()
