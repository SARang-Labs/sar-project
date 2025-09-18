import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdFMCS, rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
import google.generativeai as genai
from openai import OpenAI
import json
from patent_etl_pipeline.database import SessionLocal, Patent, SAR_Analysis, AI_Hypothesis
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 현재 파일의 디렉토리 기준으로 docking_pipeline 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
docking_path = os.path.join(current_dir, 'docking_pipeline')

if docking_path not in sys.path:
    sys.path.append(docking_path)

# --- 결과 저장 함수 ---
def save_results_to_db(patent_number, cliff_data, hypothesis_text, llm_provider, context_info=None):
    """
    [SQLAlchemy] 분석 결과(cliff)와 AI 가설을 데이터베이스에 저장합니다.
    """
    db = SessionLocal()
    try:
        # patent_number를 이용해 patent_id를 찾습니다.
        patent = db.query(Patent).filter(Patent.patent_number == patent_number).first()
        if not patent:
            print(f"오류: DB에서 특허 '{patent_number}'를 찾을 수 없습니다.")
            return None

        # 1. sar_analyses 테이블에 분석 결과 저장
        # ID 값을 정수로 변환 (문자열인 경우 해시 또는 기본값 사용)
        def safe_int_id(id_value):
            if isinstance(id_value, (int, float)):
                return int(id_value)
            elif isinstance(id_value, str):
                try:
                    return int(id_value)
                except ValueError:
                    # 문자열 ID인 경우 해시값 사용 (또는 0으로 기본값)
                    return abs(hash(id_value)) % 1000000
            return 0
        
        new_analysis = SAR_Analysis(
            patent_id=patent.patent_id,
            compound_id_1=safe_int_id(cliff_data['mol_1'].get('ID')),
            compound_id_2=safe_int_id(cliff_data['mol_2'].get('ID')),
            similarity=cliff_data.get('similarity'),
            activity_difference=cliff_data.get('activity_diff'),
            score=cliff_data.get('score')
        )
        db.add(new_analysis)
        db.flush() # DB에 임시 반영하여 analysis_id를 얻음

        # 2. ai_hypotheses 테이블에 가설 저장
        if hypothesis_text:
            context_text = json.dumps(context_info, ensure_ascii=False) if context_info else None
            new_hypothesis = AI_Hypothesis(
                analysis_id=new_analysis.analysis_id,
                agent_name=llm_provider,
                hypothesis_text=hypothesis_text,
                context_info=context_text
            )
            db.add(new_hypothesis)

        db.commit() # 모든 변경사항을 DB에 최종 저장
        return new_analysis.analysis_id
    except Exception as e:
        print(f"DB 저장 중 오류 발생: {e}")
        db.rollback()
        return None
    finally:
        db.close()

# --- 데이터 조회 함수 ---
def get_analysis_history():
    """
    [SQLAlchemy] SAR 분석 및 AI 가설 전체 이력을 데이터베이스에서 가져옵니다.
    """
    db = SessionLocal()
    try:
        # SQLAlchemy ORM을 사용하여 JOIN 쿼리 작성
        query = db.query(
                    Patent.patent_number,
                    SAR_Analysis.analysis_id,
                    SAR_Analysis.analysis_timestamp,
                    SAR_Analysis.compound_id_1,
                    SAR_Analysis.compound_id_2,
                    SAR_Analysis.similarity,
                    SAR_Analysis.activity_difference,
                    SAR_Analysis.score,
                    AI_Hypothesis.hypothesis_text,
                    AI_Hypothesis.agent_name.label("agent_name")
                ).join(SAR_Analysis, Patent.patent_id == SAR_Analysis.patent_id)\
                 .outerjoin(AI_Hypothesis, SAR_Analysis.analysis_id == AI_Hypothesis.analysis_id)\
                 .order_by(Patent.patent_number, SAR_Analysis.analysis_timestamp.desc()).statement
        
        df = pd.read_sql_query(query, db.bind)
        
        # Arrow 변환 문제를 방지하기 위해 ID 컬럼들을 정수형으로 확실히 변환
        if not df.empty:
            for col in ['compound_id_1', 'compound_id_2']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        st.error(f"분석 이력 로딩 중 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        db.close()


# --- Helper Functions ---
def canonicalize_smiles(smiles):
    """SMILES를 RDKit의 표준 Isomeric SMILES로 변환합니다."""
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return None

def get_structural_difference_keyword(smiles1, smiles2):
    """두 SMILES의 구조적 차이를 나타내는 키워드를 반환합니다."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if not mol1 or not mol2:
        return None

    # 최대 공통 부분구조(MCS) 찾기
    mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=5)
    if mcs_result.numAtoms == 0:
        return "significant structural difference"
    
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    
    # 각 분자에서 MCS를 제외한 부분(차이점) 찾기
    diff1_mol = Chem.ReplaceCore(mol1, mcs_mol)
    diff2_mol = Chem.ReplaceCore(mol2, mcs_mol)

    fragments = []
    try:
        if diff1_mol:
            for frag in Chem.GetMolFrags(diff1_mol, asMols=True):
                try:
                    fragments.append(Chem.MolToSmiles(frag))
                except:
                    continue
        if diff2_mol:
            for frag in Chem.GetMolFrags(diff2_mol, asMols=True):
                try:
                    fragments.append(Chem.MolToSmiles(frag))
                except:
                    continue
    except Exception:
        pass
    
    # 간단한 작용기 이름으로 변환
    if fragments:
        # 가장 흔한 작용기 이름 몇 개만 간단히 매핑
        common_names = {
            'c1ccccc1': 'phenyl', 'c1ccncc1': 'pyridine', '[F]': 'fluorine',
            '[Cl]': 'chlorine', '[OH]': 'hydroxyl', '[CH3]': 'methyl'
        }
        # 가장 긴 fragment를 대표로 사용
        longest_frag = max(fragments, key=len)
        for smiles_frag, name in common_names.items():
            if smiles_frag in longest_frag:
                return name
        return "moiety modification"
        
    return "structural modification"

def check_stereoisomers(smiles1, smiles2):
    """두 SMILES가 입체이성질체인지 확인합니다."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if not mol1 or not mol2:
        return False
    
    # 2D 구조가 같고 3D 구조가 다르면 입체이성질체
    canonical_1 = Chem.MolToSmiles(mol1, isomericSmiles=False)
    canonical_2 = Chem.MolToSmiles(mol2, isomericSmiles=False)
    isomeric_1 = Chem.MolToSmiles(mol1, isomericSmiles=True)
    isomeric_2 = Chem.MolToSmiles(mol2, isomericSmiles=True)
    
    return canonical_1 == canonical_2 and isomeric_1 != isomeric_2

def calculate_molecular_properties(mol):
    """분자의 주요 물리화학적 특성을 계산합니다."""
    if not mol:
        return {}
    
    properties = {}
    try:
        properties['molecular_weight'] = Descriptors.MolWt(mol)
        properties['logp'] = Descriptors.MolLogP(mol)
        properties['hbd'] = Descriptors.NumHDonors(mol)
        properties['hba'] = Descriptors.NumHAcceptors(mol)
        properties['tpsa'] = Descriptors.TPSA(mol)
        properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        properties['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        properties['heavy_atoms'] = mol.GetNumHeavyAtoms()
        properties['formal_charge'] = Chem.rdmolops.GetFormalCharge(mol)
    except Exception:
        # 오류 발생 시 기본값들로 채움
        properties = {
            'molecular_weight': 0.0,
            'logp': 0.0,
            'hbd': 0,
            'hba': 0,
            'tpsa': 0.0,
            'rotatable_bonds': 0,
            'aromatic_rings': 0,
            'heavy_atoms': 0,
            'formal_charge': 0
        }
    
    return properties

def get_activity_cliff_summary(cliff_data, activity_col=None):
    """
    Activity Cliff 데이터의 요약 정보를 생성합니다.
    분류형('activity')과 숫자형('pIC50') 활성도 정보를 모두 포함하여 외부 모듈과의 호환성을 완벽하게 보장합니다.
    """
    mol1_info, mol2_info = cliff_data['mol_1'], cliff_data['mol_2']
    high_props, low_props = cliff_data.get('mol1_properties', {}), cliff_data.get('mol2_properties', {})

    # 숫자형 활성도 컬럼이 있을 경우에만 고/저 활성 비교
    if activity_col and pd.api.types.is_numeric_dtype(pd.Series([mol1_info.get(activity_col), mol2_info.get(activity_col)])):
        if mol1_info.get(activity_col, 0) > mol2_info.get(activity_col, 0):
            high_active, low_active = mol1_info, mol2_info
        else:
            high_active, low_active = mol2_info, mol1_info
            high_props, low_props = cliff_data.get('mol2_properties', {}), cliff_data.get('mol1_properties', {})
    else:
        # 그 외의 경우(예: 정량 분석)는 순서대로 할당
        high_active, low_active = mol1_info, mol2_info

    def _create_compound_summary(compound_info, props):
        """내부 헬퍼 함수: 화합물 요약 정보를 생성"""
        # 'activity' 키: 분석 기준이 된 컬럼의 값을 그대로 사용 (e.g., "Moderately Active")
        activity_display_key = activity_col if activity_col else 'Activity'
        activity_display_value = compound_info.get(activity_display_key)
        
        # 'pic50' 키: 데이터에 'pIC50' 컬럼이 있으면 그 숫자 값을, 없으면 0.0을 사용
        pic50_numeric_value = compound_info.get('pIC50')
        pic50_value = pic50_numeric_value if isinstance(pic50_numeric_value, (int, float)) else 0.0
        
        return {
            'id': compound_info.get('ID'),
            'smiles': compound_info.get('SMILES'),
            'activity': activity_display_value,
            'pic50': pic50_value,
            'properties': props
        }

    summary = {
        'high_activity_compound': _create_compound_summary(high_active, high_props),
        'low_activity_compound': _create_compound_summary(low_active, low_props),
        'cliff_metrics': {
            'similarity': cliff_data.get('similarity'),
            'activity_difference': cliff_data.get('activity_difference'), # find_activity_cliffs에서 계산된 값을 그대로 사용
            'structural_difference_type': cliff_data.get('structural_difference'),
            'is_stereoisomer_pair': cliff_data.get('is_stereoisomer'),
            'same_scaffold': cliff_data.get('same_scaffold'),
            'cliff_score': cliff_data.get('score')
        },
        'property_differences': {
            'mw_diff': abs(high_props.get('molecular_weight', 0) - low_props.get('molecular_weight', 0)),
            'logp_diff': abs(high_props.get('logp', 0) - low_props.get('logp', 0)),
            'tpsa_diff': abs(high_props.get('tpsa', 0) - low_props.get('tpsa', 0)),
            'hbd_diff': abs(high_props.get('hbd', 0) - low_props.get('hbd', 0)),
            'hba_diff': abs(high_props.get('hba', 0) - low_props.get('hba', 0))
        }
    }
    return summary

# --- Phase 1: 데이터 준비 및 탐색 ---
def load_data(df_from_db):
    """
    데이터베이스에서 로드된 데이터프레임을 받아 후처리를 수행합니다.
    'Target'을 포함한 모든 컬럼을 유지합니다.
    """
    try:
        df = df_from_db.copy()

        # 1. pIC50, pKi 컬럼을 찾아 'pIC50'로 통합 (pIC50 우선)
        if "pKi" in df.columns and "pIC50" not in df.columns:
            df.rename(columns={"pKi": "pIC50"}, inplace=True)
        elif "pKi" in df.columns and "pIC50" in df.columns:
            df['pIC50'] = df['pIC50'].fillna(df['pKi'])
            df.drop(columns=['pKi'], inplace=True)
        
        # 2. 필수 컬럼(SMILES) 존재 여부 확인
        if 'SMILES' not in df.columns:
            st.error("오류: 데이터에 'SMILES' 컬럼이 없습니다.")
            return None, []

        # 3. SMILES 표준화 및 유효하지 않은 데이터 제거
        initial_rows = len(df)
        df['SMILES'] = df['SMILES'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None)
        df.dropna(subset=['SMILES'], inplace=True)
        
        invalid_smiles_count = initial_rows - len(df)
        if invalid_smiles_count > 0:
            st.warning(f"경고: {invalid_smiles_count}개의 유효하지 않은 SMILES 데이터가 제외되었습니다.")

        # 4. 분석 가능한 활성 컬럼 목록 반환
        activity_cols = [col for col in ["pIC50", "pKi"] if col in df.columns and pd.to_numeric(df[col], errors='coerce').notna().any()]
        if not activity_cols:
            st.warning("경고: 분석 가능한 숫자형 활성 데이터(pIC50 또는 pKi)가 없습니다.")

        return df, activity_cols

    except Exception as e:
        st.error(f"데이터 후처리 중 오류 발생: {e}")
        return None, []

# --- Phase 2: 핵심 패턴 자동 추출 ---
@st.cache_data
def find_activity_cliffs(df, similarity_threshold, activity_diff_threshold, activity_col='pIC50'):
    """DataFrame에서 Activity Cliff 쌍을 찾고 스코어를 계산하여 정렬합니다."""
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    
    fpgenerator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    df['fp'] = [fpgenerator.GetFingerprint(m) for m in df['mol']]
    
    df['scaffold'] = df['mol'].apply(lambda m: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None)
    
    cliffs = []
    # 데이터프레임의 활성 컬럼을 숫자형으로 변환 (오류 발생 시 NaN으로)
    df[activity_col] = pd.to_numeric(df[activity_col], errors='coerce')
    # 활성 값이 NaN인 행 제거
    df.dropna(subset=[activity_col], inplace=True)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            sim = DataStructs.TanimotoSimilarity(df['fp'].iloc[i], df['fp'].iloc[j])
            if sim >= similarity_threshold:
                act_diff = abs(df[activity_col].iloc[i] - df[activity_col].iloc[j])
                if act_diff >= activity_diff_threshold:
                    score = act_diff * (sim - similarity_threshold) * (1 if df['scaffold'].iloc[i] == df['scaffold'].iloc[j] else 0.5)
                    
                    # Activity Cliff 정보 강화
                    mol1_info = df.iloc[i].to_dict()
                    mol2_info = df.iloc[j].to_dict()
                    
                    # 표준 Isomeric SMILES 추가
                    mol1_info['canonical_smiles'] = canonicalize_smiles(mol1_info['SMILES'])
                    mol2_info['canonical_smiles'] = canonicalize_smiles(mol2_info['SMILES'])
                    
                    # 구조적 차이 키워드 추가
                    try:
                        structural_diff = get_structural_difference_keyword(mol1_info['SMILES'], mol2_info['SMILES'])
                    except:
                        structural_diff = "구조적 차이 분석 불가"
                    
                    # 입체이성질체 여부 확인
                    try:
                        is_stereoisomer = check_stereoisomers(mol1_info['SMILES'], mol2_info['SMILES'])
                    except:
                        is_stereoisomer = False
                    
                    # 분자 특성 계산
                    mol1_props = calculate_molecular_properties(df['mol'].iloc[i])
                    mol2_props = calculate_molecular_properties(df['mol'].iloc[j])
                    
                    cliff_data = {
                        'mol_1': mol1_info, 
                        'mol_2': mol2_info, 
                        'similarity': sim, 
                        'activity_difference': act_diff,
                        'score': score,
                        'structural_difference': structural_diff,
                        'is_stereoisomer': is_stereoisomer,
                        'mol1_properties': mol1_props,
                        'mol2_properties': mol2_props,
                        'same_scaffold': df['scaffold'].iloc[i] == df['scaffold'].iloc[j],
                        'scaffold_1': df['scaffold'].iloc[i],
                        'scaffold_2': df['scaffold'].iloc[j]
                    }
                    
                    cliffs.append(cliff_data)
    
    cliffs.sort(key=lambda x: x['score'], reverse=True)
    return cliffs

def find_quantitative_pairs(df, similarity_threshold, activity_col):
    """
    구조적으로 유사하지만 활성 분류(Activity)가 다른 화합물 쌍을 찾습니다.
    """
    df_quant = df.dropna(subset=['SMILES', 'Activity', activity_col]).copy()
    
    # RDKit 객체 및 지문 생성
    df_quant['mol'] = df_quant['SMILES'].apply(Chem.MolFromSmiles)
    df_quant.dropna(subset=['mol'], inplace=True)
    fpgenerator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    df_quant['fp'] = [fpgenerator.GetFingerprint(m) for m in df_quant['mol']]
    df_quant.reset_index(inplace=True, drop=True)

    pairs = []
    # 모든 쌍을 비교하여 조건에 맞는 쌍을 찾습니다.
    for i in range(len(df_quant)):
        for j in range(i + 1, len(df_quant)):
            sim = DataStructs.TanimotoSimilarity(df_quant.iloc[i]['fp'], df_quant.iloc[j]['fp'])
            if sim >= similarity_threshold and df_quant.iloc[i]['Activity'] != df_quant.iloc[j]['Activity']:
                pairs.append({'mol1_index': i, 'mol2_index': j, 'similarity': sim})

    # 활성 분류 차이 점수 계산 및 정렬
    activity_map = {'Highly Active': 4, 'Moderately Active': 3, 'Weakly Active': 2, 'Inactive': 1}
    for pair in pairs:
        activity1 = df_quant.iloc[pair['mol1_index']]['Activity']
        activity2 = df_quant.iloc[pair['mol2_index']]['Activity']
        score1 = activity_map.get(activity1, 0)
        score2 = activity_map.get(activity2, 0)
        pair['activity_category_diff'] = abs(score1 - score2)
    
    pairs.sort(key=lambda x: x.get('activity_category_diff', 0), reverse=True)
    
    # 나중에 UI에서 원본 데이터를 참조할 수 있도록 처리된 데이터프레임도 함께 반환
    return pairs, df_quant

# --- Phase 3: LLM 기반 해석 및 가설 생성 (도킹 데이터 활용) ---

def generate_hypothesis_cliff(cliff, target_name, api_key, llm_provider, activity_col='pIC50'):  # activity_col은 기본 파라미터로 사용
    if not api_key:
        return "사이드바에 API 키를 입력해주세요.", None

    # 개선된 Activity Cliff 정보 활용
    cliff_summary = get_activity_cliff_summary(cliff)
    high_active = cliff_summary['high_activity_compound']
    low_active = cliff_summary['low_activity_compound']
    metrics = cliff_summary['cliff_metrics']
    prop_diffs = cliff_summary['property_differences']
    
    # 도킹 시뮬레이션 결과 가져오기
    docking_results = get_docking_context(high_active['smiles'], low_active['smiles'], target_name)
    docking1 = docking_results['compound1']
    docking2 = docking_results['compound2']
    
    # 도킹 데이터를 프롬프트에 포함할 형식으로 변환
    docking_prompt_addition = f"""
    
    **도킹 시뮬레이션 결과:**
    
    화합물 A (낮은 활성):
    - 결합 친화도: {docking2['binding_affinity_kcal_mol']} kcal/mol
    - 수소결합: {', '.join(docking2['interaction_fingerprint']['Hydrogenbonds']) if docking2['interaction_fingerprint']['Hydrogenbonds'] else '없음'}
    - 소수성 상호작용: {', '.join(docking2['interaction_fingerprint']['Hydrophobic']) if docking2['interaction_fingerprint']['Hydrophobic'] else '없음'}
    - 물다리: {', '.join(docking2['interaction_fingerprint']['Waterbridges']) if docking2['interaction_fingerprint']['Waterbridges'] else '없음'}
    - 염다리: {', '.join(docking2['interaction_fingerprint']['Saltbridges']) if docking2['interaction_fingerprint']['Saltbridges'] else '없음'}
    - 할로겐결합: {', '.join(docking2['interaction_fingerprint']['Halogenbonds']) if docking2['interaction_fingerprint']['Halogenbonds'] else '없음'}
    - 반데르발스 접촉: {', '.join(docking2['interaction_fingerprint']['VdWContact']) if docking2['interaction_fingerprint']['VdWContact'] else '없음'}
       
    화합물 B (높은 활성):
    - 결합 친화도: {docking1['binding_affinity_kcal_mol']} kcal/mol
    - 수소결합: {', '.join(docking1['interaction_fingerprint']['Hydrogenbonds']) if docking1['interaction_fingerprint']['Hydrogenbonds'] else '없음'}
    - 소수성 상호작용: {', '.join(docking1['interaction_fingerprint']['Hydrophobic']) if docking1['interaction_fingerprint']['Hydrophobic'] else '없음'}
    - 물다리: {', '.join(docking1['interaction_fingerprint']['Waterbridges']) if docking1['interaction_fingerprint']['Waterbridges'] else '없음'}
    - 염다리: {', '.join(docking1['interaction_fingerprint']['Saltbridges']) if docking1['interaction_fingerprint']['Saltbridges'] else '없음'}
    - 할로겐결합: {', '.join(docking1['interaction_fingerprint']['Halogenbonds']) if docking1['interaction_fingerprint']['Halogenbonds'] else '없음'}
    - 반데르발스 접촉: {', '.join(docking1['interaction_fingerprint']['VdWContact']) if docking1['interaction_fingerprint']['VdWContact'] else '없음'}
    
    **도킹 결과 기반 가설 생성 요청:**
    위의 도킹 시뮬레이션 결과를 바탕으로, 두 화합물의 결합 친화도 차이와 상호작용 패턴의 차이가 어떻게 활성도 차이로 이어지는지 설명해주세요.
    특히 사라지거나 새로 형성된 상호작용이 활성에 미치는 영향을 중점적으로 분석해주세요.
    """
    
    context_info = docking_results  # 도킹 결과를 context_info로 사용
    
    # 입체이성질체 분석
    if metrics['is_stereoisomer_pair']:
        prompt_addition = (
            "\n\n**중요 지침:** 이 두 화합물은 동일한 2D 구조를 가진 입체이성질체(stereoisomer)입니다. "
            f"Tanimoto 유사도({metrics['similarity']:.3f})가 매우 높지만 1.00이 아닌 이유는 바로 이 3D 구조의 차이 때문입니다. "
            "SMILES 문자열의 '@' 또는 '@@' 표기를 주목하여, 3D 공간 배열(입체화학)의 차이가 어떻게 이러한 활성 차이를 유발하는지 집중적으로 설명해주세요."
        )
    else:
        prompt_addition = f"\n\n**구조적 차이 유형:** {metrics['structural_difference_type']}"
    
    # 물리화학적 특성 차이 정보 추가
    props_info = f"""
    **물리화학적 특성 차이:**
    - 분자량 차이: {prop_diffs['mw_diff']:.2f} Da
    - LogP 차이: {prop_diffs['logp_diff']:.2f}
    - TPSA 차이: {prop_diffs['tpsa_diff']:.2f} Ų
    - 수소결합 공여체 차이: {prop_diffs['hbd_diff']}개
    - 수소결합 수용체 차이: {prop_diffs['hba_diff']}개
    - 같은 스캐폴드: {'예' if metrics['same_scaffold'] else '아니오'}
    """

    user_prompt = f"""
    **분석 대상:**
    - **화합물 A (낮은 활성):**
      - ID: {low_active['id']}
      - 표준 SMILES: {low_active['smiles']}
      - 활성도 (pIC50): {low_active['pic50']}
      - 분자량: {low_active['properties']['molecular_weight']:.2f} Da
      - LogP: {low_active['properties']['logp']:.2f}
      - TPSA: {low_active['properties']['tpsa']:.2f} Ų
    
    - **화합물 B (높은 활성):**
      - ID: {high_active['id']}
      - 표준 SMILES: {high_active['smiles']}
      - 활성도 (pIC50): {high_active['pic50']}
      - 분자량: {high_active['properties']['molecular_weight']:.2f} Da
      - LogP: {high_active['properties']['logp']:.2f}
      - TPSA: {high_active['properties']['tpsa']:.2f} Ų
    
    **Activity Cliff 메트릭:**
    - Tanimoto 유사도: {metrics['similarity']:.3f}
    - 활성도 차이 (ΔpIC50): {metrics['activity_difference']}
    - Cliff 점수: {metrics['cliff_score']:.3f}
    
    {props_info}
    
    **분석 요청:**
    두 화합물은 구조적으로 매우 유사하지만, 활성도에서 큰 차이를 보이는 전형적인 'Activity Cliff' 사례입니다.
    위의 상세한 분자 정보와 물리화학적 특성을 종합적으로 분석하여, **이러한 활성도 차이를 유발하는** 핵심적인 구조적 요인과 그 메커니즘에 대한 과학적 가설을 제시해주세요.
    특히 타겟 단백질 {target_name}와의 상호작용 관점에서 설명해주세요.{prompt_addition}{docking_prompt_addition}
    """

    try:
        if llm_provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            system_prompt = "당신은 숙련된 신약 개발 화학자입니다. 두 화합물의 구조-활성 관계(SAR)와 도킹 시뮬레이션 결과에 대한 분석을 요청받았습니다. 도킹 결과에서 나타난 상호작용 패턴의 차이를 중심으로, 분석 결과를 전문가의 관점에서 명확하고 간결하게 마크다운 형식으로 작성해주세요."
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
            return response.choices[0].message.content, context_info
        
        elif llm_provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            full_prompt = "당신은 숙련된 신약 개발 화학자입니다. 도킹 시뮬레이션 결과를 바탕으로 다음 요청에 대해 전문가의 관점에서 명확하고 간결하게 마크다운 형식으로 답변해주세요.\n\n" + user_prompt
            response = model.generate_content(full_prompt)
            return response.text, context_info
    except Exception as e:
        return f"{llm_provider} API 호출 중 오류 발생: {e}", None
    return "알 수 없는 LLM 공급자입니다.", None


def generate_hypothesis_quantitative(mol1, mol2, similarity, target_name, api_key, llm_provider):
    """정량(분류) 데이터에 대한 가설을 생성합니다."""
    if not api_key:
        return "사이드바에 API 키를 입력해주세요.", None
    
    # 도킹 시뮬레이션 결과 가져오기
    docking_results = get_docking_context(mol1['SMILES'], mol2['SMILES'], target_name)
    docking1 = docking_results['compound1']
    docking2 = docking_results['compound2']
    
    docking_prompt = f"""
    \n\n**도킹 시뮬레이션 결과:**
    화합물 1: 결합 친화도 {docking1['binding_affinity_kcal_mol']} kcal/mol
    화합물 2: 결합 친화도 {docking2['binding_affinity_kcal_mol']} kcal/mol
    
    도킹 결과를 바탕으로 활성 분류 차이를 설명해주세요.
    """
    
    user_prompt = f"""
    **분석 대상:** 타겟 단백질 {target_name}
    - **화합물 1:** ID {mol1['ID']}, SMILES {mol1['SMILES']}, 활성 분류: {mol1['Activity']}
    - **화합물 2:** ID {mol2['ID']}, SMILES {mol2['SMILES']}, 활성 분류: {mol2['Activity']}
    **유사도:** {similarity}
    **분석 요청:** 두 화합물은 구조적으로 매우 유사하지만 활성 분류가 다릅니다. 이 차이를 유발하는 구조적 요인에 대한 과학적 가설을 제시해주세요.{docking_prompt}
    """
    
    context_info = docking_results
    return call_llm(user_prompt, api_key, llm_provider), context_info


def call_llm(user_prompt, api_key, llm_provider):
    """LLM API를 호출하는 공통 함수입니다."""
    try:
        system_prompt = "당신은 숙련된 신약 개발 화학자입니다. 도킹 시뮬레이션 결과와 구조-활성 관계(SAR) 데이터를 바탕으로 명확하고 간결한 분석 리포트를 마크다운 형식으로 작성해주세요."
        if llm_provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
            return response.choices[0].message.content
        elif llm_provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(system_prompt + "\n\n" + user_prompt)
            return response.text
    except Exception as e:
        return f"{llm_provider} API 호출 중 오류 발생: {e}"


# --- Phase 4: 시각화 ---
def draw_highlighted_pair(smiles1, smiles2):
    """두 분자의 공통 구조를 기준으로 정렬하고 차이점을 하이라이팅하여 SVG 이미지 쌍으로 반환합니다."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if not mol1 or not mol2:
        return None, None

    # 최대 공통 부분구조(MCS)를 찾고, 이를 기준으로 분자 정렬
    mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=5)
    if mcs_result.numAtoms > 0:
        patt = Chem.MolFromSmarts(mcs_result.smartsString)
        AllChem.Compute2DCoords(patt)
        AllChem.GenerateDepictionMatching2DStructure(mol1, patt)
        AllChem.GenerateDepictionMatching2DStructure(mol2, patt)
        hit_ats1 = mol1.GetSubstructMatch(patt)
        hit_ats2 = mol2.GetSubstructMatch(patt)
    else:
        # MCS가 없을 경우, 기본 2D 좌표 생성
        rdDepictor.Compute2DCoords(mol1)
        rdDepictor.Compute2DCoords(mol2)
        hit_ats1, hit_ats2 = tuple(), tuple()

    # 하이라이트할 원자 리스트 (공통 구조가 아닌 부분)
    highlight1 = list(set(range(mol1.GetNumAtoms())) - set(hit_ats1))
    highlight2 = list(set(range(mol2.GetNumAtoms())) - set(hit_ats2))
    
    # 입체이성질체의 경우 키랄 중심도 하이라이트
    is_stereoisomer = (Chem.MolToSmiles(mol1, isomericSmiles=False) == Chem.MolToSmiles(mol2, isomericSmiles=False)) and (smiles1 != smiles2)
    if is_stereoisomer:
        chiral_centers1 = Chem.FindMolChiralCenters(mol1, includeUnassigned=True)
        chiral_centers2 = Chem.FindMolChiralCenters(mol2, includeUnassigned=True)
        highlight1.extend([c[0] for c in chiral_centers1])
        highlight2.extend([c[0] for c in chiral_centers2])

    def _mol_to_svg(mol, highlight_atoms):
        d = rdMolDraw2D.MolDraw2DSVG(400, 400)
        d.drawOptions().addStereoAnnotation = True
        d.drawOptions().clearBackground = False
        d.DrawMolecule(mol, highlightAtoms=list(set(highlight_atoms)))
        d.FinishDrawing()
        return d.GetDrawingText()

    svg1 = _mol_to_svg(mol1, highlight1)
    svg2 = _mol_to_svg(mol2, highlight2)
    
    return svg1, svg2

def get_real_docking_context(smiles1, smiles2, target_name="6G6K"):
    """
    실제 docking-pipeline을 사용하여 두 화합물의 도킹 시뮬레이션을 수행
    """
    try:
        # 절대 경로로 docking_pipeline 모듈 임포트 시도
        import sys
        import os
        
        # 현재 작업 디렉토리에서 docking_pipeline 찾기
        current_dir = os.getcwd()
        docking_path = os.path.join(current_dir, 'docking_pipeline')
        
        if not os.path.exists(docking_path):
            # 상위 디렉토리에서도 찾아보기
            parent_dir = os.path.dirname(current_dir)
            docking_path = os.path.join(parent_dir, 'docking_pipeline')
            
        if not os.path.exists(docking_path):
            raise ImportError(f"docking_pipeline 디렉토리를 찾을 수 없습니다. 경로: {docking_path}")
        
        # 경로 추가 (프로젝트 루트 경로)
        project_root = os.path.dirname(docking_path)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # 도킹 파이프라인 모듈 임포트 (패키지 경로 명확화)
        from docking_pipeline.main import run_full_docking_pipeline
        
        # 진행 상황 표시
        if 'st' in globals():
            progress_bar = st.progress(0, text="🧬 실제 도킹 시뮬레이션 준비 중...")

        results = {}
        
        # 두 화합물 순차 도킹 (안정성을 위해 병렬 대신 순차 처리)
        if 'st' in globals():
            progress_bar.progress(0.2, text="⚗️ 화합물 1 도킹 시뮬레이션 중...")
        
        result1 = run_full_docking_pipeline(smiles1, target_name.upper(), f"{target_name}_compound1")
        results['compound1'] = result1
        
        if 'st' in globals():
            progress_bar.progress(0.6, text="⚗️ 화합물 2 도킹 시뮬레이션 중...")
        
        result2 = run_full_docking_pipeline(smiles2, target_name.upper(), f"{target_name}_compound2")
        results['compound2'] = result2
        
        if 'st' in globals():
            progress_bar.progress(1.0, text="🎯 실제 도킹 분석 완료!")
            time.sleep(0.5)
            progress_bar.empty()

        # 기존 형식으로 변환
        return convert_docking_results_to_context(results, smiles1, smiles2, target_name)

    except ImportError as e:
        raise ImportError(f"도킹 파이프라인 모듈을 로드할 수 없습니다: {e}")
    except Exception as e:
        raise Exception(f"도킹 시뮬레이션 실행 중 오류: {e}")
    

# 기존 get_docking_context 함수를 실제 도킹으로 대체
def get_docking_context(smiles1, smiles2, target_name="6G6K"):
    """실제 도킹 파이프라인을 우선 사용하고, 실패 시에만 폴백"""
    
    # 디버깅 정보 (Streamlit 환경에서만)
    if 'st' in globals():
        st.info("🔄 실제 도킹 파이프라인 연동 시도 중...")
    
    # 1단계: 실제 도킹 파이프라인 시도
    try:
        return get_real_docking_context(smiles1, smiles2, target_name)
    except ImportError as e:
        if 'st' in globals():
            st.warning(f"⚠️ 도킹 파이프라인 모듈 로드 실패: {str(e)}")
            st.error("❌ 실제 도킹 데이터를 사용할 수 없습니다. 시스템 관리자에게 문의하세요.")
        raise ImportError(f"도킹 파이프라인이 필요합니다: {e}")
    except Exception as e:
        if 'st' in globals():
            st.error(f"❌ 도킹 시뮬레이션 실행 오류: {str(e)}")
        raise Exception(f"도킹 시뮬레이션 실패: {e}")
    
def convert_docking_results_to_context(results, smiles1, smiles2, target_name):
    import hashlib
    import os
    def get_result_json_path(smiles, pdb_id, outputs_dir=None):
        smiles_id = hashlib.md5(smiles.encode('utf-8')).hexdigest()[:8]
        if outputs_dir is None:
            # 기본적으로 docking_pipeline/outputs 폴더 사용
            current_dir = os.path.dirname(os.path.abspath(__file__))
            outputs_dir = os.path.join(current_dir, 'docking_pipeline', 'outputs')
        return os.path.join(outputs_dir, f"pr_{smiles_id}__{pdb_id}.json")

    """
    실제 docking-pipeline 결과를 기존 shared_context 형식으로 변환
    """

    # results 대신 JSON 파일 경로를 받아서 처리하도록 변경
    def extract_interactions_from_json(json_path):
        import json
        import re
        interactions = {
            "Hydrogenbonds": [],
            "Hydrophobic": [],
            "Waterbridges": [],
            "Saltbridges": [],
            "Halogenbonds": [],
            "VdWContact": []
        }
        with open(json_path, 'r') as f:
            data = json.load(f)
        ifp = data.get('prolif', {}).get('ifp_dataframe', {})
        type_map = {
            "hydrogenbonds": "Hydrogenbonds",
            "hydrogenbond": "Hydrogenbonds",
            "hydrophobic": "Hydrophobic",
            "waterbridges": "Waterbridges",
            "saltbridges": "Saltbridges",
            "halogenbonds": "Halogenbonds",
            "vdwcontact": "VdWContact"
        }
        for key, pose_dict in ifp.items():
            parts = key.split('|')
            if len(parts) < 3:
                continue
            residue = parts[1]  # 예: "LEU210.B"
            # residue에서 숫자+문자만 추출
            residue_name_match = re.match(r"([A-Z]+[0-9]+)", residue)
            residue_name = residue_name_match.group(1) if residue_name_match else residue
            interaction_type = parts[-1].lower()
            mapped_type = type_map.get(interaction_type, None)
            if mapped_type and any(v for v in pose_dict.values()):
                interactions[mapped_type].append(residue_name)
        return interactions

    # 예시: compound1, compound2 각각의 JSON 파일 경로를 받아서 처리
    # 실제 사용 시에는 파일 경로를 인자로 전달해야 함
    # 아래는 예시 경로
    # 경로가 없거나 None이면 직접 생성
    compound1_json = results.get('compound1_json')
    compound2_json = results.get('compound2_json')
    if not compound1_json:
        compound1_json = get_result_json_path(smiles1, target_name)
    if not compound2_json:
        compound2_json = get_result_json_path(smiles2, target_name)
    # 파일 존재 여부 체크
    if not (os.path.exists(compound1_json) and os.path.exists(compound2_json)):
        raise Exception(f"도킹 결과 JSON 파일을 찾을 수 없습니다: {compound1_json}, {compound2_json}")
    import json
    with open(compound1_json, 'r') as f1:
        compound1_result = json.load(f1)
    with open(compound2_json, 'r') as f2:
        compound2_result = json.load(f2)

    context_result = {
        "compound1": {
            "smiles": compound1_result.get('smiles', ''),
            "pdb_id": compound1_result.get('pdb_id', ''),
            "binding_affinity_kcal_mol": compound1_result.get('binding_affinity_kcal_mol', -7.5),
            "interaction_fingerprint": extract_interactions_from_json(compound1_json)
        },
        "compound2": {
            "smiles": compound2_result.get('smiles', ''),
            "pdb_id": compound2_result.get('pdb_id', ''),
            "binding_affinity_kcal_mol": compound2_result.get('binding_affinity_kcal_mol', -7.0),
            "interaction_fingerprint": extract_interactions_from_json(compound2_json)
        },
        "_metadata": {
            "data_source": "real_docking_pipeline",
            "timestamp": time.time(),
            "vina_version": "AutoDock Vina",
            "prolif_analysis": "enabled",
            "note": "실제 도킹 파이프라인 결과 사용"
        }
    }

    # Streamlit 환경에서 성공 메시지 표시
    if 'st' in globals():
        st.success("🎯 실제 AutoDock Vina + ProLIF 결과를 사용합니다!")
        
        # 디버깅 정보 표시 (선택적)
        with st.expander("🔍 도킹 결과 변환 디버깅 정보", expanded=False):
            st.write("**원본 결과 구조:**")
            st.json({
                "compound1_keys": list(compound1_result.keys()) if compound1_result else [],
                "compound2_keys": list(compound2_result.keys()) if compound2_result else [],
                "prolif_structure": str(type(compound1_result.get('prolif', {}).get('ifp_dataframe'))) if compound1_result.get('prolif') else "None"
            })
            
            st.write("**변환된 상호작용:**")
            st.json({
                "compound1_interactions": context_result["compound1"]["interaction_fingerprint"],
                "compound2_interactions": context_result["compound2"]["interaction_fingerprint"]
            })

    return context_result

