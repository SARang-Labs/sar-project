import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdFMCS, rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
import google.generativeai as genai
from openai import OpenAI
import requests
import xml.etree.ElementTree as ET
import joblib
import json
import os
from urllib.parse import quote 


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
    
    # 간단한 작용기 이름으로 변환 (예시)
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
    except Exception as e:
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
    분류형('activity')과 숫자형('pki') 활성도 정보를 모두 포함하여 외부 모듈과의 호환성을 완벽하게 보장합니다.
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
        
        # 'pki' 키: 데이터에 'pKi' 컬럼이 있으면 그 숫자 값을, 없으면 0.0을 사용
        pki_numeric_value = compound_info.get('pKi')
        pki_value = pki_numeric_value if isinstance(pki_numeric_value, (int, float)) else 0.0
        
        return {
            'id': compound_info.get('ID'),
            'smiles': compound_info.get('SMILES'),
            'activity': activity_display_value,
            'pki': pki_value,
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
@st.cache_data
def load_data(uploaded_file):
    """CSV 파일을 로드하고, pKi와 pIC50 컬럼을 지능적으로 통합하여 데이터를 전처리합니다."""
    try:
        df = pd.read_csv(uploaded_file)

        # --- 컬럼 이름 확인 방식 통일 ---
        
        # 1. 유연한 방식으로 pKi 및 pIC50 관련 컬럼 이름들을 먼저 찾습니다.
        pki_cols = [col for col in df.columns if 'pKi' in col]
        pic50_cols = [col for col in df.columns if 'pIC50' in col]

        # 2. 찾은 컬럼 이름을 기준으로 pKi 데이터를 보정/생성합니다.
        if pki_cols and pic50_cols:
            pki_col_name = pki_cols[0]
            pic50_col_name = pic50_cols[0]
            
            df[pki_col_name] = pd.to_numeric(df[pki_col_name], errors='coerce')
            df[pic50_col_name] = pd.to_numeric(df[pic50_col_name], errors='coerce')
            
            df[pki_col_name].replace(0, np.nan, inplace=True)
            df[pki_col_name].fillna(df[pic50_col_name], inplace=True)
            
            # 기준이 되는 pKi 컬럼 이름을 'pKi'로 통일합니다.
            if pki_col_name != 'pKi':
                df.rename(columns={pki_col_name: 'pKi'}, inplace=True)
            st.info("Info: 'pKi'와 'pIC50' 값을 지능적으로 병합했습니다.")

        elif not pki_cols and pic50_cols:
            pic50_col_name = pic50_cols[0]
            # pIC50 컬럼을 'pKi'라는 이름으로 생성합니다.
            df['pKi'] = df[pic50_col_name]
            st.info(f"Info: '{pic50_col_name}' 컬럼을 'pKi'로 변환했습니다.")

        if 'SMILES' not in df.columns:
            st.error("오류: CSV 파일에 'SMILES' 컬럼이 반드시 포함되어야 합니다.")
            return None, []
        if 'ID' not in df.columns:
            df.insert(0, 'ID', [f"Mol_{i+1}" for i in range(len(df))])
            st.info("Info: 'ID' 컬럼이 없어 자동으로 생성되었습니다.")
        
        activity_cols = sorted([col for col in df.columns if 'pKi' in col or 'pIC50' in col], reverse=True)
        if not activity_cols:
            st.error("오류: CSV 파일에 'pKi' 또는 'pIC50'을 포함하는 활성 데이터 컬럼이 없습니다.")
            return None, []
            
        df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)
        df.dropna(subset=['SMILES'], inplace=True)
        for col in activity_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, activity_cols
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return None, []

# --- Phase 2: 핵심 패턴 자동 추출 ---
@st.cache_data
def find_activity_cliffs(df, similarity_threshold, activity_diff_threshold, activity_col='pKi'):
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
                    
                    # 구조적 차이 키워드 추가 (안전한 처리)
                    try:
                        structural_diff = get_structural_difference_keyword(mol1_info['SMILES'], mol2_info['SMILES'])
                    except:
                        structural_diff = "구조적 차이 분석 불가"
                    
                    # 입체이성질체 여부 확인 (안전한 처리)
                    try:
                        is_stereoisomer = check_stereoisomers(mol1_info['SMILES'], mol2_info['SMILES'])
                    except:
                        is_stereoisomer = False
                    
                    # 분자 특성 계산 (안전한 처리)
                    mol1_props = calculate_molecular_properties(df['mol'].iloc[i])
                    mol2_props = calculate_molecular_properties(df['mol'].iloc[j])
                    
                    cliff_data = {
                        'mol_1': mol1_info, 
                        'mol_2': mol2_info, 
                        'similarity': sim, 
                        'activity_diff': act_diff, 
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

# --- Phase 3: LLM 기반 해석 및 가설 생성 (RAG 적용) ---

@st.cache_data
def search_pubmed_for_context(smiles1, smiles2, target_name, max_results=1):
    def fetch_articles(search_term):
        try:
            esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {'db': 'pubmed', 'term': search_term, 'retmax': max_results, 'sort': 'relevance'}
            response = requests.get(esearch_url, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            id_list = [elem.text for elem in root.findall('.//Id')]
            if not id_list: return None

            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {'db': 'pubmed', 'id': ",".join(id_list), 'retmode': 'xml'}
            response = requests.get(efetch_url, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            
            article = root.find('.//PubmedArticle')
            if article:
                title = article.findtext('.//ArticleTitle', 'No title found')
                abstract = " ".join([p.text for p in article.findall('.//Abstract/AbstractText') if p.text])
                pmid = article.findtext('.//PMID', '')
                if not abstract: abstract = 'No abstract found'
                return {"title": title, "abstract": abstract, "pmid": pmid, "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
        except Exception:
            return None
        return None

    diff_keyword = get_structural_difference_keyword(smiles1, smiles2)
    if diff_keyword and (result := fetch_articles(f'("{target_name}"[Title/Abstract]) AND ("{diff_keyword}"[Title/Abstract])')):
        return result
    
    return fetch_articles(f'("{target_name}"[Title/Abstract]) AND ("structure activity relationship"[Title/Abstract])')


def generate_hypothesis_cliff(cliff, target_name, api_key, llm_provider, activity_col='pKi'):
    if not api_key:
        return "사이드바에 API 키를 입력해주세요.", None

    # 개선된 Activity Cliff 정보 활용
    cliff_summary = get_activity_cliff_summary(cliff)
    high_active = cliff_summary['high_activity_compound']
    low_active = cliff_summary['low_activity_compound']
    metrics = cliff_summary['cliff_metrics']
    prop_diffs = cliff_summary['property_differences']
    
    context_info = search_pubmed_for_context(high_active['smiles'], low_active['smiles'], target_name)
    rag_prompt_addition = f"\n\n**참고 문헌 정보:**\n- 제목: {context_info['title']}\n- 초록: {context_info['abstract']}\n\n위 참고 문헌의 내용을 바탕으로 가설을 생성해주세요." if context_info else ""
    
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
      - 활성도 (pKi): {low_active['pki']}
      - 분자량: {low_active['properties']['molecular_weight']:.2f} Da
      - LogP: {low_active['properties']['logp']:.2f}
      - TPSA: {low_active['properties']['tpsa']:.2f} Ų
    
    - **화합물 B (높은 활성):**
      - ID: {high_active['id']}
      - 표준 SMILES: {high_active['smiles']}
      - 활성도 (pKi): {high_active['pki']}
      - 분자량: {high_active['properties']['molecular_weight']:.2f} Da
      - LogP: {high_active['properties']['logp']:.2f}
      - TPSA: {high_active['properties']['tpsa']:.2f} Ų
    
    **Activity Cliff 메트릭:**
    - Tanimoto 유사도: {metrics['similarity']:.3f}
    - 활성도 차이 (ΔpKi): {metrics['activity_difference']}
    - Cliff 점수: {metrics['cliff_score']:.3f}
    
    {props_info}
    
    **분석 요청:**
    두 화합물은 구조적으로 매우 유사하지만, 활성도에서 큰 차이를 보이는 전형적인 'Activity Cliff' 사례입니다.
    위의 상세한 분자 정보와 물리화학적 특성을 종합적으로 분석하여, **이러한 활성도 차이를 유발하는** 핵심적인 구조적 요인과 그 메커니즘에 대한 과학적 가설을 제시해주세요.
    특히 타겟 단백질 {target_name}와의 상호작용 관점에서 설명해주세요.{prompt_addition}{rag_prompt_addition}
    """

    try:
        if llm_provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            system_prompt = "당신은 숙련된 신약 개발 화학자입니다. 두 화합물의 구조-활성 관계(SAR)에 대한 분석을 요청받았습니다. 분석 결과를 전문가의 관점에서 명확하고 간결하게 마크다운 형식으로 작성해주세요."
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
            return response.choices[0].message.content, context_info
        
        elif llm_provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            full_prompt = "당신은 숙련된 신약 개발 화학자입니다. 다음 요청에 대해 전문가의 관점에서 명확하고 간결하게 마크다운 형식으로 답변해주세요.\n\n" + user_prompt
            response = model.generate_content(full_prompt)
            return response.text, context_info
    except Exception as e:
        return f"{llm_provider} API 호출 중 오류 발생: {e}", None
    return "알 수 없는 LLM 공급자입니다.", None


def generate_hypothesis_quantitative(mol1, mol2, similarity, target_name, api_key, llm_provider):
    """정량(분류) 데이터에 대한 가설을 생성합니다."""
    if not api_key:
        return "사이드바에 API 키를 입력해주세요.", None
    
    context_info = search_pubmed_for_context(mol1['SMILES'], mol2['SMILES'], target_name)
    rag_prompt_addition = f"\n\n**참고 문헌 정보:**\n- 제목: {context_info['title']}\n- 초록: {context_info['abstract']}\n\n위 문헌을 바탕으로 가설을 생성해주세요." if context_info else ""
    
    user_prompt = f"""
    **분석 대상:** 타겟 단백질 {target_name}
    - **화합물 1:** ID {mol1['ID']}, SMILES {mol1['SMILES']}, 활성 분류: {mol1['Activity']}
    - **화합물 2:** ID {mol2['ID']}, SMILES {mol2['SMILES']}, 활성 분류: {mol2['Activity']}
    **유사도:** {similarity}
    **분석 요청:** 두 화합물은 구조적으로 매우 유사하지만 활성 분류가 다릅니다. 이 차이를 유발하는 구조적 요인에 대한 과학적 가설을 제시해주세요.{rag_prompt_addition}
    """
    
    return call_llm(user_prompt, api_key, llm_provider), context_info


def call_llm(user_prompt, api_key, llm_provider):
    """LLM API를 호출하는 공통 함수입니다."""
    try:
        system_prompt = "당신은 숙련된 신약 개발 화학자입니다. 주어진 데이터를 바탕으로 구조-활성 관계(SAR)에 대한 명확하고 간결한 분석 리포트를 마크다운 형식으로 작성해주세요."
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