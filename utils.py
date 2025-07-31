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
    if diff1_mol:
        fragments.extend(Chem.MolToSmiles(frag) for frag in Chem.GetMolFrags(diff1_mol, asMols=True))
    if diff2_mol:
        fragments.extend(Chem.MolToSmiles(frag) for frag in Chem.GetMolFrags(diff2_mol, asMols=True))
    
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

# --- Phase 1: 데이터 준비 및 탐색 ---
st.cache_data
def load_data(uploaded_file):
    """업로드된 CSV 파일을 Pandas DataFrame으로 로드하고 전처리합니다."""
    try:
        # CSV 파일의 pKi 컬럼을 명시적으로 숫자형으로 읽어오도록 처리
        df = pd.read_csv(uploaded_file)
        required_cols = ['ID', 'SMILES', 'pKi']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV 파일은 {', '.join(required_cols)} 컬럼을 포함해야 합니다.")
            return None
        
        df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)
        df.dropna(subset=['SMILES'], inplace=True) # 유효하지 않은 SMILES 제거
        df['pKi'] = pd.to_numeric(df['pKi'], errors='coerce')
        df.dropna(subset=['pKi'], inplace=True) # pKi 변환 실패한 행 제거

        return df
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return None

# --- Phase 2: 핵심 패턴 자동 추출 ---
st.cache_data
def find_activity_cliffs(df, similarity_threshold, activity_diff_threshold):
    """DataFrame에서 Activity Cliff 쌍을 찾고 스코어를 계산하여 정렬합니다."""
    df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    
    fpgenerator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    df['fp'] = [fpgenerator.GetFingerprint(m) for m in df['mol']]
    
    df['scaffold'] = df['mol'].apply(lambda m: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) if m else None)
    
    cliffs = []
    # 데이터프레임의 pKi 컬럼을 숫자형으로 변환 (오류 발생 시 NaN으로)
    df['pKi'] = pd.to_numeric(df['pKi'], errors='coerce')
    # pKi 값이 NaN인 행 제거
    df.dropna(subset=['pKi'], inplace=True)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            sim = DataStructs.TanimotoSimilarity(df['fp'].iloc[i], df['fp'].iloc[j])
            if sim >= similarity_threshold:
                act_diff = abs(df['pKi'].iloc[i] - df['pKi'].iloc[j])
                if act_diff >= activity_diff_threshold:
                    score = act_diff * (sim - similarity_threshold) * (1 if df['scaffold'].iloc[i] == df['scaffold'].iloc[j] else 0.5)
                    # .to_dict()를 사용하여 Series를 딕셔너리로 변환
                    mol1_info = df.iloc[i].to_dict()
                    mol2_info = df.iloc[j].to_dict()
                    cliffs.append({'mol_1': mol1_info, 'mol_2': mol2_info, 'similarity': sim, 'activity_diff': act_diff, 'score': score})
    
    cliffs.sort(key=lambda x: x['score'], reverse=True)
    return cliffs

# --- Phase 3: LLM 기반 해석 및 가설 생성 (RAG 적용) ---

st.cache_data
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


def generate_hypothesis(cliff, target_name, api_key, llm_provider):
    if not api_key:
        return "사이드바에 API 키를 입력해주세요.", None

    mol1_info, mol2_info = cliff['mol_1'], cliff['mol_2']
    compound_a, compound_b = (mol1_info, mol2_info) if mol1_info['pKi'] < mol2_info['pKi'] else (mol2_info, mol1_info)
    
    context_info = search_pubmed_for_context(compound_a['SMILES'], compound_b['SMILES'], target_name)
    rag_prompt_addition = f"\n\n**참고 문헌 정보:**\n- 제목: {context_info['title']}\n- 초록: {context_info['abstract']}\n\n위 참고 문헌의 내용을 바탕으로 가설을 생성해주세요." if context_info else ""
    
    is_stereoisomer = (Chem.MolToSmiles(Chem.MolFromSmiles(compound_a['SMILES']), isomericSmiles=False) == Chem.MolToSmiles(Chem.MolFromSmiles(compound_b['SMILES']), isomericSmiles=False)) and (compound_a['SMILES'] != compound_b['SMILES'])
    
    # FIX: 입체이성질체일 경우, AI가 오해하지 않도록 프롬프트를 명확하게 수정
    if is_stereoisomer:
        prompt_addition = (
            "\n\n**중요 지침:** 이 두 화합물은 동일한 2D 구조를 가진 입체이성질체(stereoisomer)입니다. "
            f"Tanimoto 유사도({cliff['similarity']:.3f})가 매우 높지만 1.00이 아닌 이유는 바로 이 3D 구조의 차이 때문입니다. "
            "SMILES 문자열의 '@' 또는 '@@' 표기를 주목하여, 3D 공간 배열(입체화학)의 차이가 어떻게 이러한 활성 차이를 유발하는지 집중적으로 설명해주세요."
        )
    else:
        prompt_addition = ""

    user_prompt = f"""
    **분석 대상:**
    - **화합물 A (낮은 활성):**
      - ID: {compound_a['ID']}
      - SMILES: {compound_a['SMILES']}
      - 활성도 (pKi): {compound_a['pKi']:.2f}
    - **화합물 B (높은 활성):**
      - ID: {compound_b['ID']}
      - SMILES: {compound_b['SMILES']}
      - 활성도 (pKi): {compound_b['pKi']:.2f}
    **분석 요청:**
    두 화합물은 구조적으로 매우 유사하지만(Tanimoto 유사도: {cliff['similarity']:.3f}), 활성도에서 큰 차이(pKi 차이: {cliff['activity_diff']:.2f})를 보입니다.
    이러한 'Activity Cliff' 현상을 유발하는 핵심적인 구조적 차이점을 찾아내고, 그 차이가 어떻게 활성도 증가로 이어졌는지에 대한 화학적 가설을 설명해주세요.{prompt_addition}{rag_prompt_addition}
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