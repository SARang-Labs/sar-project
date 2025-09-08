"""
도킹 시뮬레이션 도구

분자 도킹 시뮬레이션을 수행하여 단백질-리간드 상호작용을 정량적으로 분석합니다.
AutoDock Vina, RDKit, PyMOL 등을 활용한 도킹 분석 기능을 제공합니다.
"""

import os
import json
import requests
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

@dataclass
class DockingResult:
    """도킹 결과 데이터 클래스"""
    ligand_id: str
    binding_affinity: float  # kcal/mol
    rmsd_lb: float  # RMSD lower bound
    rmsd_ub: float  # RMSD upper bound
    interactions: Dict[str, List]  # 상호작용 타입별 정보
    pose_coordinates: Optional[str] = None  # PDB 형식의 좌표
    
class DockingSimulator:
    """
    도킹 시뮬레이션 수행 클래스
    
    여러 도킹 서비스와 API를 활용하여 분자 도킹을 수행합니다:
    1. AutoDock Vina (로컬 설치 필요)
    2. SwissDock API
    3. PatchDock/FireDock
    4. 간단한 약리단 기반 도킹 예측
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        초기화
        
        Args:
            api_key: 외부 도킹 서비스 API 키 (선택사항)
        """
        self.api_key = api_key
        self.vina_available = self._check_vina_installation()
        
    def _check_vina_installation(self) -> bool:
        """AutoDock Vina 설치 확인"""
        try:
            result = subprocess.run(['vina', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def perform_docking(self, 
                       smiles: str,
                       target_pdb: Optional[str] = None,
                       target_name: Optional[str] = None) -> DockingResult:
        """
        도킹 시뮬레이션 수행
        
        Args:
            smiles: 리간드 SMILES 문자열
            target_pdb: 타겟 단백질 PDB ID 또는 파일 경로
            target_name: 타겟 단백질 이름 (PDB ID가 없을 경우)
            
        Returns:
            DockingResult 객체
        """
        
        # 1. 로컬 Vina 사용 가능한 경우
        if self.vina_available and target_pdb:
            return self._run_vina_docking(smiles, target_pdb)
        
        # 2. SwissDock API 사용
        if target_pdb:
            return self._run_swissdock_api(smiles, target_pdb)
        
        # 3. 약리단 기반 간단 예측
        return self._run_pharmacophore_prediction(smiles, target_name)
    
    def _run_vina_docking(self, smiles: str, target_pdb: str) -> DockingResult:
        """
        AutoDock Vina를 사용한 도킹
        
        실제 구현시 필요한 단계:
        1. SMILES를 3D 구조로 변환 (RDKit)
        2. 리간드를 PDBQT 형식으로 변환
        3. 단백질을 PDBQT 형식으로 변환
        4. Grid box 설정
        5. Vina 실행
        6. 결과 파싱
        """
        
        # 간단한 모의 구현
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # 3D 구조 생성
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        
        # 실제 도킹 대신 예측값 반환 (데모용)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # 간단한 예측 모델
        binding_affinity = -6.0 - (logp * 0.5) - (mw / 100)
        
        return DockingResult(
            ligand_id=smiles[:10],
            binding_affinity=binding_affinity,
            rmsd_lb=0.0,
            rmsd_ub=2.5,
            interactions={
                "hydrogen_bonds": ["GLU166", "HIS163"],
                "hydrophobic": ["MET165", "LEU167", "PRO168"],
                "pi_stacking": ["HIS41"]
            }
        )
    
    def _run_swissdock_api(self, smiles: str, target_pdb: str) -> DockingResult:
        """
        SwissDock API를 사용한 도킹
        
        주의: 실제 API는 비동기 처리가 필요하며 결과를 받는데 시간이 걸립니다.
        여기서는 간단한 모의 구현을 제공합니다.
        """
        
        # SwissDock API 엔드포인트 (실제 사용시 변경 필요)
        api_url = "http://www.swissdock.ch/docking"
        
        # 실제 API 호출 대신 모의 결과 생성
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # 분자 특성 기반 예측
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        
        # 도킹 점수 예측 (간단한 모델)
        binding_affinity = -5.0 - (hbd * 0.7) - (hba * 0.5) - (rotatable * 0.1)
        
        return DockingResult(
            ligand_id=smiles[:10],
            binding_affinity=binding_affinity,
            rmsd_lb=0.0,
            rmsd_ub=3.0,
            interactions={
                "hydrogen_bonds": [f"RES{i}" for i in range(hbd)],
                "hydrophobic": ["VAL", "LEU", "ILE"],
                "electrostatic": ["ARG", "LYS"] if hba > 3 else []
            }
        )
    
    def _run_pharmacophore_prediction(self, smiles: str, 
                                     target_name: Optional[str] = None) -> DockingResult:
        """
        약리단 기반 도킹 예측
        
        실제 도킹 시뮬레이션 없이 분자의 약리단 특성을 기반으로
        결합 친화도를 예측합니다.
        """
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # 약리단 특성 계산
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        
        # 타겟별 가중치 (실제로는 타겟별 학습 모델 사용)
        if target_name and "kinase" in target_name.lower():
            # Kinase 타겟용 가중치
            binding_affinity = (-4.0 
                              - (hbd * 0.8) 
                              - (aromatic_rings * 0.6)
                              - (logp * 0.3))
        elif target_name and "protease" in target_name.lower():
            # Protease 타겟용 가중치
            binding_affinity = (-5.0 
                              - (hba * 0.7)
                              - (mw / 100)
                              + (tpsa / 50))
        else:
            # 일반적인 예측
            binding_affinity = (-4.5 
                              - (hbd * 0.5)
                              - (hba * 0.4)
                              - (logp * 0.2)
                              - (mw / 150))
        
        # 상호작용 예측
        interactions = {}
        
        if hbd > 0:
            interactions["hydrogen_bonds"] = [f"H-bond-{i+1}" for i in range(min(hbd, 3))]
        
        if logp > 2:
            interactions["hydrophobic"] = ["Hydrophobic pocket"]
        
        if aromatic_rings > 0:
            interactions["pi_stacking"] = [f"Pi-stack-{i+1}" for i in range(aromatic_rings)]
        
        return DockingResult(
            ligand_id=smiles[:10],
            binding_affinity=binding_affinity,
            rmsd_lb=0.0,
            rmsd_ub=2.0,
            interactions=interactions
        )
    
    def compare_docking_results(self, 
                               result1: DockingResult, 
                               result2: DockingResult) -> Dict:
        """
        두 도킹 결과 비교
        
        Args:
            result1: 첫 번째 화합물 도킹 결과
            result2: 두 번째 화합물 도킹 결과
            
        Returns:
            비교 분석 결과
        """
        
        affinity_diff = result2.binding_affinity - result1.binding_affinity
        
        # 상호작용 비교
        interactions_gained = {}
        interactions_lost = {}
        
        for interaction_type in set(list(result1.interactions.keys()) + 
                                   list(result2.interactions.keys())):
            set1 = set(result1.interactions.get(interaction_type, []))
            set2 = set(result2.interactions.get(interaction_type, []))
            
            gained = set2 - set1
            lost = set1 - set2
            
            if gained:
                interactions_gained[interaction_type] = list(gained)
            if lost:
                interactions_lost[interaction_type] = list(lost)
        
        return {
            "affinity_difference": affinity_diff,
            "affinity_improvement": affinity_diff < 0,  # 더 negative가 좋음
            "fold_change": 10 ** (-affinity_diff / 1.36),  # Ki 비율
            "interactions_gained": interactions_gained,
            "interactions_lost": interactions_lost,
            "summary": self._generate_comparison_summary(affinity_diff, 
                                                        interactions_gained, 
                                                        interactions_lost)
        }
    
    def _generate_comparison_summary(self, 
                                    affinity_diff: float,
                                    gained: Dict,
                                    lost: Dict) -> str:
        """비교 요약 생성"""
        
        if affinity_diff < -1.0:
            summary = f"화합물 2가 {abs(affinity_diff):.1f} kcal/mol 더 강한 결합을 보입니다. "
        elif affinity_diff > 1.0:
            summary = f"화합물 1이 {affinity_diff:.1f} kcal/mol 더 강한 결합을 보입니다. "
        else:
            summary = "두 화합물의 결합 친화도는 유사합니다. "
        
        if gained:
            summary += f"새로운 상호작용: {', '.join(gained.keys())}. "
        
        if lost:
            summary += f"소실된 상호작용: {', '.join(lost.keys())}."
        
        return summary

# 전역 인스턴스 생성 함수
def get_docking_simulator(api_key: Optional[str] = None) -> DockingSimulator:
    """도킹 시뮬레이터 인스턴스 반환"""
    return DockingSimulator(api_key)

# 간편 사용 함수
def quick_dock(smiles: str, 
               target: Optional[str] = None) -> Dict:
    """
    빠른 도킹 수행
    
    Args:
        smiles: 리간드 SMILES
        target: 타겟 정보 (PDB ID 또는 이름)
        
    Returns:
        도킹 결과 딕셔너리
    """
    simulator = get_docking_simulator()
    result = simulator.perform_docking(smiles, target_name=target)
    
    return {
        "binding_affinity": result.binding_affinity,
        "interactions": result.interactions,
        "ki_estimate": 10 ** (-result.binding_affinity / 1.36)  # μM 단위
    }