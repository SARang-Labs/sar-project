# docking_pipeline/rdkit_mapping.py

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, List, Tuple

def smiles_to_template_mol(smiles: str, add_hs: bool = True, sanitize: bool = True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    if add_hs:
        mol = Chem.AddHs(mol)
    if sanitize:
        Chem.SanitizeMol(mol, catchErrors=False)
    
    # 3D 좌표 내장
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    return mol

def load_pose_mol_from_sdf(sdf_path: str, sanitize: bool = True) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=sanitize)
    mols = [m for m in suppl if m is not None]
    if not mols:
        raise ValueError(f"No molecule read from SDF pose file: {sdf_path}")
    
    pose = mols[0]  # 첫 Mol만 선택 (수정됨)
    if pose.GetNumConformers() == 0:
        suppl2 = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        mols2 = [m for m in suppl2 if m is not None]
        if not mols2:
            raise ValueError(f"No molecule read from SDF (sanitize=False): {sdf_path}")
        pose2 = mols2[0]  # 첫 Mol만 선택 (수정됨)
        if pose2.GetNumConformers() == 0:
            raise ValueError(f"Pose SDF has no conformers even after sanitize=False: {sdf_path}")
        pose = pose2
    return pose

def atom_mapping_by_substructure(template: Chem.Mol, pose: Chem.Mol) -> Optional[List[int]]:
    # 템플릿을 query로 매칭
    match = pose.GetSubstructMatch(Chem.RemoveHs(template)) or pose.GetSubstructMatch(template)
    if match and len(match) == template.GetNumAtoms():
        return list(match)
    
    # 폴백: 원자 수 동일시 순서 매핑
    if template.GetNumAtoms() == pose.GetNumAtoms():
        return list(range(template.GetNumAtoms()))
    
    return None

def graft_coordinates(template: Chem.Mol, pose: Chem.Mol, atom_map: List[int]) -> Chem.Mol:
    tmpl = Chem.Mol(template)
    pose_conf = pose.GetConformer()
    conf = Chem.Conformer(tmpl.GetNumAtoms())
    
    for i_tmpl, i_pose in enumerate(atom_map):
        pos = pose_conf.GetAtomPosition(i_pose)
        conf.SetAtomPosition(i_tmpl, pos)
    
    conf.Set3D(True)
    cid = tmpl.AddConformer(conf, assignId=True)
    
    # 단일 conformer만 유지
    last = tmpl.GetConformer(cid)
    new_conf = Chem.Conformer(tmpl.GetNumAtoms())
    for i in range(tmpl.GetNumAtoms()):
        new_conf.SetAtomPosition(i, last.GetAtomPosition(i))
    new_conf.Set3D(True)
    
    tmpl.RemoveAllConformers()
    tmpl.AddConformer(new_conf, assignId=True)
    
    return tmpl

def write_sdf(mol: Chem.Mol, out_sdf: str) -> str:
    w = Chem.SDWriter(out_sdf)
    w.write(mol)
    w.close()
    return out_sdf

def build_template_mapped_sdf(smiles: str, pose_sdf_path: str, out_sdf: str) -> Tuple[str, str]:
    template = smiles_to_template_mol(smiles)
    pose = load_pose_mol_from_sdf(pose_sdf_path, sanitize=True)
    
    amap = atom_mapping_by_substructure(template, pose)
    if amap is None:
        pose2 = load_pose_mol_from_sdf(pose_sdf_path, sanitize=False)
        amap = atom_mapping_by_substructure(template, pose2)
        if amap is None:
            raise ValueError("Failed to map atoms between template and pose")
        pose = pose2
    
    mapped = graft_coordinates(template, pose, amap)
    write_sdf(mapped, out_sdf)
    
    return out_sdf, "mapped"


