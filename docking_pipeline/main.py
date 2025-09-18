# docking_pipeline/main.py
import os
from typing import Dict, Any
import prepare
import executor
import parser as parser_mod
import parser as parser

def run_full_docking_pipeline(smiles: str, pdb_id: str, target_name: str, cleanup: bool = True) -> Dict[str, Any]:
    # docking_pipeline 폴더 내부 경로 지정
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(pipeline_dir, "outputs")
    receptors_dir = os.path.join(pipeline_dir, "receptors")

    # 폴더가 없으면 생성
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(receptors_dir, exist_ok=True)

    pdb_path = prepare.fetch_pdb(pdb_id, directory=receptors_dir)
    if not pdb_path or not os.path.exists(pdb_path):
        return {"error": f"Could not fetch PDB for {pdb_id}. File does not exist."}

    receptor_path = prepare.prepare_receptor(pdb_path, output_dir=receptors_dir)
    if not receptor_path:
        return {"error": f"Could not prepare receptor for {pdb_id}"}

    ligand_path = prepare.prepare_ligand(smiles, output_dir=outputs_dir)
    if not ligand_path or not os.path.exists(ligand_path) or os.path.getsize(ligand_path) == 0:
        print(f"ERROR: Ligand file was not created or is empty: {ligand_path}")
        return {"error": "Ligand preparation failed."}
    else:
        print(f"DEBUG: Ligand file created: {ligand_path} (size: {os.path.getsize(ligand_path)})")

    run_results = executor.run_vina(receptor_path, ligand_path, pdb_id, output_dir=outputs_dir)
    if not run_results or not all(run_results):
        print(f"ERROR: Vina execution failed. run_results={run_results}")
        if os.path.exists(ligand_path):
            os.remove(ligand_path)
        return {"error": "Vina execution failed."}

    pose_pdbqt_path, log_path = run_results
    for f in [pose_pdbqt_path, log_path]:
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            print(f"ERROR: Output file missing or empty after Vina: {f}")
        else:
            print(f"DEBUG: Output file created: {f} (size: {os.path.getsize(f)})")
    binding_affinity = parser.parse_vina_log(log_path)
    if binding_affinity is None:
        binding_affinity = "Affinity parsing failed or not found."

    # receptor PDB 경로 확보
    receptor_pdb_path = receptor_path.replace(".pdbqt", ".pdb")
    if not os.path.exists(receptor_pdb_path):
        receptor_pdb_path = prepare.convert_pdbqt_to_pdb(receptor_path)
        if not receptor_pdb_path:
            return {"error": "Failed to convert receptor pdbqt to pdb."}

    # 템플릿 매핑 SDF 우선
    mapped_sdf = os.path.join(outputs_dir, os.path.basename(pose_pdbqt_path).replace(".pdbqt", ".mapped.sdf"))
    raw_sdf    = os.path.join(outputs_dir, os.path.basename(pose_pdbqt_path).replace(".pdbqt", ".sdf"))
    try:
        ligand_input_for_prolif = prepare.vina_pose_to_template_mapped_sdf(smiles, pose_pdbqt_path, mapped_sdf)
    except Exception as e:
        print(f"WARNING: Template-mapped SDF failed ({e}), trying raw SDF")
        try:
            prepare.vina_pose_pdbqt_to_sdf(pose_pdbqt_path, raw_sdf)
            ligand_input_for_prolif = raw_sdf
        except Exception as e2:
            print(f"WARNING: Raw SDF conversion failed ({e2}), fallback to PDBQT supplier")
            ligand_input_for_prolif = pose_pdbqt_path

    # ProLIF 분석(근접 컷오프 조정 포함)
    prolif_result = parser.analyze_vina_poses_with_prolif(
        receptor_pdb=receptor_pdb_path,
        ligand_path=ligand_input_for_prolif,
        vicinity_cutoff=7.0
    )

    final_result = {
        "smiles": smiles,
        "pdb_id": pdb_id,
        "binding_affinity_kcal_mol": binding_affinity,
        "prolif": prolif_result,
        "receptor_std_pdb": receptor_pdb_path,
        "vina_pose_file": pose_pdbqt_path,
    }

    # 전체 도킹 결과(final_result)를 저장 (DataFrame 직렬화 처리, SMILES 해시 기반 파일명)
    import hashlib
    def smiles_hash(smiles):
        return hashlib.md5(smiles.encode('utf-8')).hexdigest()[:8]
    smiles_id = smiles_hash(smiles)
    prolif_output_path = os.path.join(outputs_dir, f"pr_{smiles_id}__{pdb_id}.json")
    try:
        import json
        import pandas as pd
        def safe_json(obj):
            # prolif 내부 ifp_dataframe이 DataFrame이면 dict로 변환 (튜플 키를 문자열로 변환)
            if isinstance(obj, dict) and 'prolif' in obj and isinstance(obj['prolif'], dict):
                if 'ifp_dataframe' in obj['prolif'] and isinstance(obj['prolif']['ifp_dataframe'], pd.DataFrame):
                    df = obj['prolif']['ifp_dataframe']
                    new_dict = {}
                    for k, v in df.to_dict().items():
                        if isinstance(k, tuple):
                            new_key = '|'.join(map(str, k))
                        else:
                            new_key = str(k)
                        new_dict[new_key] = v
                    obj['prolif']['ifp_dataframe'] = new_dict
            return obj

        with open(prolif_output_path, "w") as prolif_file:
            json.dump(safe_json(final_result), prolif_file, indent=4)
        print(f"INFO: Docking + ProLIF result saved to {prolif_output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save docking + ProLIF result: {e}")

    return final_result

