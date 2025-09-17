# docking_pipeline/parser.py

import os
from typing import Dict, Any, Optional

import prolif as plf
from prolif.molecule import Molecule
import MDAnalysis as mda

def parse_vina_log(log_path: str) -> Optional[float]:
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        
        affinity_section = False
        for line in lines:
            if line.strip().startswith("-----+"):
                affinity_section = True
                continue
            if affinity_section:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        affinity = float(parts[1])
                        return affinity
                    except ValueError:
                        continue
        
        print(f"WARNING: Affinity values not found in log: {log_path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to parse log file {log_path}: {e}")
        return None

def analyze_vina_poses_with_prolif(receptor_pdb: str, ligand_path: str, vicinity_cutoff: float = 7.0) -> Dict[str, Any]:
    u = mda.Universe(receptor_pdb)
    prot_ag = u.select_atoms("protein")
    if prot_ag.n_atoms == 0:
        return {"error": "Protein selection empty from receptor PDB"}
    
    prot = Molecule.from_mda(prot_ag, NoImplicit=False)
    
    supplier_kwargs = dict(resname="UNL", chain="L", resnumber=1)
    
    lower = ligand_path.lower()
    if lower.endswith(".sdf"):
        # SDF supplier에서 sanitize 인자 제거
        lig_iter = plf.sdf_supplier(ligand_path, **supplier_kwargs)
    elif lower.endswith(".mol2"):
        # MOL2 supplier에서 sanitize 인자 제거
        lig_iter = plf.mol2_supplier(ligand_path, **supplier_kwargs)
    elif lower.endswith(".pdbqt"):
        lig_iter = plf.pdbqt_supplier([ligand_path], template=None, converter_kwargs=None, **supplier_kwargs)
    else:
        return {"error": "Unsupported ligand poses format. Use SDF/MOL2/PDBQT."}
    
    fp = plf.Fingerprint(vicinity_cutoff=vicinity_cutoff)
    fp.run_from_iterable(lig_iter, prot)
    
    df = fp.to_dataframe(index_col="Pose")
    if df is None or df.empty:
        return {"message": "No interactions detected in poses."}
    
    return {"ifp_dataframe": df}

