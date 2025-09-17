# docking_pipeline/prepare.py
import os
import requests
import subprocess
import time
import random
import glob
from typing import Optional, Tuple
from rdkit_mapping import build_template_mapped_sdf

def fetch_pdb(pdb_id: str, directory: str = "receptors") -> Optional[str]:
    def is_valid_pdb_file(path):
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) == 0:
            return False
        with open(path, "r") as f:
            first_line = f.readline().strip()
            return first_line.startswith("HEADER") or first_line.startswith("ATOM")
    os.makedirs(directory, exist_ok=True)
    pdb_path = os.path.join(directory, f"{pdb_id}.pdb")

    def is_valid_pdb_file(path):
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) == 0:
            return False
        with open(path, "r") as f:
            first_line = f.readline().strip()
            return first_line.startswith("HEADER") or first_line.startswith("ATOM")

    # Check if the file exists and is valid
    if is_valid_pdb_file(pdb_path):
        print(f"INFO: PDB file for {pdb_id} already exists at {pdb_path}")
        return pdb_path
    else:
        if os.path.exists(pdb_path):
            print(f"WARNING: Invalid or empty PDB file detected at {pdb_path}, re-downloading...")
            os.remove(pdb_path)

    # Download the PDB file
    print(f"INFO: Downloading PDB file for {pdb_id}...")
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(pdb_path, "w") as f:
            f.write(response.text)
        print(f"INFO: Successfully downloaded {pdb_path}")
        if is_valid_pdb_file(pdb_path):
            return pdb_path
        else:
            print(f"ERROR: Downloaded file for {pdb_id} is invalid or empty.")
            if os.path.exists(pdb_path):
                os.remove(pdb_path)
    except requests.RequestException as e:
        print(f"ERROR: Failed to download PDB file for {pdb_id}. Reason: {e}")
    return None

def prepare_ligand(smiles: str, output_dir: str = "outputs") -> Optional[str]:
    # Keep original: generate ligand PDBQT for Vina input using Open Babel
    os.makedirs(output_dir, exist_ok=True)
    ligand_id = f"ligand_{int(time.time())}_{random.randint(1000, 9999)}"
    smi_path = os.path.join(output_dir, f"{ligand_id}.smi")
    pdbqt_path = os.path.join(output_dir, f"{ligand_id}.pdbqt")
    try:
        with open(smi_path, "w") as f:
            f.write(smiles)
        cmd = ["obabel", smi_path, "-O", pdbqt_path, "--gen3d", "--addhydrogens"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
            print(f"INFO: Ligand prepared successfully at {pdbqt_path}")
            os.remove(smi_path)
            return pdbqt_path
        else:
            print(f"ERROR: Ligand PDBQT file was not created or is empty: {pdbqt_path}")
            if os.path.exists(smi_path):
                os.remove(smi_path)
            if os.path.exists(pdbqt_path):
                os.remove(pdbqt_path)
            return None
    except Exception as e:
        print(f"ERROR: Ligand preparation failed. Error: {e}")
        if os.path.exists(smi_path):
            os.remove(smi_path)
        if os.path.exists(pdbqt_path):
            os.remove(pdbqt_path)
        return None

def prepare_receptor(pdb_path: str, output_dir: str = "receptors") -> Optional[str]:
    # Keep original receptor -> pdbqt for Vina input using Open Babel
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(pdb_path)
    cleaned_pdb_path = os.path.join(output_dir, base_name.replace(".pdb", "_cleaned.pdb"))
    pdbqt_path = os.path.join(output_dir, base_name.replace(".pdb", ".pdbqt"))
    if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
        print(f"INFO: Receptor PDBQT file already exists at {pdbqt_path}")
        return pdbqt_path

    important_hets = {"ZN", "MG", "CA", "FE", "NA", "CL", "K", "MN"}
    try:
        if not os.path.exists(pdb_path) or os.path.getsize(pdb_path) == 0:
            print(f"ERROR: Input PDB file does not exist or is empty: {pdb_path}")
            return None
        with open(pdb_path, 'r') as infile, open(cleaned_pdb_path, 'w') as outfile:
            for line in infile:
                if line.startswith('ATOM'):
                    outfile.write(line)
                elif line.startswith('HETATM'):
                    resname = line[17:20].strip()
                    if resname in important_hets:
                        outfile.write(line)

        pdb_with_h_path = cleaned_pdb_path.replace("_cleaned.pdb", "_with_h.pdb")
        cmd_add_h = ["obabel", cleaned_pdb_path, "-O", pdb_with_h_path, "--addhydrogens", "--pH", "7.4"]
        subprocess.run(cmd_add_h, check=True, capture_output=True, text=True)
        print("INFO: Added hydrogens to receptor")

        cmd = ["obabel", pdb_with_h_path, "-O", pdbqt_path, "-xr"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Open Babel stdout:", result.stdout)
        print("Open Babel stderr:", result.stderr)

        os.remove(cleaned_pdb_path)
        os.remove(pdb_with_h_path)
        if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
            print(f"INFO: Receptor prepared successfully at {pdbqt_path}")
            return pdbqt_path
        else:
            print(f"ERROR: Receptor PDBQT file was not created or is empty: {pdbqt_path}")
            if os.path.exists(pdbqt_path):
                os.remove(pdbqt_path)
            return None
    except Exception as e:
        print(f"ERROR: Receptor preparation failed. Error: {e}")
        if os.path.exists(cleaned_pdb_path):
            os.remove(cleaned_pdb_path)
        if os.path.exists(pdb_with_h_path):
            os.remove(pdb_with_h_path)
        if os.path.exists(pdbqt_path):
            os.remove(pdbqt_path)
        return None

def convert_pdbqt_to_pdb(pdbqt_path: str) -> Optional[str]:
    # Keep simple: use Open Babel to get the first pose PDB for optional inspection
    pdb_path = pdbqt_path.replace(".pdbqt", ".pdb")
    try:
        cmd = ["obabel", pdbqt_path, "-O", pdb_path, "-f", "1", "-l", "1", "-h"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"INFO: Converted the first pose from {pdbqt_path} to {pdb_path} (with hydrogens)")
        return pdb_path
    except Exception as e:
        print(f"ERROR: Failed to convert PDBQT to PDB. Reason: {e}")
        return None

# New helpers for MDAnalysis standardization and complex building
import MDAnalysis as mda
from MDAnalysis.core.universe import Merge

ION_RESNAMES = ["ZN", "MG", "CA", "FE", "NA", "CL", "K", "MN"]

def standardize_receptor_pdb(pdb_in: str, pdb_out: str, keep_ions=True, remove_waters=True) -> str:
    # Write a standardized PDB with proper fixed-width, element column and optional CONECT
    u = mda.Universe(pdb_in)
    sels = ["protein"]
    if keep_ions:
        sels.append("resname " + " ".join(ION_RESNAMES))
    if not remove_waters:
        sels.append("resname HOH WAT")
    sel = " or ".join(sels)
    ag = u.select_atoms(sel)
    if ag.n_atoms == 0:
        raise ValueError("No atoms selected for receptor standardization")
    # Assign chainIDs for stability; MDAnalysis will default to 'X' if missing
    ag.atoms.chainIDs = ["A"] * ag.n_atoms
    ag.write(pdb_out, bonds="all")  # ensure fixed-width and element ordering
    return pdb_out  # MDAnalysis PDB writer details [12][13]

def calculate_protein_center(pdb_path: str) -> Tuple[float, float, float]:
    # Keep original center estimation
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                except ValueError:
                    continue
    import numpy as np
    if len(coords) == 0:
        raise ValueError("No ATOM/HETATM records found for center calculation")
    center = tuple(np.mean(np.array(coords), axis=0))
    print(f"Protein center (mean): {center}")
    return center

def vina_split_single_model(in_pdbqt: str) -> str:
    workdir = os.path.dirname(in_pdbqt) or "."
    subprocess.run(["vina_split", "--input", in_pdbqt], check=True, capture_output=True, text=True)  # [11][10]
    base = os.path.splitext(os.path.basename(in_pdbqt))[0]  # 기존: os.path.splitext(...) (튜플) [11]
    patterns = [
        os.path.join(workdir, f"{base}_*.pdbqt"),
        os.path.join(workdir, "*_out_*.pdbqt"),
        os.path.join(workdir, "ligand*_*.pdbqt"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"vina_split produced no split models in {workdir}")
    single = candidates[0]  # 기존: single = candidates (리스트 반환) [11]
    if not os.path.isfile(single):
        raise FileNotFoundError(f"Split model not found on disk: {single}")
    return single

def vina_pose_pdbqt_to_sdf(pdbqt_path: str, sdf_out: str) -> str:
    cmd = ["obabel", pdbqt_path, "-O", sdf_out, "-f", "1", "-l", "1", "-h"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sdf_out

def vina_pose_to_template_mapped_sdf(smiles: str, vina_pdbqt: str, out_sdf: str) -> str:
    try:
        single = vina_split_single_model(vina_pdbqt)
    except Exception as e:
        print(f"WARNING: vina_split failed ({e}), using original out.pdbqt")
        single = vina_pdbqt
    tmp_sdf = out_sdf.replace(".sdf", ".rawpose.sdf")
    if not os.path.isfile(single):
        raise FileNotFoundError(f"Input PDBQT for obabel not found: {single}")
    # 첫 포즈 SDF 변환
    subprocess.run(["obabel", single, "-O", tmp_sdf, "-f", "1", "-l", "1", "-h"], check=True, capture_output=True, text=True)
    if not os.path.isfile(tmp_sdf) or os.path.getsize(tmp_sdf) == 0:
        raise FileNotFoundError(f"obabel did not create a valid SDF: {tmp_sdf}")
    # 템플릿 매핑
    mapped_sdf, _ = build_template_mapped_sdf(smiles, tmp_sdf, out_sdf)
    try:
        if os.path.exists(tmp_sdf):
            os.remove(tmp_sdf)
    except Exception:
        pass
    return mapped_sdf

