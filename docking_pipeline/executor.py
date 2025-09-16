# docking_pipeline/executor.py
import os
import subprocess
from typing import Optional, Tuple
from prepare import calculate_protein_center

def run_vina(receptor_path: str, ligand_path: str, target_name: str, output_dir: str = "outputs") -> Optional[Tuple[str, str]]:
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(ligand_path).replace(".pdbqt", "")
    output_pose_path = os.path.join(output_dir, f"{base_name}_out.pdbqt")
    output_log_path = os.path.join(output_dir, f"{base_name}_log.txt")

    pdb_path = receptor_path.replace(".pdbqt", ".pdb")
    center_x, center_y, center_z = calculate_protein_center(pdb_path)

    cmd = [
        "vina",
        "--receptor", receptor_path,
        "--ligand", ligand_path,
        "--out", output_pose_path,
        "--center_x", str(center_x),
        "--center_y", str(center_y),
        "--center_z", str(center_z),
        "--size_x", "40",
        "--size_y", "40",
        "--size_z", "40",
        "--exhaustiveness", "4",
        "--num_modes", "20",
        "--energy_range", "5"
    ]

    print("INFO: Running AutoDock Vina with auto center...")
    try:
        with open(output_log_path, 'w') as log_file:
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
                text=True
            )
        print(f"INFO: Vina execution finished. Log available at {output_log_path}")
        if not os.path.isfile(output_log_path):
            print("WARNING: Log file was not created.")
        return output_pose_path, output_log_path
    except FileNotFoundError:
        print("ERROR: 'vina' command not found. Is AutoDock Vina installed and in your system's PATH?")
        with open(output_log_path, 'w') as f:
            f.write("ERROR: 'vina' command not found.\n")
        return None, output_log_path
    except subprocess.CalledProcessError:
        print(f"ERROR: Vina docking failed. See log file {output_log_path}")
        if os.path.isfile(output_log_path):
            with open(output_log_path, 'r') as f:
                print(f.read())
        return None, output_log_path
    except Exception as e:
        print(f"ERROR: Vina docking failed: {e}")
        with open(output_log_path, 'w') as f:
            f.write(f"Unexpected error: {e}\n")
        return None, output_log_path

