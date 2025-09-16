# docking_pipeline/run_test.py
import pprint
from main import run_full_docking_pipeline

if __name__ == "__main__":
    # c-Myc 저해제 특허(KR101920163B1)에 언급된 화합물 56
    test_smiles = "Cc1cc(CS(=O)c2[nH]c3c(Br)ccc(Cl)c3c(=O)c2C(=O)C(C)C)no1"
    test_pdb_id = "6G6K"
    test_target_name = "c-Myc"

    print(f"--- Running Docking Pipeline ---")
    print(f"Ligand SMILES: {test_smiles}")
    print(f"Target PDB ID: {test_pdb_id}")
    print("-" * 30)

    result = run_full_docking_pipeline(
        smiles=test_smiles,
        pdb_id=test_pdb_id.upper(),
        target_name=test_target_name
    )

    print("\n--- Pipeline Finished ---")
    print("Final Result:")
    pprint.pprint(result)
