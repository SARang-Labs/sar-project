import pandas as pd
import numpy as np
import os

def process_excel_sheet(excel_source, smiles_sheet_name, ic50_sheet_name, target_column_name):
    """
    엑셀 파일 경로 또는 메모리 내 객체를 받아 특정 타겟의 IC50 데이터를 처리합니다.
    """
    try:
        # excel_source는 파일 경로(str) 또는 메모리 내 파일(BytesIO)일 수 있습니다.
        # pandas.ExcelFile은 두 경우 모두 처리 가능합니다.
        xls = pd.ExcelFile(excel_source)

        # Extract SMILES data
        if smiles_sheet_name not in xls.sheet_names:
            print(f"Error: SMILES sheet '{smiles_sheet_name}' not found in the Excel file.")
            return pd.DataFrame()
        smiles_df = xls.parse(smiles_sheet_name)
        smiles_data = smiles_df.iloc[:, [0, 1]].copy()
        smiles_data.columns = ['Compound', 'SMILES']

        # Extract IC50 data
        if ic50_sheet_name not in xls.sheet_names:
            print(f"Error: IC50 sheet '{ic50_sheet_name}' not found in the Excel file.")
            return pd.DataFrame()
        ic50_df = xls.parse(ic50_sheet_name)

        # Check if the target column exists
        if target_column_name not in ic50_df.columns:
            print(f"Error: Target column '{target_column_name}' not found in the IC50 sheet.")
            return pd.DataFrame()

        # Create a new DataFrame with the first column and the current target column
        target_ic50_df = ic50_df.loc[:, ['Compound', target_column_name]].copy()

        # Merge with smiles_data on 'Compound'
        merged_df = pd.merge(smiles_data, target_ic50_df, on='Compound')

        # Drop the 'Compound' column
        merged_df = merged_df.drop(columns=['Compound'])

        # Clean and convert IC50 values to numeric (in μM)
        merged_df[target_column_name] = merged_df[target_column_name].astype(str).str.replace(' μM', '', regex=False).str.replace('>', '', regex=False)
        merged_df[target_column_name] = pd.to_numeric(merged_df[target_column_name], errors='coerce')

        # Convert IC50 (in μM) to M and then calculate pIC50
        merged_df['p' + target_column_name] = -np.log10(merged_df[target_column_name] * 1e-6)

        # Categorize activity based on IC50 values (in μM)
        conditions = [
            (merged_df[target_column_name] < 0.1), # Highly Active: IC50 < 100 nM
            (merged_df[target_column_name] >= 0.1) & (merged_df[target_column_name] < 2), # Moderately Active
            (merged_df[target_column_name] >= 2) & (merged_df[target_column_name] < 10), # Weakly Active
            (merged_df[target_column_name] >= 10) | (merged_df[target_column_name].isna()) # Inactive
        ]
        choices = ['Highly Active', 'Moderately Active', 'Weakly Active', 'Inactive']
        merged_df['Activity'] = np.select(conditions, choices, default='Unknown')

        # Rename columns
        merged_df.rename(columns={
            'SMILES': 'SMILES',
            target_column_name: f'{target_column_name} IC50',
            'p' + target_column_name: f'{target_column_name} pIC50',
            'Activity': 'Activity'
        }, inplace=True)

        return merged_df

    except FileNotFoundError:
        # excel_source가 문자열일 때만 이 에러가 의미 있습니다.
        if isinstance(excel_source, str):
            print(f"Error: Excel file not found at {excel_source}")
        else:
            print("Error: Could not read the provided Excel file data.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return pd.DataFrame()


def parse_file_1(excel_source):
    """
    Parser for Excel File 1.
    """
    smiles_sheet = '표3'
    ic50_sheet = '표5~표14'
    target_columns = ['KU-19-19', '253J', '5637', 'J82', 'T24', 'MBT-2', 'UM-UC-3', 'PC-3', 'DU145', 'DU145/TXR', 'Lncap', 'CWR22', 'NCI-H522', 'NCI-H1437', 'A549', 'NCI-H460', 'MRC-5', 'DMS 114', 'NCI-H23', 'NCI-H12 99', 'MCF7', 'MDA-MB-231', 'MDA-MB-231 -L/DOX', 'SK-BR-3', 'BT-20', 'HCC1395', 'HCC1954', 'JIMT-1', 'MDA-M B-468', 'HL-60', 'U-937', 'Raji', 'Ramos (RA 1)', 'Daudi', 'Jurkat', 'MV-4-11', 'MOLT-4', 'PANC-1', 'AsPC-1', 'Capan-1', 'MIA PaCa-2', 'BxPC-3', 'CFPAC-1', 'Capan-2', 'HT-29', 'HCT 116', 'SW620', 'LoVo', 'HCT-15', 'RKO', 'HCT-8', 'DLD-1', 'SW48 0', 'U-251 MG', 'T98G', 'U-138 MG', 'SH-SY5Y', 'LOX-IMVI', 'SK-HEP-1', 'HeLa', 'OVCAR-3', 'ACHN', 'RT4']

    processed_dataframes = {}
    for target in target_columns:
        df = process_excel_sheet(excel_source, smiles_sheet, ic50_sheet, target)
        if not df.empty:
            processed_dataframes[target] = df
        else:
            print(f"Warning: Could not process data for target '{target}'.")

    return processed_dataframes

# Define placeholder functions for the other three files
def parse_file_2(excel_source):
    """
    Placeholder parser for Excel File 2.
    """
    print(f"Placeholder parser for file.")
    return {}

def parse_file_3(excel_source):
    """
    Placeholder parser for Excel File 3.
    """
    print(f"Placeholder parser for file.")
    return {}

def parse_file_4(excel_source):
    """
    Placeholder parser for Excel File 4.
    """
    print(f"Placeholder parser for file.")
    return {}

if __name__ == '__main__':
    # Example usage (for testing the parser)
    excel_file = os.path.join("data", "1020170094694_extracted_250611.xlsx")
    processed_data = parse_file_1(excel_file)

    if processed_data:
        print("\nSuccessfully processed data for the following targets:")
        for target, df in processed_data.items():
            print(f"- {target} (Shape: {df.shape})")
    else:
        print("\nFailed to process any data from the Excel file.")
