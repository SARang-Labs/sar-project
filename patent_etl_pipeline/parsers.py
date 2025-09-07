
import pandas as pd
import numpy as np
import os

def process_excel_sheet(excel_file_path, smiles_sheet_name, ic50_sheet_name, target_column_name):
    """
    Processes a specific target's IC50 data from an Excel file.

    Args:
        excel_file_path (str): The path to the Excel file.
        smiles_sheet_name (str): The name of the sheet containing SMILES data.
        ic50_sheet_name (str): The name of the sheet containing IC50 data.
        target_column_name (str): The name of the column containing the target's IC50 values.

    Returns:
        pandas.DataFrame: A DataFrame with SMILES, IC50, pIC50, and Activity for the target.
                          Returns an empty DataFrame if the target column is not found.
    """
    try:
        # Read the Excel file
        xls = pd.ExcelFile(excel_file_path)

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
            (merged_df[target_column_name] < 0.1), # Highly Active: IC50 < 100 nM (0.1 μM)
            (merged_df[target_column_name] >= 0.1) & (merged_df[target_column_name] < 2), # Moderately Active: 100 nM (0.1 μM) < IC50 < 2 µM
            (merged_df[target_column_name] >= 2) & (merged_df[target_column_name] < 10), # Weakly Active: 2 µM < IC50 < 10 µM
            (merged_df[target_column_name] >= 10) | (merged_df[target_column_name].isna()) # Inactive: IC50 > 10 µM or NaN
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
        print(f"Error: Excel file not found at {excel_file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return pd.DataFrame()


def parse_file_1(excel_file_path):
    """
    Parser for Excel File 1.

    Args:
        excel_file_path (str): The path to the Excel file.

    Returns:
        dict: A dictionary of DataFrames, where keys are target names and values are processed DataFrames.
              Returns an empty dictionary if processing fails.
    """
    smiles_sheet = '표3'
    ic50_sheet = '표5~표14'
    # List of target column names for File 1 (based on previous exploration)
    target_columns = ['KU-19-19', '253J', '5637', 'J82', 'T24', 'MBT-2', 'UM-UC-3', 'PC-3', 'DU145', 'DU145/TXR', 'Lncap', 'CWR22', 'NCI-H522', 'NCI-H1437', 'A549', 'NCI-H460', 'MRC-5', 'DMS 114', 'NCI-H23', 'NCI-H12 99', 'MCF7', 'MDA-MB-231', 'MDA-MB-231 -L/DOX', 'SK-BR-3', 'BT-20', 'HCC1395', 'HCC1954', 'JIMT-1', 'MDA-M B-468', 'HL-60', 'U-937', 'Raji', 'Ramos (RA 1)', 'Daudi', 'Jurkat', 'MV-4-11', 'MOLT-4', 'PANC-1', 'AsPC-1', 'Capan-1', 'MIA PaCa-2', 'BxPC-3', 'CFPAC-1', 'Capan-2', 'HT-29', 'HCT 116', 'SW620', 'LoVo', 'HCT-15', 'RKO', 'HCT-8', 'DLD-1', 'SW48 0', 'U-251 MG', 'T98G', 'U-138 MG', 'SH-SY5Y', 'LOX-IMVI', 'SK-HEP-1', 'HeLa', 'OVCAR-3', 'ACHN', 'RT4']

    processed_dataframes = {}
    for target in target_columns:
        # Call the locally defined process_excel_sheet function
        df = process_excel_sheet(excel_file_path, smiles_sheet, ic50_sheet, target)
        if not df.empty:
            processed_dataframes[target] = df
        else:
            print(f"Warning: Could not process data for target '{target}' in {excel_file_path}")

    return processed_dataframes

# Define placeholder functions for the other three files
def parse_file_2(excel_file_path):
    """
    Placeholder parser for Excel File 2.
    Args:
        excel_file_path (str): The path to the Excel file.
    Returns:
        dict: An empty dictionary.
    """
    print(f"Placeholder parser for {excel_file_path}")
    return {}

def parse_file_3(excel_file_path):
    """
    Placeholder parser for Excel File 3.
    Args:
        excel_file_path (str): The path to the Excel file.
    Returns:
        dict: An empty dictionary.
    """
    print(f"Placeholder parser for {excel_file_path}")
    return {}

def parse_file_4(excel_file_path):
    """
    Placeholder parser for Excel File 4.
    Args:
        excel_file_path (str): The path to the Excel file.
    Returns:
        dict: An empty dictionary.
    """
    print(f"Placeholder parser for {excel_file_path}")
    return {}

if __name__ == '__main__':
    # Example usage (for testing the parser)
    # Assuming the Excel file '1020170094694_extracted_250611.xlsx' is in the /content directory
    excel_file = "/content/1020170094694_extracted_250611.xlsx"
    processed_data = parse_file_1(excel_file)

    if processed_data:
        print("\nSuccessfully processed data for the following targets:")
        for target, df in processed_data.items():
            print(f"- {target} (Shape: {df.shape})")
            # Optionally display head of a few dataframes
            # print(df.head())
    else:
        print("\nFailed to process any data from the Excel file.")

