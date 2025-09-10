
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

if __name__ == '__main__':
    # Example usage (this part will not be executed during the subtask but is good for testing the script)
    excel_file = "patent_etl_pipeline/data/1020170094694_extracted_250611.xlsx"
    smiles_sheet = '표3'
    ic50_sheet = '표5~표14'
    target = 'KU-19-19' # Example target

    processed_data = process_excel_sheet(excel_file, smiles_sheet, ic50_sheet, target)

    if not processed_data.empty:
        print(f"Processed data for {target}:")
        print(processed_data.head())

        # Example of saving the processed data
        output_dir = "processed_data_example"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{target}_pIC50.csv"
        output_filepath = os.path.join(output_dir, output_filename)
        processed_data.to_csv(output_filepath, index=False)
        print(f"Processed data saved to {output_filepath}")
    else:
        print(f"Failed to process data for {target}.")
