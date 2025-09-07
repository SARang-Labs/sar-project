
import os
import sqlite3
import pandas as pd
import sys
import logging

# Add the project directory to the sys.path to import the parsers module
project_dir = "patent_etl_pipeline"
sys.path.append(project_dir)

# Configure logging
log_file = os.path.join(project_dir, "etl.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    import parsers
    logger.info("Successfully imported the 'parsers' module.")
except ImportError:
    logger.error("Error: Could not import the 'parsers' module.", exc_info=True)
    logger.error(f"Please ensure that '{project_dir}' is in your Python path or that 'parsers.py' is in the correct directory.")
    sys.exit(1)


# Define the path to the database file and the directory containing the Excel files
database_dir = os.path.join(project_dir, "database")
database_path = os.path.join(database_dir, "patent_data.db")
excel_files_dir = "/content/" # Assuming Excel files are in the /content directory

def run_etl():
    """
    Orchestrates the ETL process: extracts data using parsers,
    transforms it (already done in parsers), and loads it into the database.
    Includes logging and improved error handling.
    """
    conn = None
    logger.info("Starting ETL process.")
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        logger.info(f"Connected to database: {database_path}")

        # Define a list of the Excel files to be processed
        # NOTE: This list should be populated with the actual file names
        excel_files_to_process = [
            "1020170094694_extracted_250611.xlsx",
            "file2.xlsx", # Placeholder filename for file 2
            "file3.xlsx", # Placeholder filename for file 3
            "file4.xlsx", # Placeholder filename for file 4
        ]

        # Define a mapping from filename (or pattern) to parser function
        parser_mapping = {
            "1020170094694_extracted_250611.xlsx": parsers.parse_file_1,
            "file2.xlsx": parsers.parse_file_2, # Mapping for file 2
            "file3.xlsx": parsers.parse_file_3, # Mapping for file 3
            "file4.xlsx": parsers.parse_file_4, # Mapping for file 4
        }

        # Iterate through the list of Excel files
        for excel_file in excel_files_to_process:
            file_path = os.path.join(excel_files_dir, excel_file)
            logger.info(f"Processing file: {file_path}")

            if not os.path.exists(file_path):
                logger.error(f"Error: Excel file not found at {file_path}. Skipping.")
                continue


            if excel_file not in parser_mapping:
                logger.warning(f"No parser defined for file: {excel_file}. Skipping.")
                continue

            parser_func = parser_mapping[excel_file]

            try:
                # Call the appropriate parser function to extract and process data
                processed_dataframes = parser_func(file_path)

                if not processed_dataframes:
                    logger.info(f"No data processed from {excel_file} by the parser.")
                    continue

                # Iterate through the processed DataFrames obtained from the parser
                for target_name, df in processed_dataframes.items():
                    logger.info(f"  Loading data for target: {target_name} from file {excel_file}")

                    # Start a transaction
                    conn.execute("BEGIN TRANSACTION;")

                    try:
                        # Insert the target's name into the targets table if it doesn't already exist
                        cursor.execute("INSERT OR IGNORE INTO targets (target_name) VALUES (?)", (target_name,))
                        cursor.execute("SELECT target_id FROM targets WHERE target_name = ?", (target_name,))
                        target_id = cursor.fetchone()[0]

                        # Iterate through the rows of the DataFrame and insert into activities table
                        for index, row in df.iterrows():
                            try:
                                smiles = row['SMILES']
                                ic50 = row[f'{target_name} IC50']
                                pic50 = row[f'{target_name} pIC50']
                                activity_category = row['Activity']

                                # Insert the compound's SMILES into the compounds table if it doesn't already exist
                                cursor.execute("INSERT OR IGNORE INTO compounds (smiles) VALUES (?)", (smiles,))
                                cursor.execute("SELECT compound_id FROM compounds WHERE smiles = ?", (smiles,))
                                compound_id = cursor.fetchone()[0]

                                # Insert the activity data into the activities table
                                cursor.execute("""
                                    INSERT INTO activities (compound_id, target_id, ic50, pic50, activity_category)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (compound_id, target_id, ic50, pic50, activity_category))

                            except Exception as row_e:
                                logger.error(f"Error processing row {index} for target {target_name} in file {excel_file}: {{row_e}}", exc_info=True)
                                # Continue to the next row after logging the error

                        # Commit the transaction after successfully processing a target's data
                        conn.commit()
                        logger.info(f"  Successfully loaded data for target: {target_name}")

                    except sqlite3.Error as db_e:
                        logger.error(f"Database error while processing {target_name} in file {excel_file}: {{db_e}}", exc_info=True)
                        # Rollback the transaction in case of error
                        if conn:
                            conn.rollback()
                        logger.warning(f"  Rolled back transaction for {target_name}.")
                    except Exception as target_e:
                        logger.error(f"An error occurred while processing target {target_name} in file {excel_file}: {{target_e}}", exc_info=True)
                        # Rollback the transaction in case of error
                        if conn:
                            conn.rollback()
                        logger.warning(f"  Rolled back transaction for {target_name}.")

            except FileNotFoundError:
                # This is already handled by the os.path.exists check above, but kept for safety
                logger.error(f"Error: Excel file not found at {file_path}.", exc_info=True)
            except Exception as file_e:
                logger.error(f"An error occurred while processing file {excel_file}: {{file_e}}", exc_info=True)


    except FileNotFoundError:
        logger.error(f"Error: Database file not found at {database_path}. Please run the schema creation script first.", exc_info=True)
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during ETL process: {e}", exc_info=True)
    finally:
        # Close the database connection
        if conn:
            conn.close()
            logger.info("Database connection closed.")
        logger.info("ETL process finished.")

if __name__ == '__main__':
    run_etl()
