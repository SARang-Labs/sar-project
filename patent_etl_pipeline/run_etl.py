import os
import sqlite3
import pandas as pd
import sys
import logging
import argparse

# --- 설정 ---

LOG_FILE = "etl.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[ logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger(__name__)

try:
    import parsers
    logger.info("Successfully imported 'parsers' module.")
except ImportError:
    logger.error("Could not import 'parsers' module.", exc_info=True)
    sys.exit(1)

DATABASE_DIR = "database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "patent_data.db")
EXCEL_FILES_DIR = "data"

def run_etl(patent_number, excel_file_path):
    """
    특정 특허 번호와 엑셀 파일에 대한 ETL 프로세스를 실행합니다.
    """
    conn = None
    logger.info(f"Starting ETL process for patent '{patent_number}' with file '{os.path.basename(excel_file_path)}'.")
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        logger.info(f"Connected to database: {DATABASE_PATH}")

        # 1. 특허 번호를 patents 테이블에 삽입하고 patent_id를 가져옵니다.
        cursor.execute("INSERT OR IGNORE INTO patents (patent_number) VALUES (?)", (patent_number,))
        cursor.execute("SELECT patent_id FROM patents WHERE patent_number = ?", (patent_number,))
        patent_id = cursor.fetchone()[0]
        logger.info(f"Using patent_id: {patent_id} for patent_number: '{patent_number}'.")

        # 2. 파일 이름에 맞는 파서를 선택합니다.
        parser_mapping = {
            "1020170094694_extracted_250611.xlsx": parsers.parse_file_1,
            "file2.xlsx": parsers.parse_file_2,
            "file3.xlsx": parsers.parse_file_3,
            "file4.xlsx": parsers.parse_file_4,
        }
        excel_file_name = os.path.basename(excel_file_path)
        parser_func = parser_mapping.get(excel_file_name)

        if not parser_func:
            logger.warning(f"No parser defined for file: {excel_file_name}. Skipping.")
            return

        # 3. 파서를 실행하여 데이터를 추출하고 변환합니다.
        processed_dataframes = parser_func(excel_file_path)

        # 4. 변환된 데이터를 DB에 저장합니다.
        for target_name, df in processed_dataframes.items():
            logger.info(f"  Loading data for target: '{target_name}'")
            try:
                # target_id 가져오기
                cursor.execute("INSERT OR IGNORE INTO targets (target_name) VALUES (?)", (target_name,))
                cursor.execute("SELECT target_id FROM targets WHERE target_name = ?", (target_name,))
                target_id = cursor.fetchone()[0]

                # 각 행의 데이터를 DB에 삽입
                for index, row in df.iterrows():
                    try:
                        smiles = row['SMILES']
                        ic50 = row.get(f'{target_name} IC50')
                        pic50 = row.get(f'{target_name} pIC50')
                        activity_category = row['Activity']
                        
                        # compound_id 가져오기
                        cursor.execute("INSERT OR IGNORE INTO compounds (smiles) VALUES (?)", (smiles,))
                        cursor.execute("SELECT compound_id FROM compounds WHERE smiles = ?", (smiles,))
                        compound_id = cursor.fetchone()[0]

                        # activities 테이블에 최종 데이터 삽입 (patent_id 포함)
                        cursor.execute("""
                            INSERT INTO activities (compound_id, target_id, patent_id, ic50, pic50, activity_category)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (compound_id, target_id, patent_id, ic50, pic50, activity_category))
                    
                    except Exception as row_e:
                        logger.error(f"Error processing row {index} for target {target_name}: {row_e}", exc_info=True)
                
                # 이 타겟에 대한 모든 변경사항을 커밋
                conn.commit()
                logger.info(f"  Successfully loaded data for target: '{target_name}'")

            except Exception as e:
                logger.error(f"  Error processing target '{target_name}': {e}", exc_info=True)
                conn.rollback() # 오류 발생 시 이 타겟에 대한 변경사항 롤백
                logger.warning(f"  Rolled back transaction for target '{target_name}'.")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during ETL process: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")
        logger.info(f"ETL process for patent '{patent_number}' finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ETL for a specific patent Excel file.")
    parser.add_argument("patent_number", type=str, help="The patent number to process (e.g., '1020170094694').")
    parser.add_argument("file_name", type=str, help="The corresponding Excel file name (e.g., '1020170094694_extracted_250611.xlsx').")
    args = parser.parse_args()

    excel_file_full_path = os.path.join(EXCEL_FILES_DIR, args.file_name)

    if not os.path.exists(excel_file_full_path):
        logger.error(f"Error: Excel file not found at {excel_file_full_path}.")
    else:
        run_etl(args.patent_number, excel_file_full_path)
