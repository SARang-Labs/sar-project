import os
import sqlite3
import pandas as pd
import sys
import logging
import argparse
from sqlalchemy.orm import Session
from database import SessionLocal, Patent, Compound, Target, Activity

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

def get_or_create(db: Session, model, **kwargs):
    """
    데이터베이스에 특정 데이터가 있는지 확인하고, 없으면 새로 생성합니다.
    """
    instance = db.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        db.add(instance)
        db.commit()
        db.refresh(instance)
        return instance

def run_etl(patent_number, excel_file_path):
    """
    SQLAlchemy를 사용하여 ETL 프로세스를 실행합니다.
    """
    db: Session = SessionLocal() # SQLAlchemy 세션 시작
    logger.info(f"Starting ETL process for patent '{patent_number}'...")
    try:
        # 1. 특허 번호를 patents 테이블에 삽입/조회합니다.
        patent = get_or_create(db, Patent, patent_number=patent_number)
        logger.info(f"Using patent_id: {patent.patent_id} for patent_number: '{patent.patent_number}'.")

        # 2. 파일 이름에 맞는 파서를 선택하고 실행합니다.
        parser_mapping = {
            "1020170094694_extracted_250611.xlsx": parsers.parse_file_1,
            # ...
        }
        excel_file_name = os.path.basename(excel_file_path)
        parser_func = parser_mapping.get(excel_file_name)
        if not parser_func:
            logger.warning(f"No parser defined for file: {excel_file_name}. Skipping.")
            return
        processed_dataframes = parser_func(excel_file_path)

        # 3. 변환된 데이터를 DB에 저장합니다.
        for target_name, df in processed_dataframes.items():
            logger.info(f"  Loading data for target: '{target_name}'")
            try:
                # target_id 가져오기
                target = get_or_create(db, Target, target_name=target_name)

                # 각 행의 데이터를 DB에 삽입
                for index, row in df.iterrows():
                    # compound_id 가져오기
                    compound = get_or_create(db, Compound, smiles=row['SMILES'])
                    
                    # activities 테이블에 최종 데이터 삽입
                    new_activity = Activity(
                        compound_id=compound.compound_id,
                        target_id=target.target_id,
                        patent_id=patent.patent_id,
                        ic50=row.get(f'{target_name} IC50'),
                        pic50=row.get(f'{target_name} pIC50'),
                        activity_category=row['Activity']
                    )
                    db.add(new_activity)
                
                db.commit() # 이 타겟에 대한 모든 변경사항을 커밋
                logger.info(f"  Successfully loaded data for target: '{target_name}'")

            except Exception as e:
                logger.error(f"  Error processing target '{target_name}': {e}", exc_info=True)
                db.rollback() # 오류 발생 시 롤백
                logger.warning(f"  Rolled back transaction for target '{target_name}'.")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred during ETL process: {e}", exc_info=True)
    finally:
        db.close() # 세션 종료
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
