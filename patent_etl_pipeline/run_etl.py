import os
import pandas as pd
import sys
import logging
import argparse
from sqlalchemy.orm import Session
from .database import SessionLocal, Patent, Compound, Target, Activity

# --- 설정 ---

LOG_FILE = "etl.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[ logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout) ]
)
logger = logging.getLogger(__name__)

# 현재 디렉토리를 경로에 추가하여 parsers 모듈을 임포트합니다.
sys.path.append('.')
try:
    from . import parsers
    logger.info("Successfully imported 'parsers' module.")
except ImportError:
    logger.error("Could not import 'parsers' module.", exc_info=True)
    sys.exit(1)

# 경로 설정
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

def run_etl(patent_number, file_source, progress_bar=None):
    """
    SQLAlchemy를 사용하여 ETL 프로세스를 실행합니다.
    file_source는 파일 경로(str) 또는 메모리 내 파일(BytesIO)일 수 있습니다.
    """
    db: Session = SessionLocal() # SQLAlchemy 세션 시작
    logger.info(f"Starting ETL process for patent '{patent_number}'...")
    try:
        # 1. 특허 번호를 patents 테이블에 삽입/조회합니다.
        patent = get_or_create(db, Patent, patent_number=patent_number)
        logger.info(f"Using patent_id: {patent.patent_id} for patent_number: '{patent.patent_number}'.")

        if isinstance(file_source, str):
            excel_file_name_for_mapping = os.path.basename(file_source)
        else:
            excel_file_name_for_mapping = "1020170094694_extracted_250611.xlsx"

        # 2. 파일 이름에 맞는 파서를 선택하고 실행합니다.
        parser_mapping = {
            "1020170094694_extracted_250611.xlsx": parsers.parse_file_1,
            "file2.xlsx": parsers.parse_file_2,
            "file3.xlsx": parsers.parse_file_3,
            "file4.xlsx": parsers.parse_file_4,
        }
        parser_func = parser_mapping.get(excel_file_name_for_mapping)

        if not parser_func:
            message = f"'{excel_file_name_for_mapping}'에 대한 파서가 정의되지 않았습니다."
            logger.warning(message)
            return False, message

        processed_dataframes = parser_func(file_source)

        # 3. 변환된 데이터를 DB에 저장합니다.
        total_targets = len(processed_dataframes)
        for i, (target_name, df) in enumerate(processed_dataframes.items()):
            logger.info(f"  Loading data for target: '{target_name}'")
            try:
                target = get_or_create(db, Target, target_name=target_name)
                for index, row in df.iterrows():
                    compound = get_or_create(db, Compound, smiles=row['SMILES'])
                    new_activity = Activity(
                        compound_id=compound.compound_id,
                        target_id=target.target_id,
                        patent_id=patent.patent_id,
                        ic50=row.get(f'{target_name} IC50'),
                        pic50=row.get(f'{target_name} pIC50'),
                        activity_category=row['Activity']
                    )
                    db.add(new_activity)
                db.commit()
                logger.info(f"  Successfully loaded data for target: '{target_name}'")
                if progress_bar:
                    progress_bar.progress((i + 1) / total_targets, text=f"타겟 처리 중: {target_name}")
            except Exception as e:
                db.rollback()
                logger.error(f"  Error processing target '{target_name}': {e}", exc_info=True)

        return True, f"Successfully processed {len(processed_dataframes)} targets for patent {patent_number}."
    except Exception as e:
        db.rollback()
        logger.error(f"An unexpected error occurred during ETL process: {e}", exc_info=True)
        return False, str(e)
    finally:
        db.close()
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
        success, message = run_etl(args.patent_number, excel_file_full_path)
        if not success:
            logger.error(f"ETL process failed: {message}")