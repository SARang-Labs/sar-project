# database.py 파일에서 정의한 Base와 engine을 가져옵니다.
from database import Base, engine, DATABASE_DIR
import os

def create_all_tables():
    """
    database.py에 정의된 모든 테이블을 데이터베이스에 생성합니다.
    """
    print("SQLAlchemy를 사용하여 테이블 생성을 시작합니다...")
    # 데이터베이스 디렉토리가 없으면 생성
    os.makedirs(DATABASE_DIR, exist_ok=True)
    
    # Base.metadata.create_all() 이 한 줄이 모든 CREATE TABLE 작업을 수행합니다.
    Base.metadata.create_all(bind=engine)
    
    print("✅ 모든 테이블이 성공적으로 생성 또는 확인되었습니다.")

if __name__ == "__main__":
    create_all_tables()
