
import os
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    ForeignKey, Text, Date, MetaData, text
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import datetime

# --- 설정 ---
# 이 파일(setup_database.py)의 위치를 기준으로 경로를 설정합니다.
# 이 스크립트가 patent_etl_pipeline 폴더 안에 있다고 가정합니다.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(SCRIPT_DIR, "database")
DATABASE_NAME = "patent_data.db"
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

# SQLAlchemy 엔진 생성
engine = create_engine(
    f"sqlite:///{DATABASE_PATH}",
    connect_args={"check_same_thread": False}
)

# 모든 모델 클래스가 상속받을 기본 클래스
Base = declarative_base()


# --- 테이블 모델 정의 (파이썬 클래스) ---

class Patent(Base):
    __tablename__ = "patents"
    patent_id = Column(Integer, primary_key=True, index=True)
    patent_number = Column(String, unique=True, nullable=False, index=True)
    title = Column(String)
    publication_date = Column(Date)

class Compound(Base):
    __tablename__ = "compounds"
    compound_id = Column(Integer, primary_key=True, index=True)
    smiles = Column(String, unique=True, nullable=False, index=True)

class Target(Base):
    __tablename__ = "targets"
    target_id = Column(Integer, primary_key=True, index=True)
    target_name = Column(String, unique=True, nullable=False, index=True)

class Activity(Base):
    __tablename__ = "activities"
    activity_id = Column(Integer, primary_key=True, index=True)
    compound_id = Column(Integer, ForeignKey("compounds.compound_id"))
    target_id = Column(Integer, ForeignKey("targets.target_id"))
    patent_id = Column(Integer, ForeignKey("patents.patent_id"))
    ic50 = Column(Float)
    pic50 = Column(Float)
    activity_category = Column(String)

class SAR_Analysis(Base):
    __tablename__ = "sar_analyses"
    analysis_id = Column(Integer, primary_key=True, index=True)
    patent_id = Column(Integer, ForeignKey("patents.patent_id")) # patent_id 추가
    compound_id_1 = Column(Integer, ForeignKey("compounds.compound_id"), nullable=False)
    compound_id_2 = Column(Integer, ForeignKey("compounds.compound_id"), nullable=False)
    similarity = Column(Float)
    activity_difference = Column(Float)
    score = Column(Float)
    analysis_timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class AI_Hypothesis(Base):
    __tablename__ = "ai_hypotheses"
    hypothesis_id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("sar_analyses.analysis_id"), nullable=False)
    agent_name = Column(String)
    hypothesis_text = Column(Text)
    context_info = Column(Text) # JSON string
    hypothesis_timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def update_db_schema():
    """
    database.py에 정의된 모델을 바탕으로 데이터베이스 스키마를 생성하거나 업데이트합니다.
    """
    print("SQLAlchemy를 사용하여 데이터베이스 스키마를 확인 및 업데이트합니다...")
    os.makedirs(DATABASE_DIR, exist_ok=True)
    
    Base.metadata.create_all(bind=engine)
    
    with engine.connect() as connection:
        # PRAGMA 쿼리 실행
        result = connection.execute(text("PRAGMA table_info(sar_analyses);"))
        
        columns = [row[1] for row in result]
        if 'patent_id' not in columns:
            print("  'sar_analyses' 테이블에 'patent_id' 컬럼을 추가합니다...")

            connection.execute(text('ALTER TABLE sar_analyses ADD COLUMN patent_id INTEGER REFERENCES patents(patent_id)'))
            connection.commit()

            print("  ✅ 컬럼 추가 완료.")
        else:
            print("  'sar_analyses' 테이블에 'patent_id' 컬럼이 이미 존재합니다.")
            
    print("\n✅ 모든 테이블이 성공적으로 생성 또는 확인되었습니다.")

if __name__ == "__main__":
    update_db_schema()