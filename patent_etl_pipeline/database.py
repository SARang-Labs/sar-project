import os
import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    ForeignKey, Text, Date
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- 설정: 이 파일의 위치를 기준으로 DB 경로를 설정하여 일관성 유지 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(SCRIPT_DIR, "database")
DATABASE_NAME = "patent_data.db"
DATABASE_PATH = os.path.join(DATABASE_DIR, DATABASE_NAME)

# SQLAlchemy 엔진 생성
engine = create_engine(
    f"sqlite:///{DATABASE_PATH}",
    connect_args={"check_same_thread": False}
)

# 데이터베이스 세션 생성을 위한 클래스
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 모든 모델 클래스가 상속받을 기본 클래스
Base = declarative_base()


# --- 테이블 모델 정의 (데이터베이스 스키마) ---
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
    patent_id = Column(Integer, ForeignKey("patents.patent_id"))
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

# --- DB 생성 함수 ---
def init_db():
    """
    database.py에 정의된 모든 테이블을 데이터베이스에 생성합니다.
    """
    print("데이터베이스 스키마를 확인하고 생성합니다...")
    os.makedirs(DATABASE_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    print("✅ 데이터베이스 스키마 준비 완료.")