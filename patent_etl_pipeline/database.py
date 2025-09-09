import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

# --- 설정 ---
DATABASE_PATH = "/Users/lionkim/Desktop/debate_app/sar-project/patent_etl_pipeline/database/patent_data.db"


# SQLAlchemy 엔진 생성
# check_same_thread는 SQLite를 사용할 때 필요한 옵션입니다.
engine = create_engine(
    f"sqlite:///{DATABASE_PATH}",
    connect_args={"check_same_thread": False}
)

# 데이터베이스 세션 생성을 위한 클래스
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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

def get_db():
    """데이터베이스 세션을 생성하고 반환하는 함수"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
