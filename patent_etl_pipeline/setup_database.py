import sqlite3

db_path = "/Users/lionkim/Downloads/project_archive/sar-project/patent_etl_pipeline/database/patent_data.db"

def create_results_tables(database_path):
    """
    SAR 분석 결과와 AI 가설 저장을 위한 테이블을 생성합니다.
    이미 테이블이 존재하면 생성하지 않습니다.
    """
    conn = None
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # 1. SAR 분석 결과 저장 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sar_analyses (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            compound_id_1 INTEGER NOT NULL,
            compound_id_2 INTEGER NOT NULL,
            similarity REAL,
            activity_difference REAL,
            score REAL,
            analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (compound_id_1) REFERENCES compounds (compound_id),
            FOREIGN KEY (compound_id_2) REFERENCES compounds (compound_id)
        );
        """)
        print("✅ 'sar_analyses' 테이블 생성 또는 확인 완료.")

        # 2. AI 가설 저장 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_hypotheses (
            hypothesis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            agent_name TEXT,
            hypothesis_text TEXT,
            context_info TEXT, -- RAG로 수집된 참고 문헌 정보 (JSON 형태의 텍스트로 저장)
            hypothesis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES sar_analyses (analysis_id)
        );
        """)
        print("✅ 'ai_hypotheses' 테이블 생성 또는 확인 완료.")

        conn.commit()

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_results_tables(db_path)
