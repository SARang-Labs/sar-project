import sqlite3
import os

# --- 설정 ---
DATABASE_DIR = "database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "patent_data.db")

def setup_database_schema(database_path):
    """
    SAR 프로젝트를 위한 전체 데이터베이스 스키마를 생성하고 업데이트합니다.
    """
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    
    conn = None
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        print("데이터베이스 스키마 설정을 시작합니다...")

        # --- 기본 데이터 테이블 ---
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            compound_id INTEGER PRIMARY KEY AUTOINCREMENT,
            smiles TEXT UNIQUE NOT NULL
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS targets (
            target_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_name TEXT UNIQUE NOT NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patents (
            patent_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patent_number TEXT UNIQUE NOT NULL,
            title TEXT,
            publication_date DATE
        );
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS activities (
            activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            compound_id INTEGER,
            target_id INTEGER,
            patent_id INTEGER,
            ic50 REAL,
            pic50 REAL,
            activity_category TEXT,
            FOREIGN KEY (compound_id) REFERENCES compounds (compound_id),
            FOREIGN KEY (target_id) REFERENCES targets (target_id),
            FOREIGN KEY (patent_id) REFERENCES patents (patent_id)
        );
        """)
        print("✅ 기본 테이블(compounds, targets, patents, activities) 생성 완료.")

        # --- 분석 결과 저장 테이블 ---
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

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_hypotheses (
            hypothesis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            agent_name TEXT,
            hypothesis_text TEXT,
            context_info TEXT,
            hypothesis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES sar_analyses (analysis_id)
        );
        """)
        print("✅ 분석 결과 테이블(sar_analyses, ai_hypotheses) 생성 완료.")

        conn.commit()
        print("\n데이터베이스 스키마 설정이 성공적으로 완료되었습니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
        if conn: conn.rollback()
    finally:
        if conn: conn.close()

if __name__ == "__main__":
    setup_database_schema(DATABASE_PATH)
