# 특허 ETL 파이프라인 문서

## 프로젝트 개요

`patent_etl_pipeline` 프로젝트는 특허 출원서에 포함된 다양한 Excel 파일에서 생물학적 활성 데이터를 추출, 변환, 적재(ETL)하여 구조화된 SQL 데이터베이스로 만드는 시스템입니다. 목표는 다양한 소스의 데이터를 표준화하고 분석을 위해 쉽게 접근할 수 있도록 하는 것입니다.

## 파일 구조

프로젝트 디렉토리는 다음과 같이 구성됩니다:
```
patent_etl_pipeline/
├── database/
│   └── patent_data.db
├── data_processor.py
├── parsers.py
├── run_etl.py
└── etl.log
```

## 데이터베이스 스키마

SQLite 데이터베이스(`patent_data.db`)는 다음과 같은 테이블을 가집니다:

- **compounds**: 고유한 화학 화합물을 저장합니다.
  - `compound_id` (INTEGER PRIMARY KEY AUTOINCREMENT): 각 화합물의 고유 식별자
  - `smiles` (TEXT UNIQUE NOT NULL): 화합물 구조를 나타내는 SMILES 문자열

- **targets**: 고유한 생물학적 타겟을 저장합니다.
  - `target_id` (INTEGER PRIMARY KEY AUTOINCREMENT): 각 타겟의 고유 식별자
  - `target_name` (TEXT UNIQUE NOT NULL): 생물학적 타겟 이름 (예: KU-19-19, EGFR)

- **activities**: 화합물과 타겟을 연결하는 활성 데이터를 저장합니다.
  - `activity_id` (INTEGER PRIMARY KEY AUTOINCREMENT): 각 활성 측정의 고유 식별자
  - `compound_id` (INTEGER): `compounds` 테이블을 참조하는 외래키
  - `target_id` (INTEGER): `targets` 테이블을 참조하는 외래키
  - `ic50` (REAL): μM 단위의 IC50 값
  - `pic50` (REAL): 계산된 pIC50 값
  - `activity_category` (TEXT): 분류된 활성 수준 (예: 'Highly Active', 'Moderately Active')

## 스크립트 설명

- **`data_processor.py`**: 개별 Excel 시트를 처리하고, IC50 데이터를 추출 및 변환하며, pIC50와 활성 카테고리를 계산하는 핵심 로직을 포함합니다. 이 함수는 파서 스크립트에서 사용되도록 설계되었습니다.

- **`parsers.py`**: 4개의 특허 Excel 파일 각각에 대한 특정 파서 함수를 포함합니다. 이 함수들은 `data_processor.py`를 활용하여 각각의 파일 형식에서 관련 데이터를 추출하고 처리하여 처리된 DataFrame의 사전을 반환합니다.

- **`run_etl.py`**: ETL 파이프라인을 조정하는 메인 스크립트입니다. 데이터베이스에 연결하고, 정의된 Excel 파일들을 반복 처리하며, 각 파일에 대해 적절한 파서 함수를 호출하고, 처리된 데이터를 데이터베이스 테이블(`compounds`, `targets`, `activities`)에 로드합니다. 에러 처리와 로깅을 포함합니다.

## 사용법

1. 4개의 특허 Excel 파일을 `/content/` 디렉토리에 배치합니다 (또는 `run_etl.py`의 `excel_files_dir` 변수를 업데이트합니다).
2. 필요한 라이브러리가 설치되어 있는지 확인합니다:
   ```bash
   pip install pandas openpyxl numpy
   ```
3. ETL 파이프라인 스크립트를 실행합니다:
   ```bash
   python patent_etl_pipeline/run_etl.py
   ```
스크립트는 각 파일을 처리하고, SQLite 데이터베이스(`database` 디렉토리의 `patent_data.db`)에 데이터를 로드하며, 진행 상황과 오류를 `etl.log`에 기록합니다.

## 에러 처리 및 로깅

`run_etl.py` 스크립트는 ETL 프로세스를 추적하기 위한 로깅을 포함합니다. 정보, 경고, 오류가 콘솔과 `patent_etl_pipeline` 디렉토리 내의 `etl.log` 파일에 기록됩니다. 이는 파이프라인의 실행을 모니터링하고 문제를 디버깅하는 데 도움이 됩니다.

파일을 찾을 수 없는 오류, 데이터베이스 오류, 행 처리 중 오류에 대한 특정 에러 처리가 구현되어 있습니다.

## 확장성

새로운 Excel 파일에 대한 지원을 추가하려면:

1. `parsers.py`에서 `parse_file_1`과 유사한 새로운 파서 함수를 만들어 새 파일의 구조를 처리합니다.
2. `run_etl.py`의 `excel_files_to_process` 목록에 새 파일명을 추가합니다.
3. `run_etl.py`의 `parser_mapping` 사전에 새 파일명에서 새 파서 함수로의 매핑을 추가합니다.

새 파일이 다른 유형의 데이터를 포함하거나 다른 처리 로직이 필요한 경우, `data_processor.py`를 업데이트하거나 새로운 처리 함수를 만들어야 할 수 있습니다.