# AI 기반 구조-활성 관계 분석 시스템

> **신약 개발 연구를 위한 Activity Cliff 탐지 자동화 및 구조-활성 관계(SAR) 분석 플랫폼**


### 🏢 프로젝트 정보

본 프로젝트는 [**아이젠사이언스**](https://aigensciences.com/)와의 협업을 통해 개발되었습니다.  
실제 신약 개발 현장의 요구사항을 반영하여 연구자들이 효율적으로 SAR 분석을 수행할 수 있도록 설계되었습니다.

<br>

<img width="922" height="834" alt="SAR랑 연구소 README 이미지" src="https://github.com/user-attachments/assets/3171b782-2fc5-4efa-b6e4-6baa6e91efb9" />


<br>
<br>


## 📋 필수 요구사항

- Python 3.11
- Conda (Miniforge 권장)
- OpenAI API 키 또는 Google Gemini API 키

<br>

## 🚀 설치 및 실행

### 1. 프로젝트 클론

```bash
git clone https://github.com/SARang-Labs/sar-project.git
cd sar-project
```

### 2. Conda 환경 설정 (Apple Silicon Mac)

도킹 시뮬레이션을 위해 Intel(x86_64) 기반 가상환경이 필요합니다.

> ⚠️ 이 문서는 Apple Silicon Mac을 기준으로 작성되었습니다.
> - **Intel Mac/Windows/Linux 사용자**: `conda create -n sar-env python=3.11` 명령으로 일반 가상환경 생성
> - **Apple Silicon Mac 사용자**: 아래 명령어를 따라 Intel 호환 환경 설정 필요

```bash
# Miniforge 설치 (없는 경우)
brew install miniforge

# Intel 기반 가상환경 생성
CONDA_SUBDIR=osx-64 conda create -n sar-env-x86 python=3.11

# 환경 활성화
conda activate sar-env-x86
```

### 3. 패키지 설치

```bash
# Conda로 과학 컴퓨팅 패키지 설치
conda install -c conda-forge vina openbabel

# pip로 나머지 패키지 설치
pip install -r requirements.txt
```

### 4. 앱 실행

```bash
# Conda 환경 활성화 (매번 실행 시)
conda activate sar-env-x86

# Streamlit 앱 실행
streamlit run app.py
```

브라우저가 자동으로 열리며 `http://localhost:8501`에서 앱을 사용할 수 있습니다.

<br>

## 💡 사용 방법

### 데이터 관리

특허 데이터를 데이터베이스에 저장하여 분석합니다:
1. **특허 데이터 등록**: 특허 번호와 엑셀 파일(SMILES, pIC50/pKi 정보 포함)을 업로드
2. **자동 ETL 처리**: 데이터가 자동으로 파싱되어 데이터베이스에 저장

### 주요 기능

- **Activity Cliff 탐지**: 구조적으로 유사하지만 활성도가 크게 다른 화합물 쌍 자동 발견
- **분자 구조 시각화**: 화합물 간 구조적 차이점을 하이라이팅하여 표시
- **도킹 시뮬레이션**: 타겟 단백질과의 결합 예측을 통한 과학적 근거 제공
- **전문가 협업 시스템**: 4명의 AI 전문가의 다각도 분석 및 통합
- **최종 가설 생성**: GPT-4 또는 Gemini를 활용한 구조-활성 관계 해석
- **분석 결과 자동 저장**: 모든 분석 결과를 자동으로 데이터베이스에 저장하고 추후 조회 가능

<br>

> **📄 [라이선스] 본 프로젝트는 아이젠사이언스와의 협업 프로젝트입니다.**
