# 🏗️ 건축 도면 RAG 질의 시스템

건축 도면 PDF/DWG 파일에서 메타데이터를 추출하고 OpenAI 기반 RAG (Retrieval-Augmented Generation) 시스템을 통해 지능적인 질의응답을 제공하는 통합 시스템

## 🎯 주요 기능

- **멀티포맷 지원**: PDF, DWG, DXF 파일 자동 처리
- **OpenAI 임베딩**: `text-embedding-3-small` 모델로 고품질 벡터 검색
- **ChromaDB 벡터 저장**: 프로젝트별 컬렉션 관리
- **CLI + 노트북 통합**: 명령줄 도구와 Jupyter 노트북 인터페이스
- **이미지 연동**: 검색 결과와 관련된 도면 이미지 자동 표시
- **LangSmith 추적**: AI 모델 성능 모니터링 및 디버깅

## 🏗️ 시스템 아키텍처

```
PDF/DWG 문서 → 메타데이터 추출 → OpenAI 임베딩 → ChromaDB 저장
                                                    ↓
사용자 질의 → 벡터 검색 → 관련 문서 + 이미지 → OpenAI LLM → 답변
                                                    ↑
                                           LangSmith 추적
```

## ⚡ 핵심 특징

- **프로젝트별 관리**: 각 건축 프로젝트를 독립적인 컬렉션으로 관리
- **통합 인터페이스**: CLI 도구와 노트북을 통한 유연한 사용
- **이미지 자동 연동**: 검색된 도면과 관련된 PNG 이미지 자동 표시
- **성능 추적**: 실행 시간, 성공률 등 상세한 성능 분석

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일에서 다음 설정 필수:
# - OPENAI_API_KEY: OpenAI API 키
# - DEFAULT_LLM_PROVIDER: openai
# - DEFAULT_EMBEDDING_PROVIDER: openai
```

### 3. 데이터 준비

```bash
# PDF/DWG 파일들을 uploads/ 폴더에 프로젝트별로 정리
# 예: uploads/부산장안_프로젝트도면/지하주차장_구조평면도.pdf
#     uploads/부산장안_프로젝트정보/설계개요.pdf
```

## 💡 사용법

### 1. 메타데이터 추출

```bash
# PDF/DWG 파일에서 메타데이터 추출
python extract_metadata_unified.py --project 부산장안_프로젝트도면

# LLM 기반 Self-Query 형태로 변환
python extract_metadata_unified.py --project 부산장안_프로젝트도면 --convert_to_self_query
```

### 2. RAG 데이터베이스 구축

```bash
# 모든 프로젝트 RAG DB 구축
python build_rag_db_v2.py

# 특정 프로젝트만 처리
python build_rag_db_v2.py --project 부산장안_프로젝트도면

# 기존 컬렉션 목록 확인
python build_rag_db_v2.py --list
```

### 3. RAG 질의 실행

```bash
# 기본 질의
python query_rag.py "지하주차장 구조평면도를 찾아주세요"

# 특정 프로젝트 대상 질의
python query_rag.py "구조평면도" -p 부산장안_프로젝트도면 -n 5

# LLM 답변 없이 검색만
python query_rag.py "화장실 배치도" --no_llm

# 사용 가능한 프로젝트 목록 확인
python query_rag.py --list_projects
```

### 4. 노트북 인터페이스

```python
# Jupyter 노트북에서 사용
from chatbot import simple_ask

# 간단한 질의
simple_ask("구조평면도에 대해 알려주세요")

# 특정 프로젝트, 이미지 포함
simple_ask("지하주차장 배치는 어떻게 되나요?", 
          project="부산장안_프로젝트도면", 
          show_images=True)
```

## ✨ 주요 개선사항

- **OpenAI 통합**: 높은 품질의 임베딩과 LLM 답변
- **이미지 연동**: 검색 결과와 관련된 도면 이미지 자동 표시  
- **프로젝트별 관리**: 각 프로젝트를 독립적인 컬렉션으로 구성
- **통합 인터페이스**: CLI와 노트북 모두에서 동일한 기능 사용

- **면적 기반 검색**: "건축면적이 500㎡ 이상인 건물"
- **층수 기반 검색**: "3층 이상인 건축물"
- **건물 유형**: "아파트 유형의 건축물"
- **복합 조건**: "면적이 1000㎡ 이상이고 5층 이하인 건물"

### 4. LangSmith 추적 확인

```bash
# LangSmith 웹 콘솔에서 추적 결과 확인
# https://smith.langchain.com/projects
```

## 📁 프로젝트 구조

```
VLM/
├── build_rag_db.py          # RAG 데이터베이스 구축 스크립트
├── query_rag.py             # RAG 질의 실행 스크립트
├── requirements.txt         # Python 의존성
├── .env.example            # 환경변수 템플릿
├── src/                    # 소스 코드
│   ├── self_query_config.py      # Self-Query 설정
│   ├── convert_to_self_query.py  # 메타데이터 변환
│   ├── llm_metadata_extractor.py # 메타데이터 추출
│   ├── metadata_vector_db.py     # 벡터 DB 관리
│   └── prompt_manager.py         # 프롬프트 관리
├── prompts/                # AI 프롬프트
│   ├── metadata_extraction.yaml      # 메타데이터 추출 프롬프트
│   ├── metadata_extraction.yaml      # 통합 메타데이터 추출 프롬프트
│   └── rag_query.yaml              # RAG 질의 프롬프트
├── uploads/                # PDF 업로드 폴더
└── chroma_db/             # ChromaDB 데이터
```

## 🔧 기술 스택

- **LangChain**: RAG 파이프라인 구축
- **ChromaDB**: 벡터 데이터베이스
- **Ollama**: 로컬 LLM 실행
- **LangSmith**: AI 모델 추적 및 모니터링
- **Sentence Transformers**: 텍스트 임베딩
- **PyMuPDF**: PDF 문서 처리

## 📋 요구사항

- Python 3.8+
- Ollama (로컬 LLM 서버)
- 8GB+ RAM (권장)
- LangSmith API 키 (선택사항)

## 🤝 기여 방법

1. 프로젝트를 포크합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

**최근 업데이트**: 2025년 6월 19일  
**버전**: v2.0 (Self-Query RAG)  
**상태**: 프로덕션 준비 완료
