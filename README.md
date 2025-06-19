# 건축 PDF RAG 시스템 (Self-Query 지원)

건축 PDF 문서에서 메타데이터를 추출하고 Self-Query 기반 RAG (Retrieval-Augmented Generation) 시스템을 통해 지능적인 질의응답을 제공하는 시스템

## 🎯 주요 기능

- **PDF 메타데이터 추출**: 건축 도면 PDF에서 구조화된 메타데이터 자동 추출
- **Self-Query RAG**: 고급 필터링을 지원하는 Self-Query 기반 검색
- **ChromaDB 벡터 저장**: 고성능 벡터 데이터베이스를 통한 의미론적 검색
- **Ollama LLM 통합**: 로컬 LLM을 활용한 질의응답
- **LangSmith 추적**: AI 모델 성능 모니터링 및 추적
- **CLI 인터페이스**: 사용자 친화적인 명령줄 도구

## 🏗️ 시스템 아키텍처

```
PDF 문서 → 메타데이터 추출 → Self-Query 변환 → ChromaDB 저장
                                                    ↓
사용자 질의 → Self-Query 검색 → 관련 문서 검색 → Ollama LLM → 답변
                                                    ↑
                                           LangSmith 추적
```

## ⚡ 핵심 특징

- **Self-Query 지원**: 복잡한 필터링 조건을 자연어로 처리
- **메타데이터 기반 검색**: 면적, 층수, 건물 유형 등 구조화된 데이터 검색
- **실시간 추적**: LangSmith를 통한 모델 성능 모니터링
- **확장 가능**: 다양한 건축 프로젝트 지원

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

# .env 파일에서 다음 설정:
# - OLLAMA_MODEL: 사용할 Ollama 모델명
# - LANGSMITH_API_KEY: LangSmith API 키 (선택사항)
```

### 3. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (시스템에 맞게)
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드 (예: gemma3:12b-it-qat)
ollama pull gemma3:12b-it-qat
```

## 💡 사용법

### 1. RAG 데이터베이스 구축

```bash
# PDF 파일들을 uploads/ 폴더에 프로젝트별로 정리
# 예: uploads/부산장안지구/설계개요.pdf, 보일람표.pdf, ...

# RAG 데이터베이스 구축
python build_rag_db.py

# 특정 프로젝트만 처리
python build_rag_db.py --project_name="부산장안지구"
```

### 2. RAG 질의 실행

```bash
# 기본 질의
python query_rag.py "건축면적이 얼마인가요?"

# 특정 프로젝트 대상 질의
python query_rag.py "건축면적이 얼마인가요?" --project="부산장안지구"

# Self-Query 고급 검색 (복잡한 조건)
python query_rag.py "면적이 1000㎡ 이상인 건축물의 층수는?"

# 사용 가능한 프로젝트 목록 확인
python query_rag.py --list_projects
```

### 3. Self-Query 기능

Self-Query를 통해 다음과 같은 고급 검색이 가능합니다:

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
│   ├── pdf_metadata_extraction.yaml  # PDF 메타데이터 프롬프트
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
