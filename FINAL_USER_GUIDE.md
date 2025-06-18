# 건축 도면 RAG 챗봇 - 최종 사용자 가이드

## 🏗️ 프로젝트 개요

이 프로젝트는 **DWG 건축 도면 파일을 자동으로 분석하고 질의응답이 가능한 RAG(Retrieval-Augmented Generation) 챗봇 시스템**입니다. 실제 건축 도면에서 메타데이터를 추출하고, 벡터 데이터베이스에 저장한 후, AI 언어 모델을 통해 자연어로 질문할 수 있습니다.

## 🎯 주요 기능

### 1. 완전 자동화된 DWG 처리 파이프라인
- **DWG → DXF 변환**: ODA File Converter를 사용한 배치 변환
- **메타데이터 추출**: ezdxf와 LLM을 활용한 지능형 메타데이터 추출
- **벡터 임베딩**: Jina AI v3를 사용한 고품질 임베딩 생성
- **벡터 DB 저장**: ChromaDB 기반 효율적인 저장 및 검색

### 2. 고급 RAG 챗봇 시스템
- **의미 검색**: 자연어 질문을 벡터 공간에서 검색
- **컨텍스트 기반 답변**: 검색된 문서를 바탕으로 한 정확한 답변 생성
- **대화 기록**: 세션별 대화 내용 저장 및 관리
- **다국어 지원**: 한국어 기반 질의응답

### 3. 모니터링 및 추적
- **LangSmith 통합**: 모든 LLM 호출과 성능 추적
- **상세 로깅**: 각 단계별 처리 과정 기록
- **오류 처리**: 강력한 예외 처리 및 폴백 메커니즘

## 🚀 시작하기

### 1. 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd /home/ubuntu-lynn/VLM

# 가상환경 활성화 (선택사항)
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일이 다음과 같이 구성되어 있는지 확인하세요:

```bash
# Ollama LLM 설정
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=gemma3:12b-it-qat
OLLAMA_TIMEOUT=60

# 임베딩 모델 설정
EMBEDDING_MODEL=jinaai/jina-embeddings-v3
EMBEDDING_DIMENSION=1024
HUGGING_FACE_API_KEY=your_hugging_face_api_key

# LangSmith 추적 설정
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=UOS-vllm-rag-chatbot
```

### 3. RAG 챗봇 실행

#### 대화형 모드 (추천)
```bash
python final_rag_chatbot.py
```

#### 데모 모드
```bash
python final_rag_chatbot.py --demo
```

#### 배치 테스트 모드
```bash
python rag_chatbot.py --mode test
```

## 💬 챗봇 사용법

### 기본 질문 예시

```
🙋 질문: 주동 입면도에 대해 설명해주세요
🙋 질문: 이 프로젝트에는 어떤 도면들이 있나요?
🙋 질문: 평면도와 입면도의 차이점은?
🙋 질문: 지하주차장 관련 도면이 있나요?
🙋 질문: 건축 설계 도면의 특징은?
```

### 챗봇 명령어

- `help` 또는 `도움말`: 사용법 가이드 표시
- `stats` 또는 `통계`: 시스템 정보 및 통계 표시
- `save` 또는 `저장`: 현재 대화 기록을 JSON 파일로 저장
- `clear` 또는 `초기화`: 화면 지우기
- `quit`, `exit`, `종료`: 챗봇 종료

## 🔧 고급 사용법

### 1. 새로운 DWG 파일 추가 처리

```bash
# 1. DWG 파일을 uploads/ 디렉토리에 복사
cp your_dwg_files/*.dwg uploads/new_project/

# 2. DWG → DXF 변환
python convert_dwg_to_dxf.py uploads/new_project/

# 3. 메타데이터 추출
python extract_metadata.py uploads/new_project/

# 4. 벡터 DB에 임베딩 및 저장
python src/metadata_vector_db.py --embed uploads/new_project/metadata/
```

### 2. 벡터 DB 관리

```bash
# 현재 DB 상태 확인
python debug_vector_db.py

# 새로운 컬렉션 생성
python src/metadata_vector_db.py --collection new_collection_name

# DB 초기화 (주의: 모든 데이터 삭제)
python src/metadata_vector_db.py --reset
```

### 3. 사용자 정의 프롬프트

`prompts/` 디렉토리의 YAML 파일을 수정하여 프롬프트를 커스터마이징할 수 있습니다:

- `rag_query.yaml`: RAG 답변 생성용 프롬프트
- `metadata_extraction.yaml`: 메타데이터 추출용 프롬프트
- `relationship_inference.yaml`: 관계 추론용 프롬프트

## 📊 시스템 구조

```
/home/ubuntu-lynn/VLM/
├── final_rag_chatbot.py          # 최종 RAG 챗봇 (메인)
├── rag_chatbot.py                # 기본 RAG 챗봇
├── src/
│   ├── metadata_vector_db.py     # 벡터 DB 관리
│   ├── embedding_config.py       # 임베딩 설정
│   ├── llm_metadata_extractor.py # LLM 메타데이터 추출기
│   └── prompt_manager.py         # 프롬프트 관리
├── chroma_db/                    # ChromaDB 데이터베이스
├── metadata/                     # 추출된 메타데이터 JSON 파일
├── uploads/                      # 업로드된 DWG 파일들
├── prompts/                      # 프롬프트 템플릿
└── logs/                        # 로그 파일
```

## 🎨 출력 화면 예시

RAG 챗봇은 컬러풀한 인터페이스를 제공합니다:

```
======================================================================
🏗️  건축 도면 RAG 챗봇 v2.0
   DWG 파일 메타데이터 기반 질의응답 시스템
======================================================================

💡 이 챗봇은 건축 도면에 대한 질문에 답변합니다.
📋 사용 가능한 명령어:
   • 'help' 또는 '도움말' - 사용법 보기
   • 'stats' 또는 '통계' - 시스템 정보 보기
   • 'save' 또는 '저장' - 대화 기록 저장

✅ RAG 시스템 초기화 완료
  📊 벡터 DB: 2개 문서
  🧠 LLM: gemma3:12b-it-qat
  🎯 컬렉션: test_architectural_metadata

🙋 질문: 주동 입면도에 대해 설명해주세요

🔍 검색 중...

🤖 답변:
주동 입면도에 대한 정보를 제공해 드리겠습니다...

📋 참조된 문서 (2개):
  1. 주동입면도 (관련도: 0.591)
  2. 대지 지구 적도 평면도 (관련도: 0.401)
```

## 🔍 성능 및 특징

### 벡터 검색 성능
- **임베딩 모델**: Jina AI v3 (1024차원)
- **검색 정확도**: 코사인 유사도 기반 의미 검색
- **응답 속도**: 평균 2-5초 (GPU 사용 시)

### LLM 성능
- **모델**: Ollama Gemma 3 12B (양자화)
- **컨텍스트 길이**: 최대 8K 토큰
- **언어**: 한국어 특화 답변 생성

### 확장성
- **다중 프로젝트**: 여러 건축 프로젝트 동시 관리 가능
- **대용량 데이터**: 수천 개의 도면 파일 처리 가능
- **실시간 추가**: 새로운 도면 파일의 실시간 추가 및 검색

## 🛠️ 문제 해결

### 자주 발생하는 문제

1. **Ollama 연결 실패**
   ```bash
   # Ollama 서비스 시작
   ollama serve
   
   # 모델 다운로드 확인
   ollama list
   ollama pull gemma3:12b-it-qat
   ```

2. **임베딩 모델 로딩 실패**
   ```bash
   # Hugging Face 로그인
   huggingface-cli login
   
   # CUDA 메모리 확인
   nvidia-smi
   ```

3. **벡터 DB 초기화 문제**
   ```bash
   # ChromaDB 재설정
   rm -rf chroma_db/
   python src/metadata_vector_db.py --reset
   ```

### 로그 확인

```bash
# 상세 로그 확인
tail -f logs/reranker.log

# LangSmith에서 LLM 호출 추적 확인
# https://smith.langchain.com/ 에서 프로젝트 확인
```

## 🚀 성능 최적화 팁

1. **GPU 사용**: CUDA가 활성화된 환경에서 실행
2. **배치 처리**: 여러 파일을 한 번에 처리
3. **캐싱**: 임베딩 결과 캐싱으로 재처리 방지
4. **메모리 관리**: 대용량 파일 처리 시 배치 크기 조절

## 📈 향후 개선 사항

- [ ] 웹 인터페이스 추가 (Gradio/Streamlit)
- [ ] 다중 언어 지원 확장
- [ ] 이미지 인식 기능 추가 (도면 스캔 이미지)
- [ ] API 서버 구성
- [ ] 데이터베이스 백업/복구 기능
- [ ] 사용자별 대화 세션 관리

## 📞 지원 및 문의

문제가 발생하거나 기능 개선 요청이 있으시면:

1. 로그 파일 확인 (`logs/` 디렉토리)
2. LangSmith 추적 정보 확인
3. GitHub Issues 또는 개발팀 문의

---

**🎉 축하합니다! 건축 도면 RAG 챗봇이 성공적으로 구축되었습니다.**

이제 실제 DWG 파일들을 업로드하고 AI와 대화하며 건축 도면에 대한 질문을 자유롭게 할 수 있습니다!
