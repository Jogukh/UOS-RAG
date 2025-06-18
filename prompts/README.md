# 📝 VLM 프롬프트 관리 가이드

이 디렉토리는 VLM 건축 도면 분석 시스템에서 사용하는 모든 LLM 프롬프트를 관리합니다.

## 📁 디렉토리 구조

```
prompts/
├── README.md                    # 이 파일
├── config.yaml                 # 프롬프트 시스템 설정
├── metadata_extraction.yaml    # 메타데이터 추출 프롬프트
├── relationship_inference.yaml # 관계 추론 프롬프트
├── text_analysis.yaml         # 텍스트 분석 프롬프트
├── rag_query.yaml             # RAG 질의응답 프롬프트
├── system_test.yaml           # 시스템 테스트 프롬프트
└── gemma_chat_wrapper.yaml    # Gemma 채팅 래퍼 프롬프트
```

## 🔧 프롬프트 파일 형식

각 프롬프트 파일은 YAML 형식으로 작성되며 다음 구조를 따릅니다:

```yaml
# 프롬프트 설명 (주석)
name: "프롬프트 이름"
type: "프롬프트 유형"
description: "프롬프트 설명"
used_by:
  - "사용하는 모듈명1.py"
  - "사용하는 모듈명2.py"
input_params:
  - "파라미터1"
  - "파라미터2"
output_format: "예상 출력 형식"
version: "1.0"

template: |
  실제 프롬프트 내용
  {파라미터1}을 사용한 템플릿
  {파라미터2}도 포함 가능
```

## 📝 프롬프트 수정 가이드

### 1. 기존 프롬프트 수정
원하는 프롬프트 파일을 직접 편집하면 됩니다.

```bash
# 예: 메타데이터 추출 프롬프트 수정
vim prompts/metadata_extraction.yaml
```

### 2. 새 프롬프트 추가
1. 새 YAML 파일 생성
2. `config.yaml`의 `prompt_files` 섹션에 추가
3. 필요시 `prompt_types`에 새 유형 추가

### 3. 파라미터 변경
- `input_params` 리스트 수정
- `template`에서 해당 파라미터 사용/제거
- 관련 코드에서도 파라미터 변경 필요

## 🎯 프롬프트별 상세 설명

### metadata_extraction.yaml
- **목적**: PDF 텍스트에서 건축 도면 메타데이터 추출
- **사용 모듈**: `llm_metadata_extractor.py`
- **출력**: JSON 형식의 구조화된 메타데이터
- **핵심 파라미터**: `file_name`, `page_number`, `text_content`

### relationship_inference.yaml
- **목적**: 두 건축 도면 간의 관계 분석
- **사용 모듈**: `llm_relationship_inferencer.py`
- **출력**: 관계유형, 관계강도, 관계설명
- **핵심 파라미터**: 두 도면의 각종 메타데이터

### text_analysis.yaml
- **목적**: 도면 텍스트에서 다른 도면 참조 찾기
- **사용 모듈**: `llm_relationship_inferencer.py`
- **출력**: 참조도면, 참조내용
- **핵심 파라미터**: `drawing_text`, `other_drawings_info`

### rag_query.yaml
- **목적**: RAG 시스템 질의응답
- **사용 모듈**: `query_rag.py`
- **출력**: 자연어 답변
- **핵심 파라미터**: `retrieved_documents_text`, `query_text`

### system_test.yaml
- **목적**: 시스템 연결 테스트
- **사용 모듈**: 테스트 스크립트들
- **출력**: 간단한 인사 응답
- **파라미터**: 없음

### gemma_chat_wrapper.yaml
- **목적**: Gemma 모델용 채팅 형식 래핑
- **사용 모듈**: `query_rag.py`
- **출력**: 채팅 형식 텍스트
- **핵심 파라미터**: `system_message`, `user_message`

## 🔄 프롬프트 버전 관리

### 버전 업데이트
프롬프트를 수정할 때는 `version` 필드를 업데이트하세요:

```yaml
version: "1.1"  # 마이너 수정
version: "2.0"  # 메이저 변경
```

### 백업 권장사항
중요한 변경 전에는 백업을 생성하세요:

```bash
cp metadata_extraction.yaml metadata_extraction.yaml.backup
```

## 🧪 프롬프트 테스트

프롬프트 변경 후 반드시 테스트하세요:

```bash
# 프롬프트 매니저 테스트
python src/prompt_manager.py

# 전체 시스템 테스트
python test_ollama_integration.py
```

## ⚠️ 주의사항

1. **파라미터 일치**: 코드에서 사용하는 파라미터와 YAML 파일의 `input_params`가 일치해야 합니다.

2. **특수 문자**: YAML에서 `{`, `}` 등은 이스케이프가 필요할 수 있습니다.

3. **인코딩**: 모든 파일은 UTF-8로 저장하세요.

4. **들여쓰기**: YAML은 들여쓰기에 민감하므로 주의하세요.

## 🔗 관련 파일

- `src/prompt_manager.py`: 프롬프트 로더 및 관리자
- `src/llm_metadata_extractor.py`: 메타데이터 추출 모듈
- `src/llm_relationship_inferencer.py`: 관계 추론 모듈
- `query_rag.py`: RAG 질의응답 모듈

---

💡 **팁**: 프롬프트를 수정할 때는 작은 변경부터 시작해서 점진적으로 개선하는 것이 좋습니다.
