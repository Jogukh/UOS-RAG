# VLM 기반 건축 도면 분석 시스템

AutoCAD PDF 도면에서 벡터 그래픽을 직접 분석하여 건축 요소를 추출하고 구조화된 데이터로 변환하는 AI 시스템

## 🎯 주요 기능

- **벡터 기반 도면 분석**: PyMuPDF를 이용한 직접적인 벡터 데이터 추출
- **VLM 패턴 인식**: Qwen2.5-VL 모델을 활용한 시각적 패턴 분석
- **LangGraph 워크플로우**: 체계적인 분석 파이프라인 관리
- **건축 요소 검출**: 벽, 문, 창문, 공간 등 자동 식별
- **신뢰도 기반 통합**: 다중 분석 결과의 지능적 결합

## 🏗️ 시스템 아키텍처

```
AutoCAD PDF → 벡터 데이터 추출 → 패턴 분석 → 구조화된 건축 데이터
              ↓                    ↑
              이미지 변환 → VLM 분석 ↗
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 준비

```bash
# models/ 폴더에 Qwen2.5-VL 모델 설치 필요
# 모델 다운로드는 별도 안내 참조
```

### 3. 기본 사용법

```python
from src.vlm_pattern_workflow_fixed import VLMPatternWorkflow

# 워크플로우 초기화
workflow = VLMPatternWorkflow()

# PDF 도면 분석
result = workflow.analyze_page("uploads/architectural-plan.pdf", page_number=0)

# 결과 확인
print(f"벽: {result['final_analysis']['summary']['total_walls']}개")
print(f"문: {result['final_analysis']['summary']['total_doors']}개")
print(f"창문: {result['final_analysis']['summary']['total_windows']}개")
print(f"공간: {result['final_analysis']['summary']['total_spaces']}개")
```

## 📁 프로젝트 구조

```
VLM/
├── src/
│   ├── architectural_vector_analyzer.py     # 벡터 기반 건축 분석기
│   ├── qwen_vlm_analyzer_clean.py          # VLM 분석기
│   ├── vlm_pattern_workflow_fixed.py       # LangGraph 통합 워크플로우
│   └── vlm_enhanced_pattern_analyzer.py    # VLM 향상 패턴 분석기
├── models/                                  # AI 모델들
├── uploads/                                # 테스트 PDF 파일들
├── vector_db/                             # 벡터 데이터베이스
└── requirements.txt                       # 의존성 목록
```

## 🔧 주요 구성 요소

### 1. 벡터 분석기 (`architectural_vector_analyzer.py`)
- PyMuPDF를 이용한 벡터 데이터 직접 추출
- 건축 요소별 패턴 매칭 알고리즘
- 기하학적 분석을 통한 구조 요소 식별

### 2. VLM 분석기 (`qwen_vlm_analyzer_clean.py`)
- Qwen2.5-VL 모델 기반 시각적 패턴 인식
- 최적화된 프롬프트 엔지니어링
- JSON 스키마 기반 구조화된 출력

### 3. 통합 워크플로우 (`vlm_pattern_workflow_fixed.py`)
- LangGraph 기반 상태 관리
- 병렬 처리를 통한 성능 최적화
- 오류 처리 및 신뢰도 검증

## 📊 성능 특징

- **정확도**: 벡터 + VLM 하이브리드 분석으로 높은 정확도
- **속도**: 병렬 처리 구조로 빠른 분석 시간
- **확장성**: LangGraph 기반 모듈식 설계
- **신뢰성**: 다중 검증 및 오류 복구 메커니즘

## 🎓 기술 스택

- **AI 모델**: Qwen2.5-VL (Vision-Language Model)
- **워크플로우**: LangChain + LangGraph
- **벡터 처리**: PyMuPDF + Shapely
- **컴퓨팅**: PyTorch + CUDA

## 📈 개발 현황

- ✅ 핵심 분석 엔진 완성
- ✅ LangGraph 워크플로우 통합
- ✅ VLM 모델 최적화
- 🔄 대용량 데이터셋 테스트 진행 중
- 📋 실시간 API 개발 예정

## 🤝 기여 방법

1. 이슈 등록
2. 기능 개발
3. 테스트 케이스 추가
4. 풀 리퀘스트 제출

## 📄 라이선스

MIT License

## 📞 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 등록해 주세요.

---

**최근 업데이트**: 2025년 6월 11일  
**버전**: v1.0 (베타)  
**상태**: 핵심 기능 구현 완료
