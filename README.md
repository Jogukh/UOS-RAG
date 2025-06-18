# VLM 기반 건축 도면 분석 시스템 (vLLM 최적화)

AutoCAD PDF 도면에서 벡터 그래픽을 직접 분석하여 건축 요소를 추출하고 구조화된 데이터로 변환하는 고성능 AI 시스템

## 🎯 주요 기능

- **벡터 기반 도면 분석**: PyMuPDF를 이용한 직접적인 벡터 데이터 추출
- **고성능 VLM 추론**: vLLM을 활용한 최적화된 비전-언어 모델 처리
- **Qwen2.5-VL 통합**: 최신 멀티모달 모델을 통한 정확한 패턴 분석
- **LangGraph 워크플로우**: 체계적인 분석 파이프라인 관리
- **건축 요소 검출**: 벽, 문, 창문, 공간 등 자동 식별
- **신뢰도 기반 통합**: 다중 분석 결과의 지능적 결합
- **배치 처리 지원**: 다중 도면 동시 분석
- **성능 최적화**: GPU 메모리 효율적 사용 및 처리량 향상

## 🏗️ 시스템 아키텍처

```
AutoCAD PDF → 벡터 데이터 추출 → 패턴 분석 → 구조화된 건축 데이터
              ↓                    ↑
              이미지 변환 → vLLM 추론 ↗
                          (Qwen2.5-VL)
```

## ⚡ vLLM 최적화 특징

- **고속 추론**: transformers 대비 최대 5-10배 속도 향상
- **메모리 효율성**: PagedAttention으로 GPU 메모리 최적 사용
- **배치 처리**: 동시 다중 요청 처리로 처리량 극대화
- **자동 스케일링**: GPU 리소스에 따른 자동 설정 최적화
- **Fallback 지원**: vLLM 불가 시 transformers 자동 전환

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 의존성 설치 (vLLM 포함)
pip install -r requirements.txt

# vLLM 수동 설치 (권장)
pip install vllm>=0.7.2
```

### 2. vLLM 설정 최적화

```bash
# 시스템에 맞는 vLLM 설정 자동 생성
python src/vllm_config.py

# 생성된 설정 파일들:
# - configs/vllm_auto.json (자동 감지)
# - configs/vllm_throughput.json (처리량 최적화)
# - configs/vllm_latency.json (지연시간 최적화)
# - configs/vllm_memory.json (메모리 최적화)
```

### 3. LangSmith 추적 설정 (선택사항)

```bash
# 1. LangSmith API 키 발급
# https://smith.langchain.com/settings 에서 API 키 생성

# 2. .env 파일 설정
cp .env.example .env
# .env 파일에서 다음 설정:
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=VLM-Architecture-Analysis

# 3. LangSmith 통합 테스트
python test_langsmith_integration.py

# 4. 추적 결과 확인
# https://smith.langchain.com/projects/p/VLM-Architecture-Analysis
```

#### LangSmith 추적 혜택
- 🔍 **실시간 모니터링**: 모든 LLM 호출과 워크플로우 단계 추적
- 📊 **성능 분석**: 응답 시간, 토큰 사용량, 비용 분석
- 🐛 **디버깅 지원**: 오류 발생 시 상세한 컨텍스트 제공
- 📈 **개선 지표**: 모델 성능과 정확도 추이 모니터링

### 4. vLLM-LangChain OpenAI API 연동

이 프로젝트는 vLLM 서버를 OpenAI API 형태로 실행하여 LangChain과 연동합니다.

#### vLLM 서버 시작
```bash
# 1. vLLM 서버 시작 (별도 터미널에서 실행)
# Option A: Gemma-3 12B GGUF (권장 - 메모리 효율적)
./start_vllm_gemma.sh

# Option B: 수동으로 Gemma 시작
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-3-12b-it-qat-q4_0-gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192

# Option C: 다른 모델 (더 큰 GPU 메모리 필요)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Reranker-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192

# 2. 서버 상태 확인
curl http://localhost:8000/v1/models
```

#### 지원되는 모델
- **google/gemma-3-12b-it-qat-q4_0-gguf** (권장): 4-bit 양자화, 8GB GPU에 최적화
- **hyokwan/llama31_common**: Llama 3.1 기반 모델
- **Qwen/Qwen3-Reranker-4B**: 리랭킹 전용 모델 (더 큰 메모리 필요)

#### LangChain 연동 사용
```python
from src.llm_metadata_extractor import LLMMetadataExtractor
from src.llm_relationship_inferencer import LLMDrawingRelationshipInferencer

# 메타데이터 추출기 (자동으로 vLLM 서버 연결)
extractor = LLMMetadataExtractor()

# 건축 도면 텍스트에서 메타데이터 추출
sample_text = """
도면번호: A01-001
도면명: 1층 평면도
축척: 1/100
거실: 20.5㎡
주방: 12.3㎡
"""

metadata = extractor.extract_metadata_from_text(
    sample_text, 
    "building_plan.pdf", 
    1
)

print(f"도면번호: {metadata['drawing_number']}")
print(f"도면제목: {metadata['drawing_title']}")
```

#### 연동 테스트
```bash
# vLLM-LangChain 연동 테스트
python test_vllm_langchain_integration.py
```

### 5. 기본 사용법

#### vLLM 분석기 사용 (Legacy)
```python
from src.vllm_analyzer import VLLMAnalyzer

# vLLM 분석기 초기화 (자동 최적화)
analyzer = VLLMAnalyzer(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8
)

# 모델 로드
analyzer.load_model()

# 이미지 분석
result = workflow.analyze_page("uploads/architectural-plan.pdf", page_number=0)

# 결과 확인
print(f"벽: {result['final_analysis']['summary']['total_walls']}개")
print(f"문: {result['final_analysis']['summary']['total_doors']}개")
print(f"창문: {result['final_analysis']['summary']['total_windows']}개")
print(f"공간: {result['final_analysis']['summary']['total_spaces']}개")
```

#### 배치 처리 (고급)
```python
import asyncio
from src.vllm_analyzer import VLLMAnalyzer

# 여러 이미지 동시 분석
analyzer = VLLMAnalyzer()
analyzer.load_model()

# 비동기 배치 처리
async def batch_analysis():
    results = await analyzer.analyze_batch(
        images=image_list,
        analysis_types=["element_detection"] * len(image_list)
    )
    return results

# 실행
results = asyncio.run(batch_analysis())
```

### 6. 테스트 및 벤치마크

```bash
# 통합 테스트 실행
python test_vllm_integration.py

# 성능 비교 테스트
python src/vllm_analyzer.py  # 단일 테스트
python src/qwen_vlm_analyzer_fixed.py  # 기존 방식 테스트
```

## 📁 프로젝트 구조

```
VLM/
├── src/
│   ├── architectural_vector_analyzer.py     # 벡터 기반 건축 분석기
│   ├── qwen_vlm_analyzer_fixed.py          # VLM 분석기 (vLLM 지원)
│   ├── vllm_analyzer.py                    # 순수 vLLM 분석기
│   ├── vllm_config.py                      # vLLM 설정 및 최적화
│   ├── vlm_pattern_workflow_fixed.py       # LangGraph 통합 워크플로우
│   └── vlm_enhanced_pattern_analyzer.py    # VLM 향상 패턴 분석기
├── configs/                                # vLLM 설정 파일들
│   ├── vllm_auto.json                     # 자동 감지 설정
│   ├── vllm_throughput.json               # 처리량 최적화
│   ├── vllm_latency.json                  # 지연시간 최적화
│   └── vllm_memory.json                   # 메모리 최적화
├── models/                                 # AI 모델들
├── uploads/                               # 테스트 PDF 파일들
├── vector_db/                            # 벡터 데이터베이스
├── test_vllm_integration.py             # vLLM 통합 테스트
└── requirements.txt                      # 의존성 목록
```

## 🔧 주요 구성 요소

### 1. 벡터 분석기 (`architectural_vector_analyzer.py`)
- PyMuPDF를 이용한 벡터 데이터 직접 추출
- 건축 요소별 패턴 매칭 알고리즘
- 기하학적 분석을 통한 구조 요소 식별

### 2. vLLM 분석기 (`vllm_analyzer.py`)
- 순수 vLLM 기반 고성능 멀티모달 추론
- 자동 GPU 리소스 최적화
- 배치 처리 및 비동기 분석 지원
- 메모리 효율적 처리

### 3. 하이브리드 VLM 분석기 (`qwen_vlm_analyzer_fixed.py`)
- vLLM 우선, transformers fallback 지원
- 자동 모델 로딩 및 설정 최적화
- 메모리 관리 및 정리 기능

### 4. 설정 관리자 (`vllm_config.py`)
- 하드웨어별 자동 설정 감지
- 성능 최적화 프로파일 제공
- 런타임 설정 조정 도구

### 5. 통합 워크플로우 (`vlm_pattern_workflow_fixed.py`)
- LangGraph 기반 상태 관리
- vLLM 통합 지원
- 병렬 처리를 통한 성능 최적화
- 오류 처리 및 신뢰도 검증

## ⚡ 성능 최적화 가이드

### GPU 메모리 최적화
```python
# 메모리 제한 환경
analyzer = VLLMAnalyzer(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    gpu_memory_utilization=0.7,  # 낮은 메모리 사용률
    max_model_len=4096           # 짧은 컨텍스트
)

# 고메모리 환경
analyzer = VLLMAnalyzer(
    model_name="Qwen/Qwen2.5-VL-72B-Instruct",
    gpu_memory_utilization=0.95,  # 높은 메모리 사용률
    tensor_parallel_size=4        # 다중 GPU 사용
)
```

### 처리량 vs 지연시간 최적화
```python
# 처리량 우선 (배치 처리)
from src.vllm_config import VLLMOptimizer
config = VLLMOptimizer.optimize_for_throughput(base_config)

# 지연시간 우선 (실시간 처리)
config = VLLMOptimizer.optimize_for_latency(base_config)
```

## 📊 성능 특징

### vLLM vs transformers 비교
| 특징 | vLLM | transformers |
|------|------|--------------|
| **추론 속도** | 5-10배 빠름 | 기준 속도 |
| **메모리 효율성** | PagedAttention | 표준 어텐션 |
| **배치 처리** | 동적 배치 | 정적 배치 |
| **GPU 활용률** | 90%+ | 60-70% |
| **동시 요청** | 수십개 | 수개 |

### 권장 하드웨어 사양
- **최소**: NVIDIA RTX 3080 (10GB VRAM)
- **권장**: NVIDIA RTX 4090 (24GB VRAM)  
- **최적**: NVIDIA A100 (40/80GB VRAM)
- **다중 GPU**: 2-4x RTX 4090 또는 A100

## 🚨 문제 해결

### vLLM 설치 오류
```bash
# CUDA 버전 확인
nvidia-smi

# 올바른 CUDA 버전으로 PyTorch 재설치
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# vLLM 재설치
pip install vllm --no-cache-dir
```

### 메모리 부족 오류
```python
# 설정 조정
analyzer = VLLMAnalyzer(
    gpu_memory_utilization=0.6,  # 메모리 사용률 감소
    max_model_len=2048,          # 컨텍스트 길이 감소
    enable_chunked_prefill=True  # 청크 처리 활성화
)
```

### Fallback 모드 사용
```python
# vLLM 실패 시 자동 fallback
analyzer = QwenVLMAnalyzer(use_vllm=True)  # vLLM 시도 후 transformers로 fallback
```

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
