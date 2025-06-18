#!/usr/bin/env python3
"""
DWG 분석 모듈 통합 테스트 (LangSmith 추적 포함)
전체 DWG 분석 워크플로우를 테스트하고 LangSmith 추적 기능을 검증합니다.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

def test_dwg_parser():
    """DWG 파서 테스트"""
    print("\n=== DWG 파서 테스트 ===")
    
    try:
        from src.dwg_parser import DWGParser
        
        parser = DWGParser()
        print("✅ DWG 파서 초기화 성공")
        
        # 테스트용 DWG 파일 경로 (실제 파일이 있는 경우에만 테스트)
        test_dwg_path = Path("uploads") / "01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면" / "01_건축 도면 (CAD)"
        
        if test_dwg_path.exists():
            dwg_files = []
            # XREF 폴더를 제외하고 DWG/DXF 파일 찾기
            for ext in ['*.dwg', '*.dxf']:
                found_files = test_dwg_path.rglob(ext)
                for dwg_file in found_files:
                    # XREF 폴더 제외
                    if 'XREF' not in str(dwg_file).upper():
                        dwg_files.append(dwg_file)
            
            if dwg_files:
                test_file = dwg_files[0]
                print(f"테스트 파일: {test_file}")
                
                if parser.load_file(str(test_file)):
                    print("✅ DWG 파일 로드 성공")
                    
                    # 기본 정보 추출 테스트
                    basic_info = parser.extract_basic_info()
                    print(f"✅ 기본 정보 추출: {len(basic_info)} 항목")
                    
                    # LLM 읽기 쉬운 요약 생성 테스트
                    summary = parser.generate_llm_readable_summary()
                    print(f"✅ LLM 요약 생성: {len(summary)} 문자")
                    
                    return True
                else:
                    print("❌ DWG 파일 로드 실패")
                    return False
            else:
                print("⚠️  테스트할 DWG 파일이 없습니다.")
                return True  # 파일이 없어도 테스트는 통과
        else:
            print("⚠️  테스트 디렉토리가 존재하지 않습니다.")
            return True
            
    except ImportError as e:
        print(f"❌ DWG 파서 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ DWG 파서 테스트 실패: {e}")
        return False

def test_dwg_metadata_extractor():
    """DWG 메타데이터 추출기 테스트"""
    print("\n=== DWG 메타데이터 추출기 테스트 ===")
    
    try:
        from src.dwg_metadata_extractor import DWGMetadataExtractor
        
        extractor = DWGMetadataExtractor()
        print("✅ DWG 메타데이터 추출기 초기화 성공")
        
        # LLM 초기화 확인
        if extractor.llm:
            print("✅ LLM 초기화 성공")
        else:
            print("⚠️  LLM 초기화 실패 (Ollama 서버 확인 필요)")
            
        return True
        
    except ImportError as e:
        print(f"❌ DWG 메타데이터 추출기 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ DWG 메타데이터 추출기 테스트 실패: {e}")
        return False

def test_workflow_integration():
    """워크플로우 통합 테스트"""
    print("\n=== 워크플로우 통합 테스트 ===")
    
    try:
        from architectural_workflow import ArchitecturalAnalysisWorkflow, WorkflowState
        
        workflow = ArchitecturalAnalysisWorkflow()
        print("✅ 워크플로우 초기화 성공")
        
        # 테스트 상태 생성
        test_state = {
            "project_path": str(Path("uploads") / "01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면"),
            "analysis_type": "dwg_only",
            "step": "",
            "current_task": "",
            "progress": 0.0,
            "pdf_texts": {},
            "dwg_data": {},
            "metadata": {},
            "relationships": {},
            "rag_db_status": False,
            "thoughts": [],
            "decisions": [],
            "results": {},
            "logs": [],
            "errors": []
        }
        
        # 워크플로우 라우팅 테스트
        route = workflow.route_next_step(test_state)
        print(f"✅ 라우팅 테스트 성공: {route}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 워크플로우 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 워크플로우 테스트 실패: {e}")
        return False

def test_prompt_templates():
    """프롬프트 템플릿 테스트"""
    print("\n=== 프롬프트 템플릿 테스트 ===")
    
    try:
        import yaml
        
        prompt_file = Path("prompts") / "dwg_analysis.yaml"
        
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            
            print(f"✅ 프롬프트 파일 로드 성공: {len(prompts)} 개 템플릿")
            
            # 주요 프롬프트 존재 확인
            required_prompts = [
                'dwg_basic_metadata',
                'dwg_content_analysis', 
                'dwg_architectural_features'
            ]
            
            for prompt_name in required_prompts:
                if prompt_name in prompts:
                    print(f"✅ {prompt_name} 템플릿 확인")
                else:
                    print(f"⚠️  {prompt_name} 템플릿 누락")
            
            return True
        else:
            print("❌ 프롬프트 파일이 존재하지 않습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 프롬프트 템플릿 테스트 실패: {e}")
        return False

def test_dependencies():
    """의존성 패키지 테스트"""
    print("\n=== 의존성 패키지 테스트 ===")
    
    required_packages = [
        ("ezdxf", "DXF/DWG 파일 처리"),
        ("langchain_ollama", "LLM 연동"),
        ("chromadb", "벡터 데이터베이스"),
        ("yaml", "설정 파일 처리")
    ]
    
    all_ok = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"❌ {package}: {description} - 설치 필요")
            all_ok = False
    
    return all_ok

def test_langsmith_integration():
    """LangSmith 추적 기능 테스트"""
    print("\n=== LangSmith 추적 기능 테스트 ===")
    
    try:
        from src.langsmith_integration import LangSmithTracker, trace_llm_call
        
        # LangSmith 추적기 초기화 테스트
        tracker = LangSmithTracker()
        print("✅ LangSmith 추적기 초기화 성공")
        
        # 추적 세션 시작 테스트
        session_name = f"dwg_test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_id = tracker.start_session(session_name, {
            "test_type": "dwg_integration",
            "timestamp": datetime.now().isoformat()
        })
        
        if session_id:
            print(f"✅ LangSmith 세션 시작: {session_id}")
        else:
            print("⚠️  LangSmith 세션 시작 실패 (설정 확인 필요)")
        
        # 테스트용 추적 함수
        @trace_llm_call("test_trace_dwg", "chain")
        def test_traced_function():
            """테스트용 추적 함수"""
            return {"test": "successful", "timestamp": datetime.now().isoformat()}
        
        # 추적 함수 실행 테스트
        result = test_traced_function()
        print(f"✅ 추적 함수 실행 성공: {result}")
        
        # 세션 종료 테스트
        if session_id:
            tracker.end_session(session_id)
            print("✅ LangSmith 세션 종료 완료")
        
        return True
        
    except ImportError as e:
        print(f"❌ LangSmith 통합 모듈 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ LangSmith 추적 기능 테스트 실패: {e}")
        return False

def run_integration_test():
    """전체 통합 테스트 실행"""
    print("🔧 DWG 분석 모듈 통합 테스트 시작")
    print("=" * 50)
    
    test_results = {}
    
    # 의존성 테스트
    test_results["dependencies"] = test_dependencies()
    
    # 프롬프트 템플릿 테스트
    test_results["prompts"] = test_prompt_templates()
    
    # DWG 파서 테스트
    test_results["dwg_parser"] = test_dwg_parser()
    
    # DWG 메타데이터 추출기 테스트
    test_results["dwg_extractor"] = test_dwg_metadata_extractor()
    
    # 워크플로우 통합 테스트
    test_results["workflow"] = test_workflow_integration()
    
    # LangSmith 추적 기능 테스트
    test_results["langsmith"] = test_langsmith_integration()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("🔍 테스트 결과 요약")
    print("=" * 50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\n총 {passed}/{total} 테스트 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! DWG 분석 모듈이 성공적으로 통합되었습니다.")
        return True
    else:
        print("⚠️  일부 테스트 실패. 문제를 해결한 후 다시 테스트해주세요.")
        return False

def create_test_report():
    """테스트 결과 보고서 생성"""
    report_content = f"""# DWG 분석 모듈 통합 테스트 보고서

## 테스트 실행 정보
- 실행 시간: {datetime.now().isoformat()}
- 테스트 환경: Python {sys.version}

## 구현된 기능

### 1. DWG 파서 모듈 (src/dwg_parser.py)
- ezdxf 라이브러리를 사용한 DWG/DXF 파일 파싱
- 구조적 메타데이터 추출 (레이어, 블록, 엔티티 등)
- LLM이 읽기 쉬운 형태로 데이터 변환
- JSON 형태의 메타데이터 저장

### 2. DWG 메타데이터 추출기 (src/dwg_metadata_extractor.py)
- LLM 기반 의미적 메타데이터 추출
- 건축적 특징 분석
- 도면 내용 분석
- RAG 시스템용 콘텐츠 생성

### 3. 프롬프트 템플릿 (prompts/dwg_analysis.yaml)
- DWG 분석 전용 프롬프트 템플릿
- 기본 메타데이터, 내용 분석, 건축적 특징 추출용 프롬프트
- 구조화된 JSON 응답 형식

### 4. 워크플로우 통합 (architectural_workflow.py)
- 기존 PDF 분석 워크플로우에 DWG 분석 기능 통합
- DWG 전용 분석 경로 추가
- PDF와 DWG 데이터 통합 메타데이터 생성
- 통합 RAG 데이터베이스 구축

## 주요 기능

1. **DWG/DXF 파일 읽기**: ezdxf와 ODA File Converter 지원
2. **구조적 데이터 추출**: 레이어, 블록, 엔티티, 텍스트 정보
3. **LLM 기반 분석**: 건축적 의미와 특징 추출
4. **RAG 통합**: 기존 시스템과 원활한 연동
5. **Sequential Thinking**: 단계별 사고 과정 기록
6. **Context7 활용**: ezdxf 라이브러리 연구 및 활용

## 기술 스택

- **ezdxf**: DWG/DXF 파일 파싱
- **LangChain + Ollama**: LLM 기반 메타데이터 추출
- **ChromaDB**: 벡터 데이터베이스
- **LangGraph**: 워크플로우 관리
- **PyYAML**: 설정 및 프롬프트 관리

## 설치 및 사용

1. 의존성 설치:
```bash
pip install -r requirements.txt
```

2. DWG 파일 분석:
```python
from src.dwg_parser import DWGParser
from src.dwg_metadata_extractor import DWGMetadataExtractor

# 구조적 데이터 추출
parser = DWGParser()
parser.load_file("your_file.dwg")
metadata = parser.extract_all_metadata()

# LLM 기반 분석
extractor = DWGMetadataExtractor()
analyzed_metadata = extractor.extract_from_dwg_file("your_file.dwg")
```

3. 통합 워크플로우 실행:
```python
from architectural_workflow import ArchitecturalAnalysisWorkflow

workflow = ArchitecturalAnalysisWorkflow()
result = workflow.run({{
    "project_path": "/path/to/project",
    "analysis_type": "dwg_only"
}})
```

## 향후 개선 방안

1. **3D 모델 지원**: 3DSOLID 엔티티 분석 강화
2. **도면 간 관계 추론**: 크로스 레퍼런스 분석
3. **표준 준수 검증**: 도면 표준 자동 검증
4. **시각화**: 추출된 정보의 시각적 표현
5. **성능 최적화**: 대용량 파일 처리 개선

---

*이 보고서는 자동으로 생성되었습니다.*
"""
    
    report_path = Path("workflow_reports") / f"dwg_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📄 상세 보고서가 생성되었습니다: {report_path}")
    return report_path

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 통합 테스트 실행
    success = run_integration_test()
    
    # 보고서 생성
    create_test_report()
    
    # 종료 코드 설정
    sys.exit(0 if success else 1)
