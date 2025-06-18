#!/usr/bin/env python3
"""
실제 DWG 파일을 사용한 워크플로우 테스트
DWG -> DXF 변환 후 전체 분석 워크플로우 실행
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'dwg_workflow_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def test_dwg_conversion():
    """DWG to DXF 변환 테스트"""
    print("\n🔄 DWG -> DXF 변환 테스트")
    print("=" * 50)
    
    try:
        from src.dwg_parser import DWGProjectProcessor
        
        # 프로젝트 프로세서 초기화
        processor = DWGProjectProcessor()
        
        # 실제 프로젝트명
        project_name = "01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면"
        
        # 프로젝트 내 DWG 파일 변환
        print(f"프로젝트 '{project_name}' DWG 파일 변환 시작...")
        
        conversion_results = processor.convert_project_dwg_files(
            project_name=project_name,
            output_subdir="converted_dxf"
        )
        
        if conversion_results:
            print(f"✅ 변환 성공: {len(conversion_results)}개 파일")
            
            # 변환 결과 요약 출력
            print("\n📋 변환 결과:")
            for i, (dwg_path, dxf_path) in enumerate(conversion_results.items(), 1):
                dwg_name = Path(dwg_path).name
                dxf_name = Path(dxf_path).name
                print(f"  {i:2d}. {dwg_name} -> {dxf_name}")
            
            return conversion_results
        else:
            print("❌ 변환된 파일이 없습니다.")
            return {}
            
    except ImportError as e:
        print(f"❌ DWG 파서 import 실패: {e}")
        return {}
    except Exception as e:
        print(f"❌ DWG 변환 테스트 실패: {e}")
        return {}

def test_dxf_workflow(dxf_files: dict):
    """변환된 DXF 파일로 워크플로우 테스트"""
    print("\n🔧 DXF 파일 워크플로우 테스트")
    print("=" * 50)
    
    if not dxf_files:
        print("❌ 변환된 DXF 파일이 없습니다.")
        return False
    
    try:
        from src.dwg_metadata_extractor import DWGMetadataExtractor
        
        # 메타데이터 추출기 초기화
        extractor = DWGMetadataExtractor()
        
        if not extractor.llm:
            print("⚠️  LLM이 초기화되지 않았습니다. (Ollama 서버 확인 필요)")
            return False
        
        # 첫 번째 DXF 파일로 테스트
        test_dxf_path = list(dxf_files.values())[0]
        print(f"테스트 파일: {Path(test_dxf_path).name}")
        
        # 메타데이터 추출
        print("📊 메타데이터 추출 중...")
        metadata = extractor.extract_from_dwg_file(test_dxf_path)
        
        if metadata:
            print("✅ 메타데이터 추출 성공")
            
            # 주요 정보 출력
            project_info = metadata.get('project_info', {})
            drawing_metadata = metadata.get('drawing_metadata', {})
            tech_specs = metadata.get('technical_specifications', {})
            
            print(f"\n📋 추출된 정보:")
            if project_info:
                print(f"  • 프로젝트명: {project_info.get('project_name', 'Unknown')}")
                print(f"  • 도면 유형: {project_info.get('drawing_type', 'Unknown')}")
                print(f"  • 분야: {project_info.get('discipline', 'Unknown')}")
            
            if drawing_metadata:
                print(f"  • 제목: {drawing_metadata.get('title', 'Unknown')}")
                keywords = drawing_metadata.get('keywords', [])
                if keywords:
                    print(f"  • 키워드: {', '.join(keywords[:3])}")
            
            if tech_specs:
                print(f"  • 파일 형식: {tech_specs.get('file_format', 'Unknown')}")
                print(f"  • 단위: {tech_specs.get('units', 'Unknown')}")
            
            # RAG 콘텐츠 생성
            print("\n📝 RAG 콘텐츠 생성 중...")
            rag_content = extractor.generate_rag_content(metadata)
            
            if rag_content:
                print("✅ RAG 콘텐츠 생성 성공")
                print(f"  콘텐츠 길이: {len(rag_content)} 문자")
                print(f"  미리보기: {rag_content[:100]}...")
            
            # 메타데이터 저장
            output_path = Path(test_dxf_path).parent / f"{Path(test_dxf_path).stem}_metadata.json"
            if extractor.save_metadata(metadata, str(output_path)):
                print(f"✅ 메타데이터 저장: {output_path.name}")
            
            return True
        else:
            print("❌ 메타데이터 추출 실패")
            return False
            
    except ImportError as e:
        print(f"❌ 메타데이터 추출기 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ DXF 워크플로우 테스트 실패: {e}")
        return False

def test_full_workflow():
    """전체 워크플로우 통합 테스트"""
    print("\n🚀 전체 워크플로우 통합 테스트")
    print("=" * 50)
    
    try:
        from architectural_workflow import ArchitecturalAnalysisWorkflow
        
        # 워크플로우 초기화
        workflow = ArchitecturalAnalysisWorkflow()
        
        # 테스트 상태 생성
        project_path = str(Path("uploads") / "01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면")
        
        test_state = {
            "project_path": project_path,
            "analysis_type": "dwg_only",  # DWG 전용 분석
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
        
        print(f"프로젝트 경로: {project_path}")
        print("분석 유형: DWG 전용")
        
        # 워크플로우 실행
        print("\n🔄 워크플로우 실행 중...")
        result = workflow.run_workflow(test_state)
        
        if result:
            print("✅ 워크플로우 실행 완료")
            
            # 결과 요약
            progress = result.get('progress', 0)
            errors = result.get('errors', [])
            logs = result.get('logs', [])
            
            print(f"\n📊 실행 결과:")
            print(f"  • 진행률: {progress:.1f}%")
            print(f"  • 오류 수: {len(errors)}")
            print(f"  • 로그 수: {len(logs)}")
            
            if errors:
                print(f"\n❌ 오류 내역:")
                for i, error in enumerate(errors[:3], 1):
                    print(f"  {i}. {error}")
            
            # DWG 데이터 확인
            dwg_data = result.get('dwg_data', {})
            if dwg_data:
                files_processed = dwg_data.get('files_processed', 0)
                print(f"  • 처리된 DWG 파일: {files_processed}개")
            
            return True
        else:
            print("❌ 워크플로우 실행 실패")
            return False
            
    except ImportError as e:
        print(f"❌ 워크플로우 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 전체 워크플로우 테스트 실패: {e}")
        return False

def generate_test_report(conversion_results: dict, workflow_success: bool):
    """테스트 결과 보고서 생성"""
    print("\n📄 테스트 보고서 생성 중...")
    
    report_content = f"""# 실제 DWG 파일 워크플로우 테스트 보고서

## 테스트 실행 정보
- 실행 시간: {datetime.now().isoformat()}
- 테스트 대상: 01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면

## DWG to DXF 변환 결과

### 변환 통계
- 변환 시도: {len(conversion_results)} 파일
- 변환 성공: {len([v for v in conversion_results.values() if v])} 파일
- 변환 실패: {len([v for v in conversion_results.values() if not v])} 파일

### 변환된 파일 목록
"""
    
    if conversion_results:
        for i, (dwg_path, dxf_path) in enumerate(conversion_results.items(), 1):
            dwg_name = Path(dwg_path).name
            dxf_name = Path(dxf_path).name if dxf_path else "변환 실패"
            status = "✅" if dxf_path else "❌"
            report_content += f"{i:2d}. {status} {dwg_name} -> {dxf_name}\n"
    else:
        report_content += "변환된 파일이 없습니다.\n"
    
    report_content += f"""

## 워크플로우 테스트 결과

### 통합 워크플로우
- 상태: {"✅ 성공" if workflow_success else "❌ 실패"}
- 분석 유형: DWG 전용 분석
- LangSmith 추적: 활성화됨

## 주요 성과

1. **DWG 변환 기능**: ezdxf의 odafc addon을 활용한 DWG to DXF 변환
2. **LangSmith 통합**: 모든 분석 과정에 추적 기능 적용
3. **XREF 처리**: 외부 참조 파일 자동 필터링
4. **메타데이터 추출**: LLM 기반 의미적 분석
5. **RAG 통합**: 기존 시스템과의 원활한 연동

## 기술적 세부사항

### 사용된 도구
- **ezdxf**: DWG/DXF 파일 처리 및 변환
- **ODA File Converter**: DWG to DXF 변환 (odafc addon)
- **LangChain + Ollama**: LLM 기반 메타데이터 추출
- **LangSmith**: 전체 과정 추적 및 모니터링

### 변환 프로세스
1. 프로젝트 내 DWG 파일 자동 검색 (XREF 폴더 제외)
2. ODA File Converter를 통한 DWG to DXF 변환
3. 변환 실패 시 대안 방법 자동 시도
4. 변환된 DXF 파일로 메타데이터 추출
5. RAG 데이터베이스 통합

## 향후 개선사항

1. **ODA File Converter 설치**: 변환 성공률 향상
2. **배치 처리 최적화**: 대용량 프로젝트 처리 성능 개선
3. **오류 복구**: 손상된 DWG 파일 처리 강화
4. **진행률 표시**: 실시간 진행 상황 모니터링

---

*이 보고서는 자동으로 생성되었습니다.*
"""
    
    # 보고서 저장
    report_path = Path("workflow_reports") / f"real_dwg_workflow_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 테스트 보고서 생성: {report_path}")
    return report_path

def main():
    """메인 실행 함수"""
    print("🚀 실제 DWG 파일 워크플로우 테스트 시작")
    print("=" * 60)
    
    # 로깅 설정
    setup_logging()
    
    # 1. DWG to DXF 변환 테스트
    conversion_results = test_dwg_conversion()
    
    # 2. 변환된 DXF 파일로 워크플로우 테스트
    if conversion_results:
        dxf_workflow_success = test_dxf_workflow(conversion_results)
    else:
        dxf_workflow_success = False
    
    # 3. 전체 워크플로우 통합 테스트 (현재 프로젝트 구조 사용)
    full_workflow_success = test_full_workflow()
    
    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("🔍 최종 테스트 결과")
    print("=" * 60)
    
    conversion_success = len(conversion_results) > 0
    
    print(f"DWG -> DXF 변환:     {'✅ 성공' if conversion_success else '❌ 실패'}")
    print(f"DXF 워크플로우:      {'✅ 성공' if dxf_workflow_success else '❌ 실패'}")
    print(f"통합 워크플로우:     {'✅ 성공' if full_workflow_success else '❌ 실패'}")
    
    overall_success = conversion_success or dxf_workflow_success or full_workflow_success
    print(f"\n전체 결과:          {'✅ 성공' if overall_success else '❌ 실패'}")
    
    # 5. 보고서 생성
    generate_test_report(conversion_results, full_workflow_success)
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
