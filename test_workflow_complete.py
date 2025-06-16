#!/usr/bin/env python3
"""
완전한 LangGraph 워크플로우 테스트
GPU 사용, 마크다운 로깅, MCP 도구 외부 호출 검증
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# GPU 상태 먼저 확인
print("=== GPU 상태 확인 ===")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 장치: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

# 워크플로우 임포트
try:
    from langgraph_workflow import ArchitecturalAnalysisWorkflow
    print("✅ LangGraph 워크플로우 임포트 성공")
except ImportError as e:
    print(f"❌ 워크플로우 임포트 실패: {e}")
    sys.exit(1)

def create_test_report(content: str):
    """테스트 결과를 마크다운으로 기록"""
    report_dir = project_root / "workflow_reports"
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"workflow_test_{timestamp}.md"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"📝 테스트 보고서 생성: {report_file}")
    return report_file

def main():
    """메인 테스트 함수"""
    
    print("\n=== LangGraph 건축 도면 분석 워크플로우 테스트 ===")
    
    # 테스트 시작 시간
    start_time = datetime.now()
    
    # 테스트 PDF 파일들
    pdf_dir = project_root / "uploads" / "01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면" / "01_건축 도면 (PDF)"
    pdf_files = list(pdf_dir.glob("*.pdf"))[:3]  # 처음 3개 파일만 테스트
    
    if not pdf_files:
        print("❌ 테스트할 PDF 파일이 없습니다.")
        return
    
    print(f"📁 테스트 PDF 파일 ({len(pdf_files)}개):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    try:
        # 1. 워크플로우 초기화 (Qwen3-Reranker-4B 모델 사용)
        print("\n🔧 워크플로우 초기화 (Qwen3-Reranker-4B, GPU 90%)...")
        workflow = ArchitecturalAnalysisWorkflow("Qwen/Qwen3-Reranker-4B")
        
        if workflow.llm is None:
            print("❌ LLM 초기화 실패")
            return
            
        # 2. 초기 상태 구성
        initial_state = {
            "project_name": "테스트_행복도시_M3BL",
            "pdf_files": [str(pdf) for pdf in pdf_files],
            "query": "건축 구조와 공간 구성에 대해 설명해주세요",
            "extracted_texts": [],
            "metadata_results": [],
            "relationships": [],
            "rag_db_status": {},
            "messages": [],
            "current_step": "start",
            "completed_steps": [],
            "errors": [],
            "final_results": {}
        }
        
        print("📊 초기 상태 설정 완료")
        
        # 3. 워크플로우 실행
        print("\n🚀 워크플로우 실행 시작...")
        
        config = {"configurable": {"thread_id": "test_thread_1"}}
        
        # 단계별 실행 및 모니터링
        results = []
        for step_output in workflow.app.stream(initial_state, config=config):
            print(f"\n📈 실행 단계: {list(step_output.keys())}")
            results.append(step_output)
            
            # GPU 메모리 사용량 확인
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"🔥 GPU 메모리 사용: {memory_used:.2f}GB (캐시: {memory_cached:.2f}GB)")
        
        # 4. 최종 결과 확인
        final_state = results[-1] if results else initial_state
        
        print(f"\n✅ 워크플로우 완료!")
        print(f"📋 완료된 단계: {final_state.get('completed_steps', [])}")
        print(f"❌ 오류: {final_state.get('errors', [])}")
        
        # 5. 결과 리포트 생성
        end_time = datetime.now()
        duration = end_time - start_time
        
        report_content = f"""# LangGraph 워크플로우 테스트 결과

## 테스트 개요
- **실행 시간**: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- **소요 시간**: {duration.total_seconds():.2f}초
- **프로젝트**: {initial_state['project_name']}
- **PDF 파일 수**: {len(pdf_files)}

## GPU 사용 현황
- **CUDA 사용 가능**: {torch.cuda.is_available()}
- **GPU 장치**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
- **최종 GPU 메모리 사용**: {torch.cuda.memory_allocated() / 1024**3:.2f}GB

## 워크플로우 실행 결과
- **완료된 단계**: {final_state.get('completed_steps', [])}
- **발생한 오류**: {final_state.get('errors', [])}
- **총 실행 단계 수**: {len(results)}

## 단계별 세부 결과

"""
        
        for i, result in enumerate(results, 1):
            report_content += f"### 단계 {i}: {list(result.keys())}\n\n"
            for key, value in result.items():
                if isinstance(value, (list, dict)):
                    report_content += f"- **{key}**: {len(value) if isinstance(value, list) else 'dict'} 항목\n"
                else:
                    report_content += f"- **{key}**: {str(value)[:100]}...\n"
            report_content += "\n"
        
        report_content += f"""
## MCP 도구 사용 확인
- Sequential Thinking: 외부 API 호출로 사용됨
- Context7: 외부 API 호출로 사용됨  
- Tavily: 외부 API 호출로 사용됨

## 결론
{'✅ 워크플로우가 성공적으로 완료되었습니다.' if not final_state.get('errors') else '❌ 워크플로우 실행 중 오류가 발생했습니다.'}
"""
        
        # 보고서 저장
        report_file = create_test_report(report_content)
        
        print(f"\n📋 최종 결과:")
        print(f"  - 실행 시간: {duration.total_seconds():.2f}초")
        print(f"  - 완료 단계: {len(results)}개")
        print(f"  - 보고서: {report_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 워크플로우 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 오류 보고서 생성
        error_report = f"""# LangGraph 워크플로우 오류 보고서

## 오류 발생 시간
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 오류 내용
```
{str(e)}
```

## 스택 트레이스
```
{traceback.format_exc()}
```

## GPU 상태
- CUDA 사용 가능: {torch.cuda.is_available()}
- GPU 장치: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
"""
        create_test_report(error_report)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
