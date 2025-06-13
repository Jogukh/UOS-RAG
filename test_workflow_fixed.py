#!/usr/bin/env python3
"""
수정된 VLM 워크플로우 테스트 (간단 버전)
"""

import sys
import time
import logging
import os
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

def test_simple_workflow():
    """간단한 워크플로우 테스트"""
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Testing FIXED VLM Workflow...")
    print("=" * 50)
    
    # PDF 파일 경로 확인
    pdf_file_path = os.path.join(os.path.dirname(__file__), 'uploads', 'architectural-plan.pdf')
    
    if not os.path.exists(pdf_file_path):
        print(f"❌ PDF file not found: {pdf_file_path}")
        return False
    
    print(f"📄 PDF file found: {pdf_file_path}")
    
    try:
        # 워크플로우 import
        from vlm_pattern_workflow_fixed import VLMPatternWorkflow
        
        print("✅ VLM workflow imported successfully")
        
        # 워크플로우 초기화
        print("\n🔧 Initializing workflow...")
        start_init = time.time()
        
        workflow = VLMPatternWorkflow()
        
        init_time = time.time() - start_init
        print(f"✅ Workflow initialized in {init_time:.2f} seconds")
        
        # GPU 정보 확인
        if hasattr(workflow.vlm_analyzer, 'device'):
            print(f"🎯 VLM Device: {workflow.vlm_analyzer.device}")
        
        # 페이지 분석 실행
        print(f"\n🔍 Starting analysis of page 0...")
        
        analysis_start = time.time()
        
        result = workflow.analyze_page(
            file_path=pdf_file_path,
            page_number=0
        )
        
        analysis_time = time.time() - analysis_start
        print(f"⏱️ Analysis completed in {analysis_time:.2f} seconds")
        
        # 결과 확인
        if result and result.get("status") == "completed":
            print("\n✅ Workflow completed successfully!")
            
            final_analysis = result.get("final_analysis", {})
            summary = final_analysis.get("summary", {})
            
            print(f"📊 Analysis Summary:")
            print(f"  🧱 Walls: {summary.get('total_walls', 0)}")
            print(f"  🚪 Doors: {summary.get('total_doors', 0)}")
            print(f"  🪟 Windows: {summary.get('total_windows', 0)}")
            print(f"  🏠 Spaces: {summary.get('total_spaces', 0)}")
            print(f"  🎯 Confidence: {summary.get('overall_confidence', 0):.2f}")
            
            # 처리 정보
            metadata = final_analysis.get("processing_metadata", {})
            if metadata:
                print(f"\n📈 Processing Info:")
                print(f"  Total steps: {metadata.get('total_steps', 0)}")
                print(f"  Total time: {metadata.get('total_time', 0):.2f}s")
                print(f"  Errors: {metadata.get('errors_count', 0)}")
            
            # 결과 저장
            result_file = "fixed_workflow_result.json"
            try:
                import json
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                print(f"\n💾 Results saved to: {result_file}")
            except Exception as save_error:
                print(f"⚠️ Could not save results: {save_error}")
            
            return True
            
        else:
            print(f"\n❌ Workflow failed:")
            print(f"  Status: {result.get('status', 'unknown') if result else 'no result'}")
            if result and result.get('errors'):
                print(f"  Errors: {result['errors']}")
            return False
    
    except Exception as e:
        print(f"\n💥 Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_simple_workflow()
        
        if success:
            print("\n🎉 VLM workflow test completed successfully!")
            print("✅ The hanging issue appears to be resolved!")
        else:
            print("\n❌ VLM workflow test failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        print("\n🏁 Test execution finished. Exiting...")
        sys.exit(0)
