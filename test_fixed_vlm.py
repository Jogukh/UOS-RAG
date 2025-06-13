#!/usr/bin/env python3
"""
수정된 VLM 분석기 테스트
"""

import sys
import time
import logging
from PIL import Image, ImageDraw
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

from qwen_vlm_analyzer_fixed import QwenVLMAnalyzer

def test_fixed_vlm_analyzer():
    """수정된 VLM 분석기 테스트"""
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Testing FIXED VLM analyzer...")
    print("=" * 50)
    
    # 테스트 이미지 생성
    test_image = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([50, 50, 350, 250], outline='black', width=3)
    draw.line([50, 150, 350, 150], fill='black', width=2)
    draw.line([200, 50, 200, 250], fill='black', width=2)
    draw.text((100, 180), "Test Floor Plan", fill='black')
    
    # VLM 분석기 초기화
    print("\n🔧 Initializing VLM analyzer...")
    analyzer = QwenVLMAnalyzer()
    
    # 메모리 사용량 확인 (로드 전)
    print("\n📊 Memory usage (before loading):")
    memory_before = analyzer.get_memory_usage()
    for key, value in memory_before.items():
        if 'gpu' in key:
            print(f"  {key}: {value:.2f} GB")
    
    # 모델 로드
    print("\n⚡ Loading VLM model...")
    start_time = time.time()
    
    try:
        if analyzer.load_model():
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            print(f"🎯 Device: {analyzer.device}")
            
            # 메모리 사용량 확인 (로드 후)
            print("\n📊 Memory usage (after loading):")
            memory_after = analyzer.get_memory_usage()
            for key, value in memory_after.items():
                if 'gpu' in key:
                    print(f"  {key}: {value:.2f} GB")
            
            # VLM 분석 실행
            print("\n🔍 Running VLM analysis...")
            analysis_start = time.time()
            
            result = analyzer.analyze_image(
                test_image, 
                prompt_type='element_detection'
            )
            
            analysis_time = time.time() - analysis_start
            print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
            
            # 결과 확인
            if result.get("status") == "success":
                print("\n📋 Analysis Result:")
                print(f"  Status: {result['status']}")
                print(f"  Prompt Type: {result['prompt_type']}")
                print(f"  Raw Response Length: {len(result['raw_response'])} characters")
                
                # 파싱된 결과 확인
                parsed = result.get("parsed_result", {})
                if parsed:
                    print(f"  Parsed Elements: {len(parsed.get('detected_elements', []))}")
                    print(f"  Key Points: {len(parsed.get('key_points', []))}")
                    
                    # 첫 번째 요소 출력 (있으면)
                    elements = parsed.get('detected_elements', [])
                    if elements:
                        print(f"  First Element: {elements[0].get('element', 'N/A')}")
                
                print("\n✅ VLM analysis completed without hanging!")
                
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
            # 메모리 정리
            print("\n🧹 Cleaning up memory...")
            analyzer.cleanup_memory()
            
            # 최종 메모리 사용량
            print("\n📊 Memory usage (after cleanup):")
            memory_final = analyzer.get_memory_usage()
            for key, value in memory_final.items():
                if 'gpu' in key:
                    print(f"  {key}: {value:.2f} GB")
            
            print("\n✅ Test completed successfully!")
            return True
            
        else:
            print("❌ Failed to load VLM model")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n🏁 Test execution completed")

if __name__ == "__main__":
    try:
        success = test_fixed_vlm_analyzer()
        if success:
            print("\n🎉 All tests passed! VLM analyzer is working correctly.")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🏁 Program exiting normally...")
    sys.exit(0)
