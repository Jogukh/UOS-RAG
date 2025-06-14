#!/usr/bin/env python3
"""
간단한 Qwen2.5-VL 테스트
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

def create_simple_test_image():
    """간단한 테스트 이미지 생성"""
    img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # 간단한 평면도 그리기
    draw.rectangle([50, 50, 350, 80], outline='black', width=2)  # 상단 벽
    draw.rectangle([50, 220, 350, 250], outline='black', width=2)  # 하단 벽
    draw.rectangle([50, 50, 80, 250], outline='black', width=2)  # 좌측 벽
    draw.rectangle([320, 50, 350, 250], outline='black', width=2)  # 우측 벽
    
    # 문
    draw.rectangle([180, 220, 220, 250], fill='white', outline='red', width=2)
    
    # 창문
    draw.rectangle([150, 50, 200, 80], fill='lightblue', outline='blue', width=2)
    
    return img

def test_env_config():
    """환경 설정 테스트"""
    try:
        from env_config import get_env_config
        config = get_env_config()
        print("✅ Environment config loaded successfully")
        print(f"   Model: {config.get('model_name')}")
        print(f"   Device: {config.get('device')}")
        return True
    except Exception as e:
        print(f"❌ Environment config failed: {e}")
        return False

def test_qwen_analyzer():
    """QwenVLMAnalyzer 테스트"""
    try:
        from qwen_vlm_analyzer_fixed import QwenVLMAnalyzer
        
        print("🔍 Testing QwenVLMAnalyzer...")
        analyzer = QwenVLMAnalyzer(use_vllm=False)  # transformers 사용
        
        # 모델 로드 시도
        print("   Loading model...")
        if analyzer.load_model():
            print("   ✅ Model loaded successfully")
            
            # 테스트 이미지 생성
            test_image = create_simple_test_image()
            test_image.save("test_image.png")
            print("   📸 Test image created: test_image.png")
            
            # 분석 시도
            print("   🧠 Analyzing image...")
            result = analyzer.analyze_image(test_image, "element_detection")
            
            if result:
                print("   ✅ Analysis completed successfully")
                print(f"   📊 Result: {result}")
                return True
            else:
                print("   ❌ Analysis failed")
                return False
        else:
            print("   ❌ Failed to load model")
            return False
            
    except Exception as e:
        print(f"❌ QwenVLMAnalyzer test failed: {e}")
        return False

def main():
    print("🧪 Simple Qwen2.5-VL Test")
    print("=" * 40)
    
    # 환경 설정 테스트
    env_ok = test_env_config()
    
    # 분석기 테스트
    if env_ok:
        analyzer_ok = test_qwen_analyzer()
        
        if analyzer_ok:
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️  Some tests failed")
    else:
        print("\n❌ Environment configuration failed")

if __name__ == "__main__":
    main()
