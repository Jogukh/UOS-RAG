#!/usr/bin/env python3
"""
vLLM 기반 VLM 시스템 사용 예제
"""

import sys
import logging
from pathlib import Path

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

def example_vllm_basic():
    """기본 vLLM 사용 예제"""
    print("🚀 vLLM 기본 사용 예제")
    
    try:
        from vllm_analyzer import VLLMAnalyzer
        from PIL import Image, ImageDraw
        
        # vLLM 분석기 초기화
        analyzer = VLLMAnalyzer(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8
        )
        
        # 모델 로드
        if analyzer.load_model():
            print("✅ vLLM 모델 로드 성공")
            
            # 테스트 이미지 생성
            image = Image.new('RGB', (400, 300), 'white')
            draw = ImageDraw.Draw(image)
            draw.rectangle([50, 50, 350, 250], outline='black', width=2)
            draw.text((100, 100), "Test Room", fill='black')
            
            # 분석 실행
            result = analyzer.analyze_image(image, "element_detection")
            print(f"📊 분석 결과: {result['status']}")
            
            # 정리
            analyzer.cleanup()
            
        else:
            print("❌ vLLM 모델 로드 실패")
            
    except ImportError as e:
        print(f"⚠️  vLLM 사용 불가: {e}")
        print("💡 pip install vllm 로 설치하세요")


def example_hybrid_analyzer():
    """하이브리드 분석기 사용 예제 (vLLM + fallback)"""
    print("🔄 하이브리드 분석기 사용 예제")
    
    try:
        from qwen_vlm_analyzer_fixed import QwenVLMAnalyzer
        from PIL import Image, ImageDraw
        
        # vLLM 우선, fallback 지원 분석기
        analyzer = QwenVLMAnalyzer(use_vllm=True)
        
        if analyzer.load_model():
            inference_type = "vLLM" if analyzer.use_vllm else "transformers"
            print(f"✅ 모델 로드 성공 ({inference_type})")
            
            # 테스트 이미지
            image = Image.new('RGB', (600, 400), 'white')
            draw = ImageDraw.Draw(image)
            draw.rectangle([100, 100, 500, 300], outline='black', width=3)
            draw.line([100, 200, 500, 200], fill='black', width=2)
            
            # 분석 실행
            result = analyzer.analyze_image(image, "architectural_basic")
            print(f"📊 분석 완료: {result.get('model_info', {}).get('inference_type', 'unknown')}")
            
            # 메모리 정리
            analyzer.cleanup_memory()
            
        else:
            print("❌ 모델 로드 실패")
            
    except ImportError as e:
        print(f"⚠️  분석기 사용 불가: {e}")


def example_config_optimization():
    """설정 최적화 예제"""
    print("⚙️  설정 최적화 예제")
    
    try:
        from vllm_config import VLLMConfig, VLLMOptimizer
        import json
        
        # 설정 관리자 초기화
        config_manager = VLLMConfig()
        
        # 자동 설정 감지
        auto_config = config_manager.auto_detect_config()
        print("🔍 자동 감지된 설정:")
        print(json.dumps(auto_config, indent=2))
        
        # 최적화 옵션들
        optimizer = VLLMOptimizer()
        
        # 처리량 최적화
        throughput_config = optimizer.optimize_for_throughput(auto_config)
        print("\n⚡ 처리량 최적화 설정:")
        print(f"  - 동시 시퀀스: {throughput_config.get('max_num_seqs', 'N/A')}")
        print(f"  - GPU 메모리: {throughput_config.get('gpu_memory_utilization', 'N/A')}")
        
        # 지연시간 최적화
        latency_config = optimizer.optimize_for_latency(auto_config)
        print("\n🚀 지연시간 최적화 설정:")
        print(f"  - CUDA Graph: {not latency_config.get('enforce_eager', False)}")
        print(f"  - 동시 시퀀스: {latency_config.get('max_num_seqs', 'N/A')}")
        
    except ImportError as e:
        print(f"⚠️  설정 관리자 사용 불가: {e}")


def example_batch_processing():
    """배치 처리 예제"""
    print("📦 배치 처리 예제")
    
    try:
        import asyncio
        from vllm_analyzer import VLLMAnalyzer
        from PIL import Image, ImageDraw
        
        # 여러 테스트 이미지 생성
        images = []
        for i in range(3):
            img = Image.new('RGB', (300, 200), 'white')
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 20, 280, 180], outline='black', width=2)
            draw.text((50, 50), f"Image {i+1}", fill='black')
            images.append(img)
        
        # 배치 분석 함수
        async def run_batch_analysis():
            analyzer = VLLMAnalyzer()
            
            if analyzer.load_model():
                print(f"✅ 모델 로드 완료, {len(images)}개 이미지 배치 처리 시작")
                
                # 비동기 배치 분석
                results = await analyzer.analyze_batch(
                    images=images,
                    analysis_types=["element_detection"] * len(images)
                )
                
                print(f"📊 배치 분석 완료: {len(results)}개 결과")
                for i, result in enumerate(results):
                    status = result.get('status', 'unknown')
                    print(f"  이미지 {i+1}: {status}")
                
                analyzer.cleanup()
                return results
            else:
                print("❌ 모델 로드 실패")
                return []
        
        # 비동기 실행
        results = asyncio.run(run_batch_analysis())
        
    except ImportError as e:
        print(f"⚠️  배치 처리 사용 불가: {e}")
    except Exception as e:
        print(f"❌ 배치 처리 오류: {e}")


def example_workflow_integration():
    """워크플로우 통합 예제"""
    print("🔄 워크플로우 통합 예제")
    
    try:
        from vlm_pattern_workflow_fixed import VLMPatternWorkflow
        from PIL import Image, ImageDraw
        
        # vLLM 사용 워크플로우
        workflow = VLMPatternWorkflow(use_vllm=True)
        
        # 테스트 이미지 생성
        image = Image.new('RGB', (800, 600), 'white')
        draw = ImageDraw.Draw(image)
        
        # 건축 도면 시뮬레이션
        draw.rectangle([100, 100, 700, 500], outline='black', width=3)
        draw.line([100, 300, 700, 300], fill='black', width=2)
        draw.line([400, 100, 400, 500], fill='black', width=2)
        draw.rectangle([200, 295, 220, 305], fill='brown')  # 문
        draw.rectangle([450, 95, 500, 105], fill='blue')   # 창문
        
        print("✅ 워크플로우 초기화 완료")
        print("🏗️  건축 도면 분석 시뮬레이션...")
        
        # VLM 분석기가 있으면 간단한 테스트
        if hasattr(workflow, 'vlm_analyzer') and workflow.vlm_analyzer:
            result = workflow.vlm_analyzer.analyze_image(image, "element_detection")
            inference_type = result.get('model_info', {}).get('inference_type', 'unknown')
            print(f"📊 VLM 분석 완료 ({inference_type}): {result.get('status', 'unknown')}")
        else:
            print("⚠️  VLM 분석기 초기화 실패")
            
    except ImportError as e:
        print(f"⚠️  워크플로우 사용 불가: {e}")


def main():
    """메인 예제 실행 함수"""
    logging.basicConfig(level=logging.WARNING)  # 로그 레벨 조정
    
    print("🎯 vLLM 기반 VLM 시스템 사용 예제들")
    print("=" * 50)
    
    examples = [
        ("기본 vLLM 사용", example_vllm_basic),
        ("하이브리드 분석기", example_hybrid_analyzer),
        ("설정 최적화", example_config_optimization),
        ("배치 처리", example_batch_processing),
        ("워크플로우 통합", example_workflow_integration)
    ]
    
    for name, func in examples:
        print(f"\n{'🔸 ' + name + ' 🔸'}")
        print("-" * 40)
        try:
            func()
        except Exception as e:
            print(f"❌ 예제 실행 오류: {e}")
        print()
    
    print("✅ 모든 예제 실행 완료!")
    print("\n💡 도움말:")
    print("  - vLLM 설치: pip install vllm")
    print("  - 의존성 설치: pip install -r requirements.txt")
    print("  - 테스트 실행: python test_vllm_integration.py")


if __name__ == "__main__":
    main()
