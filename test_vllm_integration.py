#!/usr/bin/env python3
"""
vLLM 기반 VLM 시스템 통합 테스트
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import time

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent))

try:
    from vllm_analyzer import VLLMAnalyzer
    from vllm_config import VLLMConfig, VLLMOptimizer
    from qwen_vlm_analyzer_fixed import QwenVLMAnalyzer
    from vlm_pattern_workflow_fixed import VLMPatternWorkflow
    HAS_ANALYZERS = True
except ImportError as e:
    HAS_ANALYZERS = False
    print(f"Warning: Analyzers not available: {e}")

try:
    from PIL import Image, ImageDraw
    import torch
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Warning: Dependencies not available: {e}")

logger = logging.getLogger(__name__)


class VLLMTestSuite:
    """vLLM 기반 VLM 시스템 테스트 스위트"""
    
    def __init__(self):
        self.config_manager = VLLMConfig()
        self.test_results = []
        
    def create_test_images(self) -> List[Image.Image]:
        """테스트용 건축 도면 이미지 생성"""
        images = []
        
        # 기본 평면도
        img1 = Image.new('RGB', (800, 600), 'white')
        draw1 = ImageDraw.Draw(img1)
        
        # 외벽
        draw1.rectangle([50, 50, 750, 550], outline='black', width=3)
        # 내벽
        draw1.line([50, 300, 750, 300], fill='black', width=2)
        draw1.line([400, 50, 400, 550], fill='black', width=2)
        # 문
        draw1.rectangle([200, 295, 220, 305], fill='brown')
        draw1.rectangle([580, 295, 600, 305], fill='brown')
        # 창문
        draw1.rectangle([100, 45, 150, 55], fill='blue')
        draw1.rectangle([450, 45, 500, 55], fill='blue')
        # 텍스트
        draw1.text((100, 100), "거실", fill='black')
        draw1.text((500, 100), "주방", fill='black')
        draw1.text((100, 400), "침실1", fill='black')
        draw1.text((500, 400), "침실2", fill='black')
        
        images.append(img1)
        
        # 복잡한 평면도
        img2 = Image.new('RGB', (1000, 800), 'white')
        draw2 = ImageDraw.Draw(img2)
        
        # 복잡한 구조
        draw2.rectangle([50, 50, 950, 750], outline='black', width=3)
        draw2.line([50, 200, 950, 200], fill='black', width=2)
        draw2.line([50, 400, 950, 400], fill='black', width=2)
        draw2.line([50, 600, 950, 600], fill='black', width=2)
        draw2.line([300, 50, 300, 750], fill='black', width=2)
        draw2.line([600, 50, 600, 750], fill='black', width=2)
        draw2.line([750, 200, 750, 600], fill='black', width=2)
        
        # 문과 창문
        for i in range(5):
            x = 100 + i * 150
            draw2.rectangle([x, 195, x+20, 205], fill='brown')
            draw2.rectangle([x, 45, x+40, 55], fill='blue')
        
        images.append(img2)
        
        return images
    
    def test_vllm_analyzer(self) -> Dict[str, Any]:
        """vLLM 분석기 테스트"""
        test_result = {
            "test_name": "vLLM Analyzer Test",
            "status": "failed",
            "results": [],
            "performance": {},
            "errors": []
        }
        
        try:
            if not HAS_DEPS:
                test_result["errors"].append("Dependencies not available")
                return test_result
            
            # 설정 자동 감지
            config = self.config_manager.auto_detect_config()
            
            # vLLM 분석기 초기화
            analyzer = VLLMAnalyzer(
                model_name=config.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.8)
            )
            
            # 모델 로드 테스트
            start_time = time.time()
            if analyzer.load_model():
                load_time = time.time() - start_time
                test_result["performance"]["model_load_time"] = load_time
                
                # 테스트 이미지 생성
                test_images = self.create_test_images()
                
                # 단일 이미지 분석 테스트
                for i, image in enumerate(test_images):
                    analysis_start = time.time()
                    result = analyzer.analyze_image(image, "element_detection")
                    analysis_time = time.time() - analysis_start
                    
                    test_result["results"].append({
                        "image_id": i,
                        "analysis_time": analysis_time,
                        "status": result.get("status", "unknown"),
                        "elements_detected": len(result.get("parsed_result", {}).get("walls", [])) if isinstance(result.get("parsed_result"), dict) else 0
                    })
                
                # 성능 통계
                analysis_times = [r["analysis_time"] for r in test_result["results"]]
                test_result["performance"].update({
                    "avg_analysis_time": sum(analysis_times) / len(analysis_times),
                    "min_analysis_time": min(analysis_times),
                    "max_analysis_time": max(analysis_times),
                    "total_images": len(test_images)
                })
                
                test_result["status"] = "success"
                
                # 정리
                analyzer.cleanup()
                
            else:
                test_result["errors"].append("Failed to load vLLM model")
                
        except Exception as e:
            test_result["errors"].append(str(e))
            logger.error(f"vLLM analyzer test failed: {e}")
        
        return test_result
    
    def test_fallback_analyzer(self) -> Dict[str, Any]:
        """Fallback 분석기 테스트 (transformers)"""
        test_result = {
            "test_name": "Fallback Analyzer Test",
            "status": "failed",
            "results": [],
            "performance": {},
            "errors": []
        }
        
        try:
            # Fallback 분석기 (transformers 사용)
            analyzer = QwenVLMAnalyzer(use_vllm=False)
            
            start_time = time.time()
            if analyzer.load_model():
                load_time = time.time() - start_time
                test_result["performance"]["model_load_time"] = load_time
                
                # 간단한 테스트 이미지
                test_image = self.create_test_images()[0]
                
                analysis_start = time.time()
                result = analyzer.analyze_image(test_image, "element_detection")
                analysis_time = time.time() - analysis_start
                
                test_result["results"].append({
                    "analysis_time": analysis_time,
                    "status": result.get("status", "unknown"),
                    "inference_type": result.get("model_info", {}).get("inference_type", "unknown")
                })
                
                test_result["performance"]["analysis_time"] = analysis_time
                test_result["status"] = "success"
                
                # 정리
                analyzer.cleanup_memory()
                
            else:
                test_result["errors"].append("Failed to load fallback model")
                
        except Exception as e:
            test_result["errors"].append(str(e))
            logger.error(f"Fallback analyzer test failed: {e}")
        
        return test_result
    
    def test_workflow_integration(self) -> Dict[str, Any]:
        """워크플로우 통합 테스트"""
        test_result = {
            "test_name": "Workflow Integration Test",
            "status": "failed",
            "results": [],
            "errors": []
        }
        
        try:
            # vLLM 사용 워크플로우
            workflow = VLMPatternWorkflow(use_vllm=True)
            
            # 간단한 테스트 이미지
            test_image = self.create_test_images()[0]
            
            # 워크플로우 실행 (간단한 분석만)
            if hasattr(workflow, 'vlm_analyzer') and workflow.vlm_analyzer:
                result = workflow.vlm_analyzer.analyze_image(test_image, "element_detection")
                
                test_result["results"].append({
                    "workflow_type": "vLLM-based",
                    "status": result.get("status", "unknown"),
                    "inference_type": result.get("model_info", {}).get("inference_type", "unknown")
                })
                
                test_result["status"] = "success"
            else:
                test_result["errors"].append("Workflow analyzer not initialized")
                
        except Exception as e:
            test_result["errors"].append(str(e))
            logger.error(f"Workflow integration test failed: {e}")
        
        return test_result
    
    def test_performance_comparison(self) -> Dict[str, Any]:
        """성능 비교 테스트"""
        test_result = {
            "test_name": "Performance Comparison",
            "status": "failed",
            "vllm_performance": {},
            "transformers_performance": {},
            "comparison": {},
            "errors": []
        }
        
        try:
            # 테스트 이미지
            test_image = self.create_test_images()[0]
            
            # vLLM 테스트 (사용 가능한 경우)
            try:
                vllm_analyzer = VLLMAnalyzer()
                if vllm_analyzer.load_model():
                    start_time = time.time()
                    vllm_result = vllm_analyzer.analyze_image(test_image, "element_detection")
                    vllm_time = time.time() - start_time
                    
                    test_result["vllm_performance"] = {
                        "analysis_time": vllm_time,
                        "status": vllm_result.get("status", "unknown"),
                        "available": True
                    }
                    vllm_analyzer.cleanup()
                else:
                    test_result["vllm_performance"]["available"] = False
                    
            except Exception as e:
                test_result["vllm_performance"] = {
                    "available": False,
                    "error": str(e)
                }
            
            # Transformers 테스트
            try:
                transformers_analyzer = QwenVLMAnalyzer(use_vllm=False)
                if transformers_analyzer.load_model():
                    start_time = time.time()
                    transformers_result = transformers_analyzer.analyze_image(test_image, "element_detection")
                    transformers_time = time.time() - start_time
                    
                    test_result["transformers_performance"] = {
                        "analysis_time": transformers_time,
                        "status": transformers_result.get("status", "unknown"),
                        "available": True
                    }
                    transformers_analyzer.cleanup_memory()
                else:
                    test_result["transformers_performance"]["available"] = False
                    
            except Exception as e:
                test_result["transformers_performance"] = {
                    "available": False,
                    "error": str(e)
                }
            
            # 성능 비교
            if (test_result["vllm_performance"].get("available") and 
                test_result["transformers_performance"].get("available")):
                
                vllm_time = test_result["vllm_performance"]["analysis_time"]
                transformers_time = test_result["transformers_performance"]["analysis_time"]
                
                test_result["comparison"] = {
                    "speedup": transformers_time / vllm_time if vllm_time > 0 else 0,
                    "vllm_faster": vllm_time < transformers_time,
                    "time_difference": abs(vllm_time - transformers_time)
                }
                
                test_result["status"] = "success"
            else:
                test_result["status"] = "partial"
                
        except Exception as e:
            test_result["errors"].append(str(e))
            logger.error(f"Performance comparison test failed: {e}")
        
        return test_result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 테스트 실행"""
        print("🧪 Starting vLLM VLM System Test Suite...")
        
        all_results = {
            "test_suite": "vLLM VLM System",
            "timestamp": time.time(),
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "partial_tests": 0
            }
        }
        
        # 테스트 목록
        tests = [
            ("vLLM Analyzer", self.test_vllm_analyzer),
            ("Fallback Analyzer", self.test_fallback_analyzer),
            ("Workflow Integration", self.test_workflow_integration),
            ("Performance Comparison", self.test_performance_comparison)
        ]
        
        # 각 테스트 실행
        for test_name, test_func in tests:
            print(f"🔍 Running {test_name}...")
            try:
                result = test_func()
                all_results["tests"].append(result)
                
                # 결과 카운트
                if result["status"] == "success":
                    all_results["summary"]["passed_tests"] += 1
                    print(f"✅ {test_name}: PASSED")
                elif result["status"] == "partial":
                    all_results["summary"]["partial_tests"] += 1
                    print(f"⚠️  {test_name}: PARTIAL")
                else:
                    all_results["summary"]["failed_tests"] += 1
                    print(f"❌ {test_name}: FAILED")
                    if result.get("errors"):
                        for error in result["errors"]:
                            print(f"   Error: {error}")
                            
            except Exception as e:
                print(f"💥 {test_name}: CRASHED - {e}")
                all_results["tests"].append({
                    "test_name": test_name,
                    "status": "crashed",
                    "error": str(e)
                })
                all_results["summary"]["failed_tests"] += 1
        
        all_results["summary"]["total_tests"] = len(tests)
        
        return all_results


def main():
    """메인 테스트 함수"""
    logging.basicConfig(level=logging.INFO)
    
    # 의존성 확인
    if not HAS_DEPS:
        print("❌ Required dependencies not available")
        return
    
    # 테스트 스위트 실행
    test_suite = VLLMTestSuite()
    results = test_suite.run_all_tests()
    
    # 결과 출력
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    summary = results["summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed_tests']}")
    print(f"⚠️  Partial: {summary['partial_tests']}")
    print(f"❌ Failed: {summary['failed_tests']}")
    
    # 상세 결과 저장
    results_file = "test_results_vllm.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # 권장사항 출력
    print("\n🔧 RECOMMENDATIONS:")
    
    vllm_available = any(
        test.get("test_name") == "vLLM Analyzer Test" and test.get("status") == "success"
        for test in results["tests"]
    )
    
    if vllm_available:
        print("✅ vLLM is working properly - Use it for best performance")
    else:
        print("⚠️  vLLM not available - Using transformers fallback")
        print("   💡 To enable vLLM: pip install vllm")
    
    # 성능 비교 결과
    perf_test = next(
        (test for test in results["tests"] if test.get("test_name") == "Performance Comparison"),
        None
    )
    
    if perf_test and perf_test.get("comparison"):
        speedup = perf_test["comparison"].get("speedup", 0)
        if speedup > 1:
            print(f"🚀 vLLM is {speedup:.2f}x faster than transformers")
        else:
            print("📊 Performance comparison inconclusive")


if __name__ == "__main__":
    main()
