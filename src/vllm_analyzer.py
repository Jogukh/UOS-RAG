#!/usr/bin/env python3
"""
vLLM 기반 최적화된 VLM 분석기
고성능 멀티모달 추론을 위한 vLLM 전용 구현
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import base64
from io import BytesIO

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    # Note: vLLM의 multimodal_utils는 버전에 따라 변경될 수 있음
    HAS_VLLM = True
except ImportError as e:
    HAS_VLLM = False
    print(f"Warning: vLLM not available: {e}")

try:
    from PIL import Image
    import torch
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Warning: Dependencies not available: {e}")

logger = logging.getLogger(__name__)


class VLLMAnalyzer:
    """vLLM 기반 고성능 VLM 분석기"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.8,
                 max_model_len: int = 8192,
                 dtype: str = "bfloat16"):
        """
        vLLM 분석기 초기화
        
        Args:
            model_name: 사용할 모델 이름 또는 경로
            tensor_parallel_size: 텐서 병렬화 크기 (GPU 개수)
            gpu_memory_utilization: GPU 메모리 사용률 (0.0-1.0)
            max_model_len: 최대 시퀀스 길이
            dtype: 데이터 타입 (bfloat16, float16, float32)
        """
        if not HAS_VLLM or not HAS_DEPS:
            raise ImportError("vLLM and dependencies are required")
        
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # vLLM 설정 (Context7 권장사항 적용)
        self.vllm_config = {
            "model": model_name,
            "trust_remote_code": True,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "enforce_eager": False,  # CUDA graph 사용
            "enable_chunked_prefill": True,  # 청크 단위 처리
            "max_num_seqs": 8,  # 동시 처리 시퀀스 수
            "enable_prefix_caching": True,  # 프리픽스 캐싱
            # Context7 문서 기반 추가 최적화
            "block_size": 16,  # KV cache 블록 크기
            "swap_space": 4,   # CPU 메모리 스왑 공간 (GB)
        }
        
        # 샘플링 파라미터 (vLLM 0.9+ 호환)
        self.default_sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            stop_token_ids=None,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            # 반복 방지 (vLLM 0.9+ 호환 파라미터)
            repetition_penalty=1.05,
            # length_penalty는 vLLM 0.9+에서 제거됨
        )
        
        # 건축 도면 분석용 프롬프트
        self.architectural_prompts = {
            "element_detection": """
이 건축 도면에서 다음 요소들을 정확히 탐지하고 JSON으로 응답하세요:

{
  "walls": [
    {"coords": [x1,y1,x2,y2], "thickness": 두께, "type": "외벽|내벽", "confidence": 0.0-1.0}
  ],
  "doors": [
    {"coords": [x,y,width,height], "type": "일반문|미닫이문", "opening_direction": "좌|우", "confidence": 0.0-1.0}
  ],
  "windows": [
    {"coords": [x,y,width,height], "type": "일반창|발코니문", "confidence": 0.0-1.0}
  ],
  "rooms": [
    {"name": "방이름", "boundary": [[x1,y1],[x2,y2],...], "area_m2": 면적, "confidence": 0.0-1.0}
  ],
  "annotations": [
    {"text": "치수|라벨", "position": [x,y], "type": "dimension|label", "confidence": 0.0-1.0}
  ]
}

정확한 좌표와 높은 신뢰도로 응답하세요.
""",
            
            "pattern_analysis": """
이 건축 도면의 패턴을 분석하여 JSON으로 응답하세요:

{
  "layout_pattern": "복도형|홀형|분산형",
  "circulation_pattern": "선형|방사형|격자형",
  "structural_pattern": "벽식|기둥식|혼합식",
  "room_arrangement": {
    "main_spaces": ["거실", "주방"],
    "private_spaces": ["침실1", "침실2"],
    "service_spaces": ["화장실", "다용도실"]
  },
  "design_principles": ["기능성", "효율성", "프라이버시"],
  "accessibility": {
    "barrier_free": true|false,
    "circulation_width": 폭,
    "level_differences": 단차수
  }
}
""",
            
            "quality_assessment": """
이 건축 도면의 품질을 평가하여 JSON으로 응답하세요:

{
  "drawing_quality": {
    "line_clarity": 0.0-1.0,
    "text_legibility": 0.0-1.0,
    "symbol_consistency": 0.0-1.0,
    "dimension_completeness": 0.0-1.0,
    "overall_score": 0.0-1.0
  },
  "information_completeness": {
    "structural_elements": 0.0-1.0,
    "architectural_elements": 0.0-1.0,
    "annotations": 0.0-1.0,
    "dimensions": 0.0-1.0
  },
  "issues_detected": [
    {"type": "missing_dimension", "location": [x,y], "severity": "low|medium|high"}
  ],
  "recommendations": ["개선사항1", "개선사항2"]
}
"""
        }
    
    def load_model(self) -> bool:
        """vLLM 모델 로드"""
        try:
            logger.info(f"Loading vLLM model: {self.model_name}")
            
            # vLLM 모델 초기화 (Context7 최적화 적용)
            self.model = LLM(
                **self.vllm_config,
                # Context7 문서 기반 멀티모달 최적화
                limit_mm_per_prompt={"image": 4},  # 이미지 입력 제한
            )
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            logger.info("vLLM model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            return False
    
    def analyze_image(self, 
                     image: Image.Image, 
                     analysis_type: str = "element_detection",
                     custom_prompt: str = None,
                     sampling_params: Optional[SamplingParams] = None) -> Dict[str, Any]:
        """
        이미지 분석 실행
        
        Args:
            image: PIL Image 객체
            analysis_type: 분석 타입 (element_detection, pattern_analysis, quality_assessment)
            custom_prompt: 사용자 정의 프롬프트
            sampling_params: 샘플링 파라미터
            
        Returns:
            분석 결과
        """
        if not self.model:
            logger.warning("Model not loaded, attempting to load...")
            if not self.load_model():
                return {"error": "Failed to load vLLM model"}
        
        try:
            # 프롬프트 선택
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self.architectural_prompts.get(
                    analysis_type, 
                    self.architectural_prompts["element_detection"]
                )
            
            # vLLM 멀티모달 입력 구성 (최신 API 방식)
            prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            # 샘플링 파라미터 설정
            if sampling_params is None:
                sampling_params = self.default_sampling_params
            
            logger.info(f"Running vLLM inference for {analysis_type}")
            
            # vLLM 추론 실행 (Context7 최신 방식)
            outputs = self.model.generate(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": {"image": image}
                },
                sampling_params=sampling_params
            )
            
            # 결과 추출
            response = outputs[0].outputs[0].text.strip()
            
            logger.info("vLLM inference completed")
            
            # JSON 파싱 시도
            parsed_result = self._parse_response(response)
            
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "raw_response": response,
                "parsed_result": parsed_result,
                "model_info": {
                    "model_name": self.model_name,
                    "inference_engine": "vLLM",
                    "config": self.vllm_config
                }
            }
            
        except Exception as e:
            logger.error(f"vLLM analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_type": analysis_type
            }
    
    async def analyze_batch(self, 
                           images: List[Image.Image],
                           analysis_types: List[str] = None,
                           custom_prompts: List[str] = None,
                           sampling_params: Optional[SamplingParams] = None) -> List[Dict[str, Any]]:
        """
        배치 이미지 분석 (비동기, Context7 최적화 적용)
        
        Args:
            images: PIL Image 객체 리스트
            analysis_types: 분석 타입 리스트
            custom_prompts: 사용자 정의 프롬프트 리스트
            sampling_params: 샘플링 파라미터
            
        Returns:
            분석 결과 리스트
        """
        if not self.model:
            if not self.load_model():
                return [{"error": "Failed to load vLLM model"}] * len(images)
        
        # 기본값 설정
        if analysis_types is None:
            analysis_types = ["element_detection"] * len(images)
        if custom_prompts is None:
            custom_prompts = [None] * len(images)
        
        # Context7 문서 기반: 배치 크기 제한
        batch_size = min(len(images), 8)  # vLLM 권장 배치 크기
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_types = analysis_types[i:i+batch_size] if i+batch_size <= len(analysis_types) else analysis_types[i:]
            batch_prompts = custom_prompts[i:i+batch_size] if i+batch_size <= len(custom_prompts) else custom_prompts[i:]
            
            # 배치 처리
            batch_tasks = []
            for j, image in enumerate(batch_images):
                analysis_type = batch_types[j] if j < len(batch_types) else batch_types[0]
                custom_prompt = batch_prompts[j] if j < len(batch_prompts) else None
                
                task = asyncio.create_task(
                    self._analyze_single_async(image, analysis_type, custom_prompt, sampling_params)
                )
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # 예외 처리
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append({
                        "status": "error",
                        "error": str(result)
                    })
                else:
                    results.append(result)
        
        return results
    
    async def _analyze_single_async(self, 
                                   image: Image.Image,
                                   analysis_type: str,
                                   custom_prompt: str,
                                   sampling_params: Optional[SamplingParams]) -> Dict[str, Any]:
        """단일 이미지 비동기 분석"""
        return await asyncio.to_thread(
            self.analyze_image, 
            image, 
            analysis_type, 
            custom_prompt, 
            sampling_params
        )
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """응답 파싱"""
        try:
            # JSON 추출 시도
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                parsed_data = json.loads(json_match.group())
                return parsed_data
            else:
                # JSON이 없으면 텍스트 분석
                return {"text_analysis": response}
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return {"text_analysis": response, "parse_error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "engine": "vLLM",
            "config": self.vllm_config,
            "loaded": self.model is not None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.model is not None:
                # vLLM 모델 정리
                del self.model
                self.model = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            logger.info("vLLM resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main():
    """테스트 메인 함수"""
    logging.basicConfig(level=logging.INFO)
    
    if not HAS_VLLM or not HAS_DEPS:
        print("❌ vLLM or dependencies not available")
        return
    
    # vLLM 분석기 초기화
    analyzer = VLLMAnalyzer(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8
    )
    
    # 모델 로드
    if analyzer.load_model():
        print("✅ vLLM model loaded successfully")
        
        # 모델 정보 출력
        model_info = analyzer.get_model_info()
        print(f"🔧 Model info: {json.dumps(model_info, indent=2)}")
        
        # 테스트 이미지 생성
        test_image = Image.new('RGB', (800, 600), 'white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        
        # 간단한 건축 도면 그리기
        draw.rectangle([100, 100, 700, 500], outline='black', width=3)  # 외벽
        draw.line([100, 300, 700, 300], fill='black', width=2)  # 내벽
        draw.line([400, 100, 400, 500], fill='black', width=2)  # 내벽
        draw.rectangle([200, 295, 220, 305], fill='black')  # 문
        draw.rectangle([500, 95, 550, 105], fill='black')  # 창문
        
        # 분석 실행
        result = analyzer.analyze_image(test_image, "element_detection")
        print("🔍 Analysis result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 정리
        analyzer.cleanup()
        
    else:
        print("❌ Failed to load vLLM model")


if __name__ == "__main__":
    main()
