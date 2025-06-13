#!/usr/bin/env python3
"""
Qwen-VL 로컬 모델 기반 VLM 분석기 (수정된 깔끔한 버전)
로컬 Qwen-VL 모델을 사용하여 PDF 벡터 그래픽을 분석
"""

import os
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# 필수 라이브러리 확인 및 import
try:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from PIL import Image, ImageDraw
    import numpy as np
    HAS_QWEN_DEPS = True
except ImportError as e:
    HAS_QWEN_DEPS = False
    print(f"Warning: Qwen-VL dependencies not available: {e}")

logger = logging.getLogger(__name__)


class QwenVLMAnalyzer:
    """Qwen-VL 로컬 모델 기반 VLM 분석기"""
    
    def __init__(self, model_path: str = None, processor_path: str = None):
        """
        Qwen-VL 분석기 초기화
        
        Args:
            model_path: Qwen-VL 모델 경로
            processor_path: Qwen-VL 프로세서 경로
        """
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 기본 경로 설정 - Qwen2.5-VL은 모델과 프로세서가 같은 폴더에 있음
        if model_path is None:
            current_dir = Path(__file__).parent.parent
            model_path = current_dir / "models" / "qwen_vlm_model"
        if processor_path is None:
            # Qwen2.5-VL의 경우 프로세서도 모델 폴더에서 로드
            current_dir = Path(__file__).parent.parent
            processor_path = current_dir / "models" / "qwen_vlm_model"
        
        self.model_path = Path(model_path)
        self.processor_path = Path(processor_path)
        
        # VLM 분석 프롬프트 템플릿
        self.analysis_prompts = {
            'architectural_basic': """
이 건축 도면 이미지를 분석해주세요. 다음 요소들을 식별하고 JSON 형식으로 응답해주세요:

{
  "architectural_elements": [
    {"type": "wall|door|window|stair", "position": [x, y], "description": "설명"}
  ],
  "structural_elements": [
    {"type": "column|beam|slab", "position": [x, y], "description": "설명"}
  ],
  "annotation_elements": [
    {"type": "dimension|text|symbol", "position": [x, y], "content": "내용"}
  ],
  "analysis_summary": "전체적인 도면 분석 요약"
}

도면의 구조와 요소들을 정확히 식별하여 응답해주세요.
""",
            
            'element_detection': """
이 이미지에서 건축 요소들을 식별하여 JSON 형식으로 응답해주세요:

{
  "detected_elements": [
    {
      "element_type": "wall|door|window|column|stair",
      "bounding_box": [x1, y1, x2, y2],
      "center_point": [x, y],
      "confidence": "high|medium|low",
      "description": "상세 설명"
    }
  ],
  "detection_summary": {
    "total_elements": 0,
    "by_type": {"walls": 0, "doors": 0, "windows": 0},
    "image_dimensions": [width, height],
    "analysis_quality": "분석 품질 평가"
  }
}
"""
        }
    
    def load_model(self) -> bool:
        """Qwen-VL 모델과 프로세서 로드"""
        if not HAS_QWEN_DEPS:
            logger.error("Qwen-VL dependencies not available")
            return False
            
        try:
            logger.info(f"Loading Qwen-VL model from {self.model_path}")
            
            # 모델 로드 - Qwen2.5-VL 사용, GPU 명시적 설정
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16,
                device_map="cuda:0" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # 모델을 명시적으로 GPU로 이동
            if self.device == "cuda" and self.model is not None:
                self.model = self.model.to(self.device)
            
            # 프로세서 로드 - 모델과 같은 경로에서 로드 (Qwen2.5-VL)
            logger.info(f"Loading Qwen-VL processor from {self.model_path}")
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            logger.info(f"Qwen-VL model and processor loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen-VL model: {e}")
            return False
    
    def cleanup_memory(self):
        """GPU 메모리 정리"""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cleaned up")
        except Exception as e:
            logger.warning(f"Error during memory cleanup: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """현재 메모리 사용량 반환"""
        memory_info = {"system_ram_gb": 0.0}
        
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
                
                memory_info.update({
                    "gpu_total_gb": gpu_mem,
                    "gpu_allocated_gb": gpu_allocated,
                    "gpu_cached_gb": gpu_cached,
                    "gpu_free_gb": gpu_mem - gpu_cached
                })
        except Exception as e:
            logger.warning(f"Error getting memory usage: {e}")
            
        return memory_info
    
    def analyze_image(self, image: Image.Image, prompt_type: str = 'architectural_basic', 
                     custom_prompt: str = None) -> Dict[str, Any]:
        """
        이미지 분석 실행
        
        Args:
            image: PIL Image 객체
            prompt_type: 분석 프롬프트 타입
            custom_prompt: 사용자 정의 프롬프트
            
        Returns:
            분석 결과 딕셔너리
        """
        if not self.model or not self.processor:
            logger.warning("Model not loaded, attempting to load...")
            if not self.load_model():
                return {"error": "Failed to load Qwen-VL model"}
        
        try:
            # 프롬프트 선택
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self.analysis_prompts.get(prompt_type, 
                                                 self.analysis_prompts['architectural_basic'])
            
            # 이미지 전처리 및 모델 입력 준비
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 입력 토큰화
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            
            # GPU로 이동 (사용 가능한 경우)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론 실행
            logger.info("Running VLM inference...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # 결과 디코딩
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.processor.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            
            logger.info("VLM analysis completed successfully")
            
            # 메모리 정리
            self.cleanup_memory()
            
            # JSON 파싱 시도
            analysis_result = self._parse_vlm_response(response)
            
            return {
                "status": "success",
                "prompt_type": prompt_type,
                "raw_response": response,
                "parsed_result": analysis_result,
                "model_info": {
                    "device": self.device,
                    "model_path": str(self.model_path)
                }
            }
            
        except Exception as e:
            logger.error(f"VLM analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "prompt_type": prompt_type
            }
    
    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """VLM 응답을 구조화된 데이터로 파싱 (개선된 버전)"""
        try:
            logger.info("Starting VLM response parsing...")
            
            # 응답 텍스트 제한 (메모리 보호)
            if len(response) > 50000:  # 50KB 제한
                response = response[:50000]
                logger.warning("Response truncated to 50KB for parsing")
            
            # 간단한 JSON 패턴 찾기
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            
            if json_match:
                try:
                    parsed_data = json.loads(json_match.group())
                    logger.info("Successfully parsed JSON from VLM response")
                    return parsed_data
                except json.JSONDecodeError:
                    pass
            
            logger.info("No valid JSON found, using text parsing")
            return self._parse_text_response(response)
                
        except Exception as e:
            logger.error(f"Error parsing VLM response: {e}")
            return {"raw_text": response[:1000] + "..." if len(response) > 1000 else response, 
                   "parse_error": str(e)}
    
    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """텍스트 응답을 구조화된 형태로 파싱 (제한된 처리)"""
        # 텍스트 길이 제한
        if len(response) > 10000:
            response = response[:10000]
        
        lines = response.strip().split('\n')[:50]  # 최대 50줄만 처리
        result = {
            "text_analysis": response,
            "key_points": [],
            "detected_elements": []
        }
        
        # 간단한 키워드 기반 요소 추출
        architectural_keywords = ["wall", "door", "window", "stair", "벽", "문", "창문", "계단"]
        
        for line in lines[:20]:  # 처음 20줄만 검사
            line = line.strip()
            if line and len(line) > 3:
                # 간단한 키워드 매칭
                found_keyword = None
                for keyword in architectural_keywords:
                    if keyword.lower() in line.lower():
                        found_keyword = keyword
                        break
                
                if found_keyword:
                    result["detected_elements"].append({
                        "element": found_keyword,
                        "description": line[:200]  # 설명 길이 제한
                    })
                else:
                    if len(result["key_points"]) < 10:  # 최대 10개 키포인트
                        result["key_points"].append(line[:200])
        
        return result


def main():
    """테스트용 메인 함수"""
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # VLM 분석기 초기화
    analyzer = QwenVLMAnalyzer()
    
    # 모델 로드 테스트
    if analyzer.load_model():
        print("✅ Qwen-VL model loaded successfully")
        
        # 테스트 이미지 생성
        test_image = Image.new('RGB', (400, 300), 'white')
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([50, 50, 350, 250], outline='black', width=2)
        draw.line([50, 150, 350, 150], fill='black', width=1)
        draw.line([200, 50, 200, 250], fill='black', width=1)
        
        # 분석 테스트
        result = analyzer.analyze_image(test_image, 'element_detection')
        print("🔍 VLM Analysis Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    else:
        print("❌ Failed to load Qwen-VL model")


if __name__ == "__main__":
    main()
