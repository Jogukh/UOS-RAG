#!/usr/bin/env python3
"""
통합 VLM 분석기 - unsloth와 vLLM 모두 지원
"""

import os
import logging
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional, Union
import json
from PIL import Image

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 의존성 확인
try:
    from unsloth import FastVisionModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    logger.warning("unsloth not available")

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    logger.warning("vLLM not available")

try:
    from transformers import AutoTokenizer, TextStreamer
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available")


class UnifiedVLMAnalyzer:
    """통합 VLM 분석기 - unsloth와 vLLM 모두 지원"""
    
    def __init__(self, 
                 model_name: str = "sabaridsnfuji/FloorPlanVisionAIAdaptor",
                 engine_type: str = "auto",
                 **kwargs):
        """
        통합 VLM 분석기 초기화
        
        Args:
            model_name: 모델 이름
            engine_type: 'unsloth', 'vllm', 'auto'
            **kwargs: 추가 설정
        """
        self.model_name = model_name
        self.engine_type = self._detect_engine_type(model_name, engine_type)
        self.model = None
        self.tokenizer = None
        self.kwargs = kwargs
        
        logger.info(f"Initialized UnifiedVLMAnalyzer with engine: {self.engine_type}")
        
    def _detect_engine_type(self, model_name: str, engine_type: str) -> str:
        """엔진 타입 자동 감지"""
        if engine_type != "auto":
            return engine_type
            
        # 특정 모델들에 대한 자동 감지
        unsloth_models = [
            "sabaridsnfuji/FloorPlanVisionAIAdaptor",
            "unsloth/llama",
            "unsloth/mistral"
        ]
        
        for unsloth_model in unsloth_models:
            if unsloth_model in model_name.lower():
                return "unsloth" if HAS_UNSLOTH else "vllm"
                
        # 기본적으로 vLLM 사용
        return "vllm" if HAS_VLLM else "unsloth"
        
    def load_model(self) -> bool:
        """모델 로드"""
        try:
            if self.engine_type == "unsloth":
                return self._load_unsloth_model()
            elif self.engine_type == "vllm":
                return self._load_vllm_model()
            else:
                logger.error(f"Unsupported engine type: {self.engine_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
            
    def _load_unsloth_model(self) -> bool:
        """unsloth 모델 로드"""
        if not HAS_UNSLOTH:
            logger.error("unsloth not available")
            return False
            
        logger.info(f"Loading unsloth model: {self.model_name}")
        
        # unsloth 모델 로드
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_name,
            load_in_4bit=True,  # 메모리 효율성
            use_gradient_checkpointing="unsloth"
        )
        
        # 추론 모드 활성화
        FastVisionModel.for_inference(self.model)
        
        logger.info("unsloth model loaded successfully")
        return True
        
    def _load_vllm_model(self) -> bool:
        """vLLM 모델 로드"""
        if not HAS_VLLM:
            logger.error("vLLM not available")
            return False
            
        logger.info(f"Loading vLLM model: {self.model_name}")
        
        # vLLM 설정
        vllm_config = {
            "model": self.model_name,
            "trust_remote_code": True,
            "tensor_parallel_size": self.kwargs.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": self.kwargs.get("gpu_memory_utilization", 0.8),
            "max_model_len": self.kwargs.get("max_model_len", 8192),
            "dtype": self.kwargs.get("dtype", "bfloat16"),
            "limit_mm_per_prompt": {"image": 4},
        }
        
        self.model = LLM(**vllm_config)
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        logger.info("vLLM model loaded successfully")
        return True
        
    def analyze_floor_plan(self, 
                          image: Image.Image,
                          instruction: str = None,
                          **generate_kwargs) -> Dict[str, Any]:
        """
        건축 도면 분석
        
        Args:
            image: PIL Image 객체
            instruction: 분석 지시사항
            **generate_kwargs: 생성 파라미터
            
        Returns:
            분석 결과
        """
        if not self.model:
            logger.warning("Model not loaded, attempting to load...")
            if not self.load_model():
                return {"error": "Failed to load model"}
                
        # 기본 instruction
        if instruction is None:
            instruction = """You are an expert in architecture and interior design. Analyze the floor plan image and describe accurately the key features, room count, layout, and any other important details you observe. Please provide the analysis in JSON format with the following structure:

{
  "room_count": {
    "total_rooms": number,
    "bedrooms": number,
    "bathrooms": number,
    "kitchen": number,
    "living_areas": number
  },
  "room_details": [
    {"name": "room_name", "dimensions": "width x height", "area_sqft": number}
  ],
  "layout_features": {
    "layout_type": "description",
    "circulation": "description",
    "primary_features": ["feature1", "feature2"]
  },
  "architectural_elements": {
    "doors": number,
    "windows": number,
    "stairs": boolean,
    "special_features": ["feature1", "feature2"]
  }
}"""
        
        try:
            if self.engine_type == "unsloth":
                return self._analyze_with_unsloth(image, instruction, **generate_kwargs)
            elif self.engine_type == "vllm":
                return self._analyze_with_vllm(image, instruction, **generate_kwargs)
            else:
                return {"error": f"Unsupported engine type: {self.engine_type}"}
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"error": str(e)}
            
    def _analyze_with_unsloth(self, image: Image.Image, instruction: str, **kwargs) -> Dict[str, Any]:
        """unsloth를 사용한 분석"""
        # 메시지 형식 구성
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        # 채팅 템플릿 적용
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # 입력 준비
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # 생성 파라미터
        generate_params = {
            "max_new_tokens": kwargs.get("max_new_tokens", 2048),
            "use_cache": True,
            "temperature": kwargs.get("temperature", 0.7),
            "min_p": kwargs.get("min_p", 0.1),
            "do_sample": True,
        }
        
        # 추론 실행
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_params)
            
        # 결과 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 입력 부분 제거
        if input_text in response:
            response = response.replace(input_text, "").strip()
            
        return {
            "status": "success",
            "engine": "unsloth",
            "model_name": self.model_name,
            "raw_response": response,
            "parsed_result": self._parse_json_response(response)
        }
        
    def _analyze_with_vllm(self, image: Image.Image, instruction: str, **kwargs) -> Dict[str, Any]:
        """vLLM을 사용한 분석"""
        # 프롬프트 구성
        prompt_text = f"USER: <image>\n{instruction}\nASSISTANT:"
        
        # 샘플링 파라미터
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_tokens", 2048),
            repetition_penalty=kwargs.get("repetition_penalty", 1.05)
        )
        
        # 추론 실행
        outputs = self.model.generate(
            {
                "prompt": prompt_text,
                "multi_modal_data": {"image": image}
            },
            sampling_params=sampling_params
        )
        
        response = outputs[0].outputs[0].text.strip()
        
        return {
            "status": "success",
            "engine": "vllm",
            "model_name": self.model_name,
            "raw_response": response,
            "parsed_result": self._parse_json_response(response)
        }
        
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """JSON 응답 파싱"""
        try:
            # JSON 블록 찾기
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # JSON이 없으면 텍스트 그대로 반환
                return {"analysis": response}
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response")
            return {"analysis": response}
            
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "engine_type": self.engine_type,
            "loaded": self.model is not None,
            "supports_unsloth": HAS_UNSLOTH,
            "supports_vllm": HAS_VLLM,
            "supports_transformers": HAS_TRANSFORMERS
        }


# 편의 함수들
def create_analyzer(model_name: str = None, **kwargs) -> UnifiedVLMAnalyzer:
    """분석기 생성 편의 함수"""
    if model_name is None:
        # .env에서 모델명 가져오기
        model_name = os.getenv("MODEL_NAME", "sabaridsnfuji/FloorPlanVisionAIAdaptor")
    
    return UnifiedVLMAnalyzer(model_name=model_name, **kwargs)


def analyze_floor_plan_image(image_path: str, **kwargs) -> Dict[str, Any]:
    """이미지 파일 경로로 직접 분석하는 편의 함수"""
    try:
        image = Image.open(image_path)
        analyzer = create_analyzer(**kwargs)
        return analyzer.analyze_floor_plan(image)
    except Exception as e:
        return {"error": f"Failed to analyze image: {e}"}


if __name__ == "__main__":
    # 테스트 실행
    analyzer = create_analyzer()
    info = analyzer.get_model_info()
    print("Model Info:", json.dumps(info, indent=2))
    
    # 모델 로드 테스트
    if analyzer.load_model():
        print("✅ Model loaded successfully")
    else:
        print("❌ Failed to load model")
