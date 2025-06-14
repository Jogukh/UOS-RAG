#!/usr/bin/env python3
"""
자동 생성된 vLLM 실행 스크립트
설정: {
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.8,
  "enable_chunked_prefill": false,
  "max_num_seqs": 2,
  "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
  "max_model_len": 8192,
  "dtype": "bfloat16",
  "trust_remote_code": true,
  "enforce_eager": true
}
"""

import logging
from vllm import LLM, SamplingParams
from vllm_analyzer import VLLMAnalyzer

def main():
    logging.basicConfig(level=logging.INFO)
    
    # vLLM 분석기 초기화
    analyzer = VLLMAnalyzer(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        dtype="bfloat16"
    )
    
    # 모델 로드
    if analyzer.load_model():
        print("✅ vLLM model loaded successfully")
        
        # 여기에 분석 코드 추가
        # analyzer.analyze_image(image, "element_detection")
        
    else:
        print("❌ Failed to load vLLM model")

if __name__ == "__main__":
    main()
