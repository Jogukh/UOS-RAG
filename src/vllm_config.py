#!/usr/bin/env python3
"""
vLLM 설정 및 최적화 가이드
VLM 모델을 vLLM으로 효율적으로 실행하기 위한 설정
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VLLMConfig:
    """vLLM 설정 관리"""
    
    def __init__(self):
        self.base_config = {
            # 모델 설정
            "model_configs": {
                "qwen2.5-7b-text": {
                    "model_name": "Qwen/Qwen2.5-7B-Instruct",
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.7,
                    "max_model_len": 32768,  # 긴 컨텍스트 지원
                    "dtype": "bfloat16",
                    "trust_remote_code": True
                },
                "qwen2.5-vl-7b": {
                    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.8,
                    "max_model_len": 8192,
                    "dtype": "bfloat16",
                    "trust_remote_code": True
                },
                "qwen2.5-vl-72b": {
                    "model_name": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "tensor_parallel_size": 4,  # 큰 모델용
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 4096,
                    "dtype": "bfloat16",
                    "trust_remote_code": True
                }
            },
            
            # 추론 최적화 설정 (Context7 권장사항 적용)
            "inference_configs": {
                "fast": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "max_tokens": 1024,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "repetition_penalty": 1.02,  # Context7 권장
                },
                "balanced": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048,
                    "frequency_penalty": 0.1,
                    "presence_penalty": 0.1,
                    "repetition_penalty": 1.05,  # Context7 권장
                },
                "creative": {
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "max_tokens": 3072,
                    "frequency_penalty": 0.2,
                    "presence_penalty": 0.2,
                    "repetition_penalty": 1.1,   # Context7 권장
                }
            },
            
            # 양자화 설정 (Context7 문서 기반)
            "quantization_configs": {
                "none": {},
                "int8": {"quantization": "bitsandbytes"},
                "int4": {"quantization": "gptq"},
                "fp8": {"quantization": "modelopt"},
            },
            
            # 하드웨어별 최적화
            "hardware_configs": {
                "single_gpu": {
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.8,
                    "enable_chunked_prefill": True,
                    "max_num_seqs": 8
                },
                "multi_gpu": {
                    "tensor_parallel_size": 2,
                    "gpu_memory_utilization": 0.9,
                    "enable_chunked_prefill": True,
                    "max_num_seqs": 16
                },
                "high_memory": {
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.95,
                    "enable_chunked_prefill": False,
                    "max_num_seqs": 32
                }
            }
        }
    
    def get_model_config(self, model_type: str = "qwen2.5-vl-7b") -> Dict[str, Any]:
        """모델 설정 반환"""
        return self.base_config["model_configs"].get(model_type, 
                                                   self.base_config["model_configs"]["qwen2.5-vl-7b"])
    
    def get_inference_config(self, mode: str = "balanced") -> Dict[str, Any]:
        """추론 설정 반환"""
        return self.base_config["inference_configs"].get(mode,
                                                       self.base_config["inference_configs"]["balanced"])
    
    def get_hardware_config(self, hardware_type: str = "single_gpu") -> Dict[str, Any]:
        """하드웨어 설정 반환"""
        return self.base_config["hardware_configs"].get(hardware_type,
                                                      self.base_config["hardware_configs"]["single_gpu"])
    
    def auto_detect_config(self) -> Dict[str, Any]:
        """자동 설정 감지"""
        import torch
        
        config = {}
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"Detected {gpu_count} GPU(s) with {gpu_memory:.1f}GB VRAM")
            
            # GPU 개수에 따른 설정
            if gpu_count == 1:
                config.update(self.get_hardware_config("single_gpu"))
                if gpu_memory > 24:
                    config["gpu_memory_utilization"] = 0.9
                    config["max_num_seqs"] = 16
                elif gpu_memory < 12:
                    config["gpu_memory_utilization"] = 0.7
                    config["max_num_seqs"] = 4
            else:
                config.update(self.get_hardware_config("multi_gpu"))
                config["tensor_parallel_size"] = min(gpu_count, 4)
            
            # 모델 선택
            if gpu_memory * gpu_count > 80:
                config.update(self.get_model_config("qwen2.5-vl-72b"))
            else:
                config.update(self.get_model_config("qwen2.5-vl-7b"))
                
        else:
            logger.warning("No CUDA GPUs detected, falling back to CPU")
            config = {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.0,
                "device": "cpu"
            }
        
        return config
    
    def save_config(self, config: Dict[str, Any], file_path: str):
        """설정을 파일로 저장"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Config saved to {file_path}")
    
    def load_config(self, file_path: str) -> Dict[str, Any]:
        """파일에서 설정 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Config loaded from {file_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {file_path}")
            return self.auto_detect_config()


class VLLMOptimizer:
    """vLLM 성능 최적화 도구"""
    
    @staticmethod
    def optimize_for_throughput(config: Dict[str, Any]) -> Dict[str, Any]:
        """처리량 최적화"""
        optimized = config.copy()
        optimized.update({
            "max_num_seqs": min(32, optimized.get("max_num_seqs", 8) * 2),
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": min(0.95, optimized.get("gpu_memory_utilization", 0.8) + 0.1)
        })
        return optimized
    
    @staticmethod
    def optimize_for_latency(config: Dict[str, Any]) -> Dict[str, Any]:
        """지연시간 최적화"""
        optimized = config.copy()
        optimized.update({
            "max_num_seqs": max(1, optimized.get("max_num_seqs", 8) // 2),
            "enable_chunked_prefill": False,
            "enforce_eager": True,  # CUDA graph 비활성화로 첫 번째 요청 속도 향상
            "gpu_memory_utilization": optimized.get("gpu_memory_utilization", 0.8)
        })
        return optimized
    
    @staticmethod
    def optimize_for_memory(config: Dict[str, Any]) -> Dict[str, Any]:
        """메모리 최적화"""
        optimized = config.copy()
        optimized.update({
            "gpu_memory_utilization": min(0.7, optimized.get("gpu_memory_utilization", 0.8)),
            "max_model_len": min(4096, optimized.get("max_model_len", 8192)),
            "enable_chunked_prefill": True,
            "block_size": 8  # 메모리 블록 크기 감소
        })
        return optimized


def create_vllm_launch_script(config: Dict[str, Any], output_path: str = "launch_vllm.py"):
    """vLLM 실행 스크립트 생성"""
    
    script_content = f'''#!/usr/bin/env python3
"""
자동 생성된 vLLM 실행 스크립트
설정: {json.dumps(config, indent=2)}
"""

import logging
from vllm import LLM, SamplingParams
from vllm_analyzer import VLLMAnalyzer

def main():
    logging.basicConfig(level=logging.INFO)
    
    # vLLM 분석기 초기화
    analyzer = VLLMAnalyzer(
        model_name="{config.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')}",
        tensor_parallel_size={config.get('tensor_parallel_size', 1)},
        gpu_memory_utilization={config.get('gpu_memory_utilization', 0.8)},
        max_model_len={config.get('max_model_len', 8192)},
        dtype="{config.get('dtype', 'bfloat16')}"
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
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 실행 권한 부여
    os.chmod(output_path, 0o755)
    
    print(f"✅ vLLM launch script created: {output_path}")


def main():
    """설정 도구 메인 함수"""
    logging.basicConfig(level=logging.INFO)
    
    config_manager = VLLMConfig()
    
    # 자동 설정 감지
    print("🔍 Detecting optimal vLLM configuration...")
    auto_config = config_manager.auto_detect_config()
    print(f"📋 Auto-detected config: {json.dumps(auto_config, indent=2)}")
    
    # 최적화 옵션
    optimizer = VLLMOptimizer()
    
    print("⚡ Throughput-optimized config:")
    throughput_config = optimizer.optimize_for_throughput(auto_config)
    print(json.dumps(throughput_config, indent=2))
    
    print("🚀 Latency-optimized config:")
    latency_config = optimizer.optimize_for_latency(auto_config)
    print(json.dumps(latency_config, indent=2))
    
    print("💾 Memory-optimized config:")
    memory_config = optimizer.optimize_for_memory(auto_config)
    print(json.dumps(memory_config, indent=2))
    
    # 설정 파일 저장
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    config_manager.save_config(auto_config, "configs/vllm_auto.json")
    config_manager.save_config(throughput_config, "configs/vllm_throughput.json")
    config_manager.save_config(latency_config, "configs/vllm_latency.json")
    config_manager.save_config(memory_config, "configs/vllm_memory.json")
    
    # 실행 스크립트 생성
    create_vllm_launch_script(auto_config, "launch_vllm_auto.py")
    create_vllm_launch_script(throughput_config, "launch_vllm_throughput.py")
    create_vllm_launch_script(latency_config, "launch_vllm_latency.py")
    
    print("✅ vLLM configuration completed!")


if __name__ == "__main__":
    main()
