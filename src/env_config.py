"""
환경 설정 관리 모듈
.env 파일을 읽어 시스템 설정을 동적으로 관리합니다.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import json

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("python-dotenv가 설치되지 않았습니다. pip install python-dotenv를 실행하세요.")

import torch


@dataclass
class DeviceConfig:
    """디바이스 설정 클래스"""
    device_type: str
    device: torch.device
    gpu_memory_utilization: float
    tensor_parallel_size: int
    pipeline_parallel_size: int


@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    model_name: str
    trust_remote_code: bool
    max_model_len: int
    quantization: str
    kv_cache_dtype: str


@dataclass
class GenerationConfig:
    """생성 설정 클래스"""
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


class EnvironmentConfig:
    """환경 설정 관리 클래스"""
    
    def __init__(self, env_file: str = ".env"):
        """
        환경 설정 초기화
        
        Args:
            env_file: .env 파일 경로
        """
        self.env_file = env_file
        self.config = {}
        self._load_config()
        
    def _load_config(self):
        """환경 설정 로드"""
        # .env 파일 로드
        if DOTENV_AVAILABLE and os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            print(f"✅ 환경 설정 파일 로드됨: {self.env_file}")
        elif os.path.exists(self.env_file):
            print(f"⚠️  python-dotenv가 없어 {self.env_file}를 수동으로 파싱합니다")
            self._manual_load_env()
        else:
            print(f"⚠️  환경 설정 파일이 없습니다: {self.env_file}")
            print("기본 설정을 사용합니다. .env.template을 참조하여 .env 파일을 생성하세요.")
            
        # 환경 변수에서 설정 읽기
        self._load_from_env()
        
    def _manual_load_env(self):
        """수동으로 .env 파일 파싱"""
        try:
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            print(f"⚠️  .env 파일 수동 로드 실패: {e}")
            
    def _load_from_env(self):
        """환경 변수에서 설정 로드"""
        # 모델 설정
        self.config['model_name'] = os.getenv('MODEL_NAME', 'Qwen/Qwen2-VL-7B-Instruct')
        self.config['trust_remote_code'] = self._get_bool('TRUST_REMOTE_CODE', True)
        self.config['max_model_len'] = self._get_int('MAX_MODEL_LEN', 4096)
        
        # 디바이스 설정
        self.config['device'] = os.getenv('DEVICE', 'auto')
        self.config['gpu_memory_utilization'] = self._get_float('GPU_MEMORY_UTILIZATION', 0.8)
        self.config['cuda_visible_devices'] = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        
        # vLLM 설정
        self.config['use_vllm'] = self._get_bool('USE_VLLM', True)
        self.config['vllm_engine_type'] = os.getenv('VLLM_ENGINE_TYPE', 'llm')
        self.config['tensor_parallel_size'] = self._get_int('TENSOR_PARALLEL_SIZE', 1)
        self.config['pipeline_parallel_size'] = self._get_int('PIPELINE_PARALLEL_SIZE', 1)
        self.config['max_num_seqs'] = self._get_int('MAX_NUM_SEQS', 256)
        self.config['block_size'] = self._get_int('BLOCK_SIZE', 16)
        
        # 양자화 설정
        self.config['quantization'] = os.getenv('QUANTIZATION', 'none')
        self.config['awq_config'] = os.getenv('AWQ_CONFIG', 'auto')
        
        # 추론 최적화
        self.config['kv_cache_dtype'] = os.getenv('KV_CACHE_DTYPE', 'auto')
        self.config['attention_backend'] = os.getenv('ATTENTION_BACKEND', 'auto')
        self.config['enable_chunked_prefill'] = self._get_bool('ENABLE_CHUNKED_PREFILL', False)
        self.config['max_num_batched_tokens'] = self._get_int('MAX_NUM_BATCHED_TOKENS', 512)
        
        # 스케줄링 설정
        self.config['scheduler_type'] = os.getenv('SCHEDULER_TYPE', 'fcfs')
        self.config['swap_space'] = self._get_int('SWAP_SPACE', 4)
        self.config['cpu_offload_gb'] = self._get_int('CPU_OFFLOAD_GB', 0)
        
        # 생성 설정
        self.config['default_max_tokens'] = self._get_int('DEFAULT_MAX_TOKENS', 2048)
        self.config['default_temperature'] = self._get_float('DEFAULT_TEMPERATURE', 0.7)
        self.config['default_top_p'] = self._get_float('DEFAULT_TOP_P', 0.9)
        self.config['default_top_k'] = self._get_int('DEFAULT_TOP_K', 50)
        self.config['default_repetition_penalty'] = self._get_float('DEFAULT_REPETITION_PENALTY', 1.1)
        
        # 시스템 설정
        self.config['log_level'] = os.getenv('LOG_LEVEL', 'INFO')
        self.config['log_file'] = os.getenv('LOG_FILE', 'logs/vllm_system.log')
        self.config['upload_dir'] = os.getenv('UPLOAD_DIR', 'uploads')
        self.config['model_dir'] = os.getenv('MODEL_DIR', 'models')
        self.config['config_dir'] = os.getenv('CONFIG_DIR', 'configs')
        
        # 성능 모니터링
        self.config['enable_metrics'] = self._get_bool('ENABLE_METRICS', True)
        self.config['metrics_port'] = self._get_int('METRICS_PORT', 8080)
        self.config['enable_profiling'] = self._get_bool('ENABLE_PROFILING', False)
        
        # API 서버 설정
        self.config['server_host'] = os.getenv('SERVER_HOST', 'localhost')
        self.config['server_port'] = self._get_int('SERVER_PORT', 8000)
        self.config['num_workers'] = self._get_int('NUM_WORKERS', 1)
        
        # 멀티모달 설정
        self.config['max_image_size'] = self._get_int('MAX_IMAGE_SIZE', 1024)
        self.config['supported_image_formats'] = os.getenv('SUPPORTED_IMAGE_FORMATS', 'jpg,jpeg,png,bmp,tiff,webp').split(',')
        self.config['image_preprocessing'] = os.getenv('IMAGE_PREPROCESSING', 'auto')
        
        # 실험적 기능
        self.config['enable_experimental'] = self._get_bool('ENABLE_EXPERIMENTAL', False)
        self.config['experimental_flags'] = os.getenv('EXPERIMENTAL_FLAGS', '')
        
        # 환경별 오버라이드
        self.config['development_mode'] = self._get_bool('DEVELOPMENT_MODE', False)
        self.config['debug_mode'] = self._get_bool('DEBUG_MODE', False)
        self.config['benchmark_mode'] = self._get_bool('BENCHMARK_MODE', False)
        
    def _get_bool(self, key: str, default: bool) -> bool:
        """불린 값 파싱"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
        
    def _get_int(self, key: str, default: int) -> int:
        """정수 값 파싱"""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default
            
    def _get_float(self, key: str, default: float) -> float:
        """실수 값 파싱"""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default
            
    def get_device_config(self) -> DeviceConfig:
        """디바이스 설정 반환"""
        device_type = self.config['device']
        
        # 자동 디바이스 감지
        if device_type == 'auto':
            if torch.cuda.is_available():
                device_type = 'cuda'
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_type = 'mps'
                device = torch.device('mps')
            else:
                device_type = 'cpu'
                device = torch.device('cpu')
        else:
            device = torch.device(device_type)
            
        return DeviceConfig(
            device_type=device_type,
            device=device,
            gpu_memory_utilization=self.config['gpu_memory_utilization'],
            tensor_parallel_size=self.config['tensor_parallel_size'],
            pipeline_parallel_size=self.config['pipeline_parallel_size']
        )
        
    def get_model_config(self) -> ModelConfig:
        """모델 설정 반환"""
        return ModelConfig(
            model_name=self.config['model_name'],
            trust_remote_code=self.config['trust_remote_code'],
            max_model_len=self.config['max_model_len'],
            quantization=self.config['quantization'],
            kv_cache_dtype=self.config['kv_cache_dtype']
        )
        
    def get_generation_config(self) -> GenerationConfig:
        """생성 설정 반환"""
        return GenerationConfig(
            max_tokens=self.config['default_max_tokens'],
            temperature=self.config['default_temperature'],
            top_p=self.config['default_top_p'],
            top_k=self.config['default_top_k'],
            repetition_penalty=self.config['default_repetition_penalty']
        )
        
    def get_vllm_engine_args(self) -> Dict[str, Any]:
        """vLLM 엔진 인수 반환"""
        device_config = self.get_device_config()
        model_config = self.get_model_config()
        
        args = {
            'model': model_config.model_name,
            'trust_remote_code': model_config.trust_remote_code,
            'max_model_len': model_config.max_model_len,
            'tensor_parallel_size': device_config.tensor_parallel_size,
            'pipeline_parallel_size': device_config.pipeline_parallel_size,
            'max_num_seqs': self.config['max_num_seqs'],
            'block_size': self.config['block_size'],
            'swap_space': self.config['swap_space'],
            'cpu_offload_gb': self.config['cpu_offload_gb'],
            'gpu_memory_utilization': device_config.gpu_memory_utilization,
            'kv_cache_dtype': model_config.kv_cache_dtype,
            'enable_chunked_prefill': self.config['enable_chunked_prefill'],
            'max_num_batched_tokens': self.config['max_num_batched_tokens'],
        }
        
        # 양자화 설정
        if model_config.quantization != 'none':
            args['quantization'] = model_config.quantization
            if model_config.quantization == 'awq' and self.config['awq_config'] != 'auto':
                args['quantization_param_path'] = self.config['awq_config']
                
        # 어텐션 백엔드 설정
        if self.config['attention_backend'] != 'auto':
            args['attention_backend'] = self.config['attention_backend']
            
        return args
        
    def setup_logging(self):
        """로깅 설정"""
        log_level = getattr(logging, self.config['log_level'].upper())
        log_file = self.config['log_file']
        
        # 로그 디렉토리 생성
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def create_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.config['upload_dir'],
            self.config['model_dir'],
            self.config['config_dir'],
            Path(self.config['log_file']).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기"""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any):
        """설정 값 설정"""
        self.config[key] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 반환"""
        return self.config.copy()
        
    def save_config(self, filename: str):
        """설정을 JSON 파일로 저장"""
        config_path = Path(self.config['config_dir']) / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
            
    def print_config(self):
        """현재 설정 출력"""
        print("🔧 현재 환경 설정:")
        print(f"   모델: {self.config['model_name']}")
        print(f"   디바이스: {self.config['device']}")
        print(f"   vLLM 사용: {self.config['use_vllm']}")
        print(f"   양자화: {self.config['quantization']}")
        print(f"   GPU 메모리 사용률: {self.config['gpu_memory_utilization']}")
        print(f"   최대 시퀀스 길이: {self.config['max_model_len']}")
        print(f"   텐서 병렬 처리: {self.config['tensor_parallel_size']}")
        print(f"   로그 레벨: {self.config['log_level']}")


# 전역 환경 설정 인스턴스
env_config = EnvironmentConfig()


def get_env_config() -> EnvironmentConfig:
    """전역 환경 설정 인스턴스 반환"""
    return env_config


if __name__ == "__main__":
    # 테스트 실행
    config = EnvironmentConfig()
    config.print_config()
    
    # 디바이스 설정 테스트
    device_config = config.get_device_config()
    print(f"\n🎯 감지된 디바이스: {device_config.device_type} ({device_config.device})")
    
    # vLLM 엔진 인수 테스트
    vllm_args = config.get_vllm_engine_args()
    print(f"\n⚡ vLLM 엔진 설정: {json.dumps(vllm_args, indent=2, ensure_ascii=False)}")
