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
    # 프로젝트 루트의 .env 파일 로드
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    DOTENV_AVAILABLE = False
    print("python-dotenv가 설치되지 않았습니다. pip install python-dotenv를 실행하세요.")

import torch

logger = logging.getLogger(__name__)


def get_env_bool(key: str, default: bool = False) -> bool:
    """환경변수에서 boolean 값 읽기"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """환경변수에서 정수 값 읽기"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid integer value for {key}, using default: {default}")
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """환경변수에서 실수 값 읽기"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"Invalid float value for {key}, using default: {default}")
        return default


def get_env_str(key: str, default: str = "") -> str:
    """환경변수에서 문자열 값 읽기"""
    return os.getenv(key, default)


def get_env_list(key: str, default: list = None, separator: str = ",") -> list:
    """환경변수에서 리스트 값 읽기"""
    if default is None:
        default = []
    value = os.getenv(key, "")
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]


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
    """환경 설정 관리 클래스 - .env 파일 기반"""
    
    def __init__(self):
        """환경 설정 초기화"""
        self.config = {}
        self._load_config()
        
    def _load_config(self):
        """환경 설정 로드"""
        logger.info("환경 설정을 로드하는 중...")
        
        # 모델 설정
        self.model_config = ModelConfig(
            model_name=get_env_str('CUSTOM_MODEL_PATH'),
            trust_remote_code=True,
            max_model_len=get_env_int('MAX_MODEL_LEN', 8192),
            quantization=get_env_str('MODEL_DTYPE', 'bfloat16'),
            kv_cache_dtype=get_env_str('KV_CACHE_DTYPE', 'auto')
        )
        
        # 디바이스 설정
        self.device_config = DeviceConfig(
            device_type=self._detect_device_type(),
            device=self._get_device(),
            gpu_memory_utilization=get_env_float('GPU_MEMORY_UTILIZATION', 0.8),
            tensor_parallel_size=get_env_int('TENSOR_PARALLEL_SIZE', 1),
            pipeline_parallel_size=1
        )
        
        # 생성 설정
        self.generation_config = GenerationConfig(
            max_tokens=get_env_int('MAX_TOKENS', 1),
            temperature=get_env_float('TEMPERATURE', 0.0),
            top_p=get_env_float('TOP_P', 1.0),
            top_k=get_env_int('TOP_K', 50),
            repetition_penalty=get_env_float('REPETITION_PENALTY', 1.0)
        )
        
        # 추가 설정들
        self.reranker_config = {
            'instruction': get_env_str('RERANKER_INSTRUCTION', 'Given a web search query, retrieve relevant passages that answer the query'),
            'batch_size': get_env_int('RERANKER_BATCH_SIZE', 32),
            'enable_flash_attention': get_env_bool('ENABLE_FLASH_ATTENTION_2', True),
            'enable_prefix_caching': get_env_bool('ENABLE_PREFIX_CACHING', True),
            'allowed_tokens': get_env_list('ALLOWED_TOKENS', ['yes', 'no']),
            'logprobs_count': get_env_int('LOGPROBS_COUNT', 20)
        }
        
        # 로깅 설정
        self.logging_config = {
            'level': get_env_str('LOG_LEVEL', 'INFO'),
            'file': get_env_str('LOG_FILE', 'logs/reranker.log')
        }
        
        # API 서버 설정
        self.api_config = {
            'host': get_env_str('API_HOST', '0.0.0.0'),
            'port': get_env_int('API_PORT', 8000),
            'enabled': get_env_bool('ENABLE_API_SERVER', False)
        }

        # LangSmith 추적 설정
        self.langsmith_config = {
            'tracing': get_env_bool('LANGSMITH_TRACING', False),
            'api_key': get_env_str('LANGSMITH_API_KEY', ''),
            'project': get_env_str('LANGSMITH_PROJECT', 'VLM-Architecture-Analysis'),
            'endpoint': get_env_str('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com'),
            'enable_sessions': get_env_bool('LANGSMITH_ENABLE_SESSIONS', True)
        }
        
        logger.info(f"✅ 환경 설정 로드 완료 - 모델: {self.model_config.model_name}")
        
    def _detect_device_type(self) -> str:
        """디바이스 타입 감지"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
            
    def _get_device(self) -> torch.device:
        """디바이스 객체 반환"""
        device_type = self._detect_device_type()
        if device_type == "cuda":
            return torch.device(f"cuda:0")
        elif device_type == "mps":
            return torch.device("mps")
        else:
            return torch.device("cpu")
            
    def get_vllm_config(self) -> Dict[str, Any]:
        """vLLM 설정 반환"""
        config = {
            'model': self.model_config.model_name,
            'tensor_parallel_size': self.device_config.tensor_parallel_size,
            'gpu_memory_utilization': self.device_config.gpu_memory_utilization,
            'max_model_len': self.model_config.max_model_len,
            'dtype': self.model_config.quantization,
            'trust_remote_code': self.model_config.trust_remote_code,
        }
        
        # Reranker 특화 설정 추가
        if get_env_bool('ENABLE_PREFIX_CACHING', True):
            config['enable_prefix_caching'] = True
            
        if get_env_bool('ENABLE_CHUNKED_PREFILL', True):
            config['enable_chunked_prefill'] = True
            
        return config
        
    def get_sampling_params(self) -> Dict[str, Any]:
        """샘플링 파라미터 반환"""
        params = {
            'temperature': self.generation_config.temperature,
            'top_p': self.generation_config.top_p,
            'max_tokens': self.generation_config.max_tokens,
            'repetition_penalty': self.generation_config.repetition_penalty,
        }
        
        # Reranker 특화 설정
        if 'yes' in self.reranker_config['allowed_tokens']:
            # yes/no 토큰만 허용하는 경우
            params['logprobs'] = self.reranker_config['logprobs_count']
            
        return params
            
    def setup_logging(self):
        """로깅 설정"""
        log_level = getattr(logging, self.logging_config['level'].upper())
        log_file = self.logging_config['file']
        
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
            'logs',
            'uploads',
            'models',
            'configs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def print_config(self):
        """현재 설정 출력"""
        print("🔧 현재 환경 설정:")
        print(f"   모델: {self.model_config.model_name}")
        print(f"   디바이스: {self.device_config.device_type}")
        print(f"   GPU 메모리 사용률: {self.device_config.gpu_memory_utilization}")
        print(f"   최대 시퀀스 길이: {self.model_config.max_model_len}")
        print(f"   텐서 병렬 처리: {self.device_config.tensor_parallel_size}")
        print(f"   로그 레벨: {self.logging_config['level']}")
        print(f"   Reranker 배치 크기: {self.reranker_config['batch_size']}")
        print(f"   Flash Attention: {self.reranker_config['enable_flash_attention']}")
        print(f"   LangSmith 추적: {self.langsmith_config['tracing']}")
        if self.langsmith_config['tracing']:
            print(f"   LangSmith 프로젝트: {self.langsmith_config['project']}")
    
    def setup_langsmith(self):
        """LangSmith 환경변수 설정"""
        if self.langsmith_config['tracing']:
            import os
            os.environ['LANGSMITH_TRACING'] = str(self.langsmith_config['tracing']).lower()
            if self.langsmith_config['api_key']:
                os.environ['LANGSMITH_API_KEY'] = self.langsmith_config['api_key']
            os.environ['LANGSMITH_PROJECT'] = self.langsmith_config['project']
            os.environ['LANGSMITH_ENDPOINT'] = self.langsmith_config['endpoint']
            logger.info(f"✅ LangSmith 추적 활성화됨 - 프로젝트: {self.langsmith_config['project']}")
        else:
            logger.info("ℹ️  LangSmith 추적이 비활성화되어 있습니다.")


# 전역 환경 설정 인스턴스
env_config = EnvironmentConfig()


def get_env_config() -> EnvironmentConfig:
    """전역 환경 설정 인스턴스 반환"""
    return env_config


if __name__ == "__main__":
    # 테스트 실행
    config = EnvironmentConfig()
    config.setup_logging()
    config.create_directories()
    config.print_config()
    
    print(f"\n🎯 샘플링 파라미터:")
    sampling_params = config.get_sampling_params()
    for key, value in sampling_params.items():
        print(f"   {key}: {value}")
