#!/usr/bin/env python3
"""
임베딩 모델 설정 및 관리
Hugging Face jinaai/jina-embeddings-v3 모델 사용 (최신 버전 대응)
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import torch

# 환경 변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """임베딩 모델 설정"""
    model_name: str = "jinaai/jina-embeddings-v3"
    api_key: Optional[str] = None
    embedding_dim: int = 1024  # Matryoshka 지원: 32, 64, 128, 256, 512, 768, 1024
    max_tokens: int = 8192  # RoPE로 최대 8192 토큰 지원
    batch_size: int = 8
    device: str = "cuda"
    precision: str = "float16"
    task: str = "retrieval.passage"  # task LoRA 어댑터 사용
    trust_remote_code: bool = True  # jinaai/jina-embeddings-v3는 필수
    truncate_dim: Optional[int] = None  # Matryoshka 임베딩 축소 차원
    
    def __post_init__(self):
        """환경 변수에서 설정 로드"""
        self.model_name = os.getenv("EMBEDDING_MODEL", self.model_name)
        self.api_key = os.getenv("HUGGINGFACE_API_KEY", self.api_key)
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", str(self.embedding_dim)))
        self.max_tokens = int(os.getenv("EMBEDDING_MAX_TOKENS", str(self.max_tokens)))
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(self.batch_size)))
        self.device = os.getenv("EMBEDDING_DEVICE", self.device)
        self.precision = os.getenv("EMBEDDING_PRECISION", self.precision)
        self.task = os.getenv("EMBEDDING_TASK", self.task)
        
        # CUDA 사용 가능 여부 확인
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA가 사용 불가능하여 CPU로 변경합니다.")
            self.device = "cpu"

class JinaEmbeddings:
    """Jina AI v3 임베딩 모델 래퍼 (최신 API 대응)"""
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Args:
            config: 임베딩 설정
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
    def _initialize_model(self):
        """모델 초기화 (최신 API 대응)"""
        try:
            # Hugging Face Transformers 사용
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            logger.info(f"Jina 임베딩 모델 로드 중: {self.config.model_name}")
            
            # 디바이스 설정
            if self.config.device == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("CUDA 디바이스 사용")
            else:
                self.device = torch.device("cpu")
                logger.info("CPU 디바이스 사용")
            
            # 모델 로드 (trust_remote_code=True 필수)
            if self.config.precision == "float16":
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                ).to(self.device)
            else:
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                ).to(self.device)
            
            # 모델 평가 모드로 설정
            self.model.eval()
            
            logger.info(f"✅ Jina 임베딩 모델 초기화 완료 - 디바이스: {self.device}")
            
        except ImportError as e:
            logger.error(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
            logger.error("다음 명령어로 설치하세요: pip install transformers torch einops")
            raise
        except Exception as e:
            logger.error(f"Jina 임베딩 모델 초기화 실패: {e}")
            raise
    
    def encode(self, texts: List[str], task: str = None, max_length: int = None, 
               truncate_dim: int = None) -> List[List[float]]:
        """
        텍스트를 임베딩 벡터로 변환 (최신 API 대응)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            task: 태스크 타입 (retrieval.passage, retrieval.query, text-matching 등)
            max_length: 최대 토큰 길이 (기본값: 8192)
            truncate_dim: Matryoshka 임베딩 축소 차원 (선택사항)
            
        Returns:
            임베딩 벡터 리스트
        """
        if self.model is None:
            raise RuntimeError("모델이 초기화되지 않았습니다.")
        
        task = task or self.config.task
        max_length = max_length or self.config.max_tokens
        
        try:
            import torch
            
            logger.info(f"임베딩 생성 중 - 텍스트 {len(texts)}개, 태스크: {task}")
            
            # Jina v3의 encode 메서드 직접 사용 (권장 방식)
            embeddings = self.model.encode(
                texts,
                task=task,
                max_length=max_length,
                truncate_dim=truncate_dim or self.config.truncate_dim
            )
            
            # numpy 배열을 리스트로 변환
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            
            logger.info(f"✅ 임베딩 생성 완료 - 차원: {len(embeddings[0]) if embeddings else 0}")
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            # 폴백: 수동 처리
            return self._encode_manual(texts, task, max_length, truncate_dim)
    
    def _encode_manual(self, texts: List[str], task: str = None, max_length: int = None, 
                      truncate_dim: int = None) -> List[List[float]]:
        """수동 임베딩 처리 (폴백 메서드)"""
        try:
            import torch
            from transformers import AutoTokenizer
            
            # 토크나이저가 없다면 로드
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True
                )
            
            embeddings = []
            
            # 배치 단위로 처리
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                # 토크나이징
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=max_length or self.config.max_tokens,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # 평균 풀링 (Jina v3 권장 방식)
                    batch_embeddings = self._mean_pooling(
                        outputs.last_hidden_state, 
                        inputs['attention_mask']
                    )
                    
                    # 정규화 (Matryoshka 임베딩 고려)
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    # Matryoshka 임베딩 축소
                    if truncate_dim and truncate_dim < batch_embeddings.shape[1]:
                        batch_embeddings = batch_embeddings[:, :truncate_dim]
                        # 축소 후 재정규화
                        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    embeddings.extend(batch_embeddings.cpu().tolist())
            
            logger.info(f"✅ {len(texts)}개 텍스트 임베딩 완료 (수동 처리)")
            return embeddings
            
        except Exception as e:
            logger.error(f"수동 임베딩 처리 실패: {e}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """평균 풀링 (Jina v3 권장 방식)"""
        import torch
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_single(self, text: str, task: str = None) -> List[float]:
        """단일 텍스트 임베딩"""
        return self.encode([text], task)[0]

def get_embedding_config() -> EmbeddingConfig:
    """환경 설정에서 임베딩 설정 로드"""
    return EmbeddingConfig()

def get_jina_embeddings() -> JinaEmbeddings:
    """Jina 임베딩 인스턴스 생성"""
    config = get_embedding_config()
    return JinaEmbeddings(config)

# 전역 인스턴스 (싱글톤 패턴)
_jina_embeddings = None

def get_shared_embeddings() -> JinaEmbeddings:
    """공유 Jina 임베딩 인스턴스 반환"""
    global _jina_embeddings
    if _jina_embeddings is None:
        _jina_embeddings = get_jina_embeddings()
    return _jina_embeddings
