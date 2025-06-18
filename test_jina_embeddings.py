#!/usr/bin/env python3
"""
jinaai/jina-embeddings-v3 모델 테스트 스크립트
환경 설정 확인 및 임베딩 생성 테스트
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """환경 변수 확인"""
    logger.info("=== 환경 변수 확인 ===")
    
    # 주요 환경 변수 확인
    env_vars = [
        "EMBEDDING_MODEL",
        "HUGGINGFACE_API_KEY",
        "EMBEDDING_DIM",
        "EMBEDDING_MAX_TOKENS",
        "EMBEDDING_TASK",
        "EMBEDDING_DEVICE"
    ]
    
    for var in env_vars:
        value = os.getenv(var, "Not Set")
        if var == "HUGGINGFACE_API_KEY" and value != "Not Set":
            # API 키는 일부만 표시
            display_value = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else value
        else:
            display_value = value
        logger.info(f"{var}: {display_value}")
    
    return True

def test_model_loading():
    """모델 로딩 테스트"""
    logger.info("\n=== 모델 로딩 테스트 ===")
    
    try:
        from src.embedding_config import get_jina_embeddings, EmbeddingConfig
        
        # 설정 객체 생성
        config = EmbeddingConfig()
        logger.info(f"모델명: {config.model_name}")
        logger.info(f"디바이스: {config.device}")
        logger.info(f"임베딩 차원: {config.embedding_dim}")
        logger.info(f"최대 토큰: {config.max_tokens}")
        logger.info(f"태스크: {config.task}")
        
        # 모델 로딩 시도
        logger.info("모델 로딩 중...")
        embeddings = get_jina_embeddings()
        logger.info("✅ 모델 로딩 성공!")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
        return None

def test_embedding_generation(embeddings):
    """임베딩 생성 테스트"""
    logger.info("\n=== 임베딩 생성 테스트 ===")
    
    if embeddings is None:
        logger.error("❌ 임베딩 모델이 로드되지 않았습니다.")
        return False
    
    try:
        # 테스트 텍스트 (건축 관련)
        test_texts = [
            "건축 도면 분석: 평면도 및 입면도",
            "주거 단지 설계 현황",
            "지하주차장 구조 및 배치도",
            "Building plan analysis and structural design"
        ]
        
        logger.info(f"테스트 텍스트 {len(test_texts)}개로 임베딩 생성...")
        
        # 임베딩 생성
        vectors = embeddings.encode(test_texts, task="retrieval.passage")
        
        logger.info(f"✅ 임베딩 생성 성공!")
        logger.info(f"벡터 개수: {len(vectors)}")
        logger.info(f"벡터 차원: {len(vectors[0]) if vectors else 0}")
        
        # 유사도 계산 테스트
        if len(vectors) >= 2:
            import numpy as np
            vec1 = np.array(vectors[0])
            vec2 = np.array(vectors[1])
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            logger.info(f"첫 번째와 두 번째 벡터 유사도: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 임베딩 생성 실패: {e}")
        return False

def test_task_specific_embeddings(embeddings):
    """태스크별 임베딩 테스트"""
    logger.info("\n=== 태스크별 임베딩 테스트 ===")
    
    if embeddings is None:
        logger.error("❌ 임베딩 모델이 로드되지 않았습니다.")
        return False
    
    try:
        test_text = "건축 도면 분석 및 설계 검토"
        
        # 다양한 태스크로 테스트
        tasks = [
            "retrieval.passage",
            "retrieval.query", 
            "text-matching",
            "classification"
        ]
        
        task_embeddings = {}
        
        for task in tasks:
            logger.info(f"태스크 '{task}' 임베딩 생성...")
            try:
                vectors = embeddings.encode([test_text], task=task)
                task_embeddings[task] = vectors[0]
                logger.info(f"✅ 태스크 '{task}' 성공 - 차원: {len(vectors[0])}")
            except Exception as e:
                logger.warning(f"⚠️  태스크 '{task}' 실패: {e}")
        
        # 태스크별 임베딩 차이 확인
        if len(task_embeddings) >= 2:
            import numpy as np
            tasks_list = list(task_embeddings.keys())
            vec1 = np.array(task_embeddings[tasks_list[0]])
            vec2 = np.array(task_embeddings[tasks_list[1]])
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            logger.info(f"태스크 '{tasks_list[0]}'와 '{tasks_list[1]}' 임베딩 유사도: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 태스크별 임베딩 테스트 실패: {e}")
        return False

def test_matryoshka_embeddings(embeddings):
    """Matryoshka 임베딩 테스트"""
    logger.info("\n=== Matryoshka 임베딩 테스트 ===")
    
    if embeddings is None:
        logger.error("❌ 임베딩 모델이 로드되지 않았습니다.")
        return False
    
    try:
        test_text = "건축 설계 및 구조 분석"
        
        # 다양한 차원으로 테스트
        dimensions = [256, 512, 768, 1024]
        
        for dim in dimensions:
            logger.info(f"차원 {dim}으로 임베딩 생성...")
            try:
                vectors = embeddings.encode([test_text], truncate_dim=dim)
                logger.info(f"✅ 차원 {dim} 성공 - 실제 차원: {len(vectors[0])}")
            except Exception as e:
                logger.warning(f"⚠️  차원 {dim} 실패: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Matryoshka 임베딩 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    logger.info("🚀 Jina Embeddings v3 테스트 시작")
    
    # 환경 변수 확인
    test_environment()
    
    # API 키 확인
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key or api_key == "your_huggingface_api_key_here":
        logger.warning("⚠️  HUGGINGFACE_API_KEY가 설정되지 않았습니다.")
        logger.warning("Hugging Face API 키를 .env 파일에 설정해주세요.")
        logger.info("하지만 로컬 모델 로딩은 시도해봅니다...")
    
    # 모델 로딩 테스트
    embeddings = test_model_loading()
    
    if embeddings:
        # 기본 임베딩 생성 테스트
        test_embedding_generation(embeddings)
        
        # 태스크별 임베딩 테스트
        test_task_specific_embeddings(embeddings)
        
        # Matryoshka 임베딩 테스트
        test_matryoshka_embeddings(embeddings)
        
        logger.info("🎉 모든 테스트 완료!")
    else:
        logger.error("❌ 모델 로딩 실패로 인해 테스트를 중단합니다.")
        
        # 문제 해결 가이드
        logger.info("\n=== 문제 해결 가이드 ===")
        logger.info("1. 다음 패키지가 설치되어 있는지 확인하세요:")
        logger.info("   pip install transformers torch einops")
        logger.info("2. .env 파일에 HUGGINGFACE_API_KEY를 설정하세요")
        logger.info("3. GPU를 사용하려면 CUDA가 설치되어 있는지 확인하세요")
        logger.info("4. 첫 실행 시에는 모델 다운로드로 시간이 소요될 수 있습니다")

if __name__ == "__main__":
    main()
