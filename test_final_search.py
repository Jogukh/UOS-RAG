#!/usr/bin/env python3
"""
최종 완성된 벡터 DB 검색 테스트
임계값을 조정하여 실제 검색 결과 확인
"""

import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """최종 검색 테스트"""
    logger.info("🔍 최종 벡터 DB 검색 테스트")
    
    try:
        from src.metadata_vector_db import MetadataVectorDB
        
        # 벡터 DB 연결 (기존 데이터 사용)
        vector_db = MetadataVectorDB(
            collection_name="test_architectural_metadata"
        )
        
        # 통계 확인
        stats = vector_db.get_collection_stats()
        logger.info(f"📊 벡터 DB 상태: {stats['total_vectors']}개 벡터")
        
        # 임계값을 낮춰서 검색 테스트
        original_threshold = vector_db.similarity_threshold
        vector_db.similarity_threshold = 0.3  # 임계값 낮춤
        
        # 다양한 검색어로 테스트
        test_queries = [
            "주동 입면도",
            "대지 구적도", 
            "건축 도면",
            "평면도",
            "설계 도면",
            "건물 구조"
        ]
        
        logger.info(f"🔍 검색 테스트 (임계값: {vector_db.similarity_threshold})")
        
        for query in test_queries:
            logger.info(f"\n검색어: '{query}'")
            results = vector_db.search_similar_metadata(query, top_k=5)
            
            if results:
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    logger.info(f"  {i}. 유사도: {result['similarity']:.4f}")
                    logger.info(f"     제목: {metadata.get('title', 'N/A')}")
                    logger.info(f"     파일: {metadata.get('filename', 'N/A')}")
                    logger.info(f"     타입: {metadata.get('drawing_type', 'N/A')}")
            else:
                logger.info("  검색 결과 없음")
        
        # 임계값 복원
        vector_db.similarity_threshold = original_threshold
        
        logger.info(f"\n🎉 검색 테스트 완료!")
        
        # 메타데이터 상세 정보 출력
        logger.info("\n📋 저장된 메타데이터 상세:")
        all_results = vector_db.search_similar_metadata("도면", top_k=10)
        for i, result in enumerate(all_results, 1):
            metadata = result['metadata']
            logger.info(f"{i}. {metadata.get('title', 'N/A')} (파일: {metadata.get('filename', 'N/A')})")
        
    except Exception as e:
        logger.error(f"❌ 검색 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
