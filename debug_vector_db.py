#!/usr/bin/env python3
"""
벡터 DB 디버깅 스크립트
임베딩과 검색 과정의 상세 정보 확인
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
    """벡터 DB 디버깅"""
    logger.info("🔍 벡터 DB 디버깅 시작")
    
    try:
        from src.metadata_vector_db import MetadataVectorDB
        
        # 벡터 DB 연결
        vector_db = MetadataVectorDB(
            collection_name="test_architectural_metadata"
        )
        
        # 1. 컬렉션 상태 확인
        logger.info("📊 컬렉션 상태 확인")
        try:
            count = vector_db.collection.count()
            logger.info(f"총 벡터 수: {count}")
            
            # 모든 데이터 조회
            all_data = vector_db.collection.get(include=['metadatas', 'documents'])
            logger.info(f"실제 데이터 수: {len(all_data['ids']) if all_data['ids'] else 0}")
            
            if all_data['ids']:
                for i, doc_id in enumerate(all_data['ids']):
                    logger.info(f"문서 {i+1}: ID={doc_id}")
                    if all_data['metadatas']:
                        metadata = all_data['metadatas'][i]
                        logger.info(f"  메타데이터: {metadata.get('title', 'N/A')}")
                    if all_data['documents']:
                        doc = all_data['documents'][i]
                        logger.info(f"  문서 길이: {len(doc)} 문자")
                        logger.info(f"  문서 시작: {doc[:100]}...")
        except Exception as e:
            logger.error(f"컬렉션 상태 확인 실패: {e}")
        
        # 2. 직접 쿼리 테스트 (임계값 무시)
        logger.info("\n🔍 직접 쿼리 테스트")
        try:
            query = "도면"
            logger.info(f"검색어: '{query}'")
            
            # 쿼리 임베딩 생성
            query_embedding = vector_db.embeddings.encode([query], task="retrieval.query")[0]
            logger.info(f"쿼리 임베딩 차원: {len(query_embedding)}")
            
            # ChromaDB 직접 쿼리 (임계값 무시)
            results = vector_db.collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=['metadatas', 'documents', 'distances']
            )
            
            logger.info(f"원시 검색 결과:")
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1.0 - distance
                    logger.info(f"  {i+1}. ID: {doc_id}")
                    logger.info(f"     거리: {distance:.6f}")
                    logger.info(f"     유사도: {similarity:.6f}")
                    if results['metadatas']:
                        metadata = results['metadatas'][0][i]
                        logger.info(f"     제목: {metadata.get('title', 'N/A')}")
            else:
                logger.info("  검색 결과 없음")
                
        except Exception as e:
            logger.error(f"직접 쿼리 테스트 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 3. 임베딩 텍스트 확인
        logger.info("\n📄 임베딩 텍스트 확인")
        try:
            # 메타데이터 다시 로드
            metadata_dir = "./uploads/01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면/metadata"
            metadata_list = vector_db.load_metadata_files(metadata_dir)
            
            for i, metadata in enumerate(metadata_list):
                logger.info(f"\n메타데이터 {i+1}:")
                # 임베딩 텍스트 생성
                embedding_text = vector_db.create_embeddings_text(metadata)
                logger.info(f"  임베딩 텍스트 길이: {len(embedding_text)} 문자")
                logger.info(f"  임베딩 텍스트: {embedding_text[:200]}...")
                
        except Exception as e:
            logger.error(f"임베딩 텍스트 확인 실패: {e}")
        
    except Exception as e:
        logger.error(f"❌ 디버깅 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
