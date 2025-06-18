#!/usr/bin/env python3
"""
메타데이터 벡터 DB 임베딩 테스트 스크립트
실제 DWG 메타데이터 파일들을 Jina 임베딩으로 벡터화하여 ChromaDB에 저장
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

def find_metadata_directory():
    """메타데이터 디렉토리 찾기"""
    # 가능한 메타데이터 디렉토리 경로들
    possible_paths = [
        "./metadata",
        "./uploads/01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면/metadata",
        "./uploads/*/metadata"
    ]
    
    for path_pattern in possible_paths:
        if "*" in path_pattern:
            # 와일드카드 패턴 처리
            import glob
            matches = glob.glob(path_pattern)
            if matches:
                for match in matches:
                    if os.path.exists(match) and os.path.isdir(match):
                        json_files = [f for f in os.listdir(match) if f.endswith('.json')]
                        if json_files:
                            return match
        else:
            if os.path.exists(path_pattern) and os.path.isdir(path_pattern):
                json_files = [f for f in os.listdir(path_pattern) if f.endswith('.json')]
                if json_files:
                    return path_pattern
    
    return None

def test_vector_db_embedding():
    """벡터 DB 임베딩 테스트"""
    logger.info("🚀 메타데이터 벡터 DB 임베딩 테스트 시작")
    
    # 메타데이터 디렉토리 찾기
    metadata_dir = find_metadata_directory()
    if not metadata_dir:
        logger.error("❌ 메타데이터 디렉토리를 찾을 수 없습니다.")
        logger.info("다음 경로들을 확인했습니다:")
        logger.info("  - ./metadata")
        logger.info("  - ./uploads/*/metadata")
        return False
    
    logger.info(f"✅ 메타데이터 디렉토리 발견: {metadata_dir}")
    
    # JSON 파일 개수 확인
    json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
    logger.info(f"📋 JSON 메타데이터 파일: {len(json_files)}개")
    
    try:
        # MetadataVectorDB 인스턴스 생성
        from src.metadata_vector_db import MetadataVectorDB
        
        logger.info("🔧 벡터 DB 초기화 중...")
        vector_db = MetadataVectorDB(
            db_path="./chroma_db",
            collection_name="test_architectural_metadata"
        )
        
        logger.info("📊 메타데이터 파일 로드 중...")
        metadata_files = vector_db.load_metadata_files(metadata_dir)
        logger.info(f"✅ 로드된 메타데이터: {len(metadata_files)}개")
        
        # 첫 번째 메타데이터 샘플 출력
        if metadata_files:
            first_metadata = metadata_files[0]
            logger.info(f"📄 샘플 메타데이터 키: {list(first_metadata.keys())}")
            if 'file_name' in first_metadata:
                logger.info(f"📋 파일명: {first_metadata['file_name']}")
        
        logger.info("🔄 메타데이터 임베딩 및 저장 중...")
        embedding_results = vector_db.embed_and_store_metadata(metadata_files)
        
        logger.info(f"✅ 임베딩 완료!")
        logger.info(f"  - 성공: {embedding_results['success_count']}개")
        logger.info(f"  - 실패: {embedding_results['error_count']}개")
        
        # 벡터 DB 통계 확인
        logger.info("📈 벡터 DB 통계 확인...")
        stats = vector_db.get_collection_stats()
        logger.info(f"  - 총 벡터 수: {stats['total_vectors']}")
        logger.info(f"  - 컬렉션명: {stats['collection_name']}")
        
        # 샘플 검색 테스트
        logger.info("🔍 샘플 검색 테스트...")
        search_query = "건축 도면 평면도"
        search_results = vector_db.search_similar_metadata(search_query, top_k=3)
        
        logger.info(f"검색 결과: {len(search_results)}개")
        for i, result in enumerate(search_results[:2]):  # 처음 2개만 표시
            logger.info(f"  {i+1}. 유사도: {result['similarity']:.4f}")
            logger.info(f"     파일명: {result['metadata'].get('file_name', 'N/A')}")
        
        logger.info("🎉 벡터 DB 임베딩 테스트 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 벡터 DB 임베딩 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """메인 함수"""
    success = test_vector_db_embedding()
    
    if success:
        logger.info("\n🎊 모든 테스트가 성공적으로 완료되었습니다!")
        logger.info("이제 다음과 같은 기능을 사용할 수 있습니다:")
        logger.info("  1. DWG → DXF 변환")
        logger.info("  2. 메타데이터 추출 및 LLM 분석")
        logger.info("  3. Jina 임베딩으로 벡터화")
        logger.info("  4. ChromaDB에 저장 및 검색")
    else:
        logger.error("\n❌ 테스트 실패!")
        logger.info("문제 해결을 위해 다음을 확인해주세요:")
        logger.info("  1. 메타데이터 JSON 파일이 존재하는지")
        logger.info("  2. ChromaDB가 정상 설치되었는지")
        logger.info("  3. Jina 임베딩 모델이 정상 로드되는지")

if __name__ == "__main__":
    main()
