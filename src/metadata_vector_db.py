#!/usr/bin/env python3
"""
메타데이터 벡터 DB 임베딩 시스템
jinaai/jina-embeddings-v3 모델을 사용하여 건축 도면 메타데이터를 벡터화
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import uuid

# ChromaDB 임포트
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    print("ChromaDB가 설치되지 않았습니다. pip install chromadb로 설치해주세요.")
    HAS_CHROMADB = False

# 프로젝트 모듈 임포트
import sys
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from embedding_config import get_jina_embeddings, EmbeddingConfig
    from env_config import get_env_str, get_env_int, get_env_float
except ImportError as e:
    print(f"모듈 임포트 실패: {e}")
    print("다음 명령어로 필요한 모듈을 설치하세요:")
    print("pip install transformers torch einops chromadb")

logger = logging.getLogger(__name__)

class MetadataVectorDB:
    """메타데이터 벡터 데이터베이스 관리자"""
    
    def __init__(self, db_path: str = None, collection_name: str = None):
        """
        Args:
            db_path: 벡터 DB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.db_path = db_path or get_env_str("VECTOR_DB_PATH", "./chroma_db")
        self.collection_name = collection_name or get_env_str("VECTOR_COLLECTION_NAME", "architectural_metadata")
        self.top_k = get_env_int("VECTOR_SEARCH_TOP_K", 10)
        self.similarity_threshold = get_env_float("VECTOR_SIMILARITY_THRESHOLD", 0.7)
        
        # 임베딩 모델 초기화
        self.embeddings = None
        self.client = None
        self.collection = None
        
        self._initialize_db()
        self._initialize_embeddings()
    
    def _initialize_db(self):
        """ChromaDB 초기화"""
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB가 설치되지 않았습니다.")
        
        try:
            # ChromaDB 클라이언트 생성
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 컬렉션 생성 또는 가져오기
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "건축 도면 메타데이터 벡터 저장소"}
            )
            
            logger.info(f"✅ ChromaDB 초기화 완료 - 컬렉션: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")
            raise
    
    def _initialize_embeddings(self):
        """임베딩 모델 초기화"""
        try:
            self.embeddings = get_jina_embeddings()
            logger.info("✅ Jina 임베딩 모델 초기화 완료")
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            raise
    
    def load_metadata_files(self, metadata_dir: str) -> List[Dict[str, Any]]:
        """메타데이터 JSON 파일들 로드"""
        metadata_path = Path(metadata_dir)
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"메타데이터 디렉토리를 찾을 수 없습니다: {metadata_dir}")
        
        metadata_list = []
        
        # 다양한 메타데이터 파일 패턴 지원
        patterns = [
            "*_metadata.json",
            "*_metadata_*.json",  # backup 파일 등
            "metadata_*.json",
            "*.json"  # 일반 JSON 파일도 시도
        ]
        
        json_files = []
        for pattern in patterns:
            matches = list(metadata_path.glob(pattern))
            for match in matches:
                if match not in json_files:  # 중복 제거
                    json_files.append(match)
        
        logger.info(f"📁 메타데이터 파일 로드 중: {len(json_files)}개 파일")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 유효한 메타데이터인지 확인 (기본 필드 존재 여부)
                if not isinstance(metadata, dict):
                    logger.warning(f"⚠️  유효하지 않은 메타데이터 형식: {json_file.name}")
                    continue
                
                # 파일 정보 추가
                metadata['_file_info'] = {
                    'filename': json_file.name,
                    'filepath': str(json_file),
                    'file_id': json_file.stem.replace('_metadata', '').replace('_backup', ''),
                    'load_timestamp': datetime.now().isoformat()
                }
                
                metadata_list.append(metadata)
                logger.debug(f"✅ 로드됨: {json_file.name}")
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON 파싱 오류 {json_file.name}: {e}")
            except Exception as e:
                logger.error(f"❌ 파일 로드 오류 {json_file.name}: {e}")
        
        logger.info(f"✅ {len(metadata_list)}개 메타데이터 파일 로드 완료")
        return metadata_list
    
    def create_embeddings_text(self, metadata: Dict[str, Any]) -> str:
        """메타데이터를 임베딩용 텍스트로 변환"""
        text_parts = []
        
        # 프로젝트 정보
        project_info = metadata.get('project_info', {})
        if project_info:
            text_parts.append(f"프로젝트: {project_info.get('project_name', '')}")
            text_parts.append(f"도면유형: {project_info.get('drawing_type', '')}")
            text_parts.append(f"분야: {project_info.get('discipline', '')}")
            text_parts.append(f"목적: {project_info.get('drawing_purpose', '')}")
        
        # 도면 메타데이터
        drawing_metadata = metadata.get('drawing_metadata', {})
        if drawing_metadata:
            text_parts.append(f"제목: {drawing_metadata.get('title', '')}")
            text_parts.append(f"설명: {drawing_metadata.get('description', '')}")
            
            keywords = drawing_metadata.get('keywords', [])
            if keywords:
                text_parts.append(f"키워드: {', '.join(keywords)}")
                
            text_parts.append(f"건물유형: {drawing_metadata.get('building_type', '')}")
            text_parts.append(f"복잡도: {drawing_metadata.get('complexity_level', '')}")
        
        # 건축적 특징
        arch_features = metadata.get('architectural_features', {})
        if arch_features:
            spatial_org = arch_features.get('spatial_organization', {})
            if spatial_org:
                text_parts.append(f"공간구성: {spatial_org.get('description', '')}")
                
                spaces = spatial_org.get('identified_spaces', [])
                if spaces:
                    text_parts.append(f"식별공간: {', '.join(spaces)}")
            
            design_elements = arch_features.get('design_elements', {})
            if design_elements:
                text_parts.append(f"설계요소: {design_elements.get('description', '')}")
        
        # 기술적 사양
        tech_specs = metadata.get('technical_specifications', {})
        if tech_specs:
            text_parts.append(f"파일형식: {tech_specs.get('file_format', '')}")
            text_parts.append(f"단위: {tech_specs.get('units', '')}")
            text_parts.append(f"축척: {tech_specs.get('scale', '')}")
        
        # 파일 정보
        file_info = metadata.get('_file_info', {})
        if file_info:
            text_parts.append(f"파일ID: {file_info.get('file_id', '')}")
        
        return " | ".join([part for part in text_parts if part.strip()])
    
    def embed_metadata(self, metadata_list: List[Dict[str, Any]]) -> bool:
        """메타데이터 리스트를 벡터 DB에 임베딩"""
        if not metadata_list:
            logger.warning("임베딩할 메타데이터가 없습니다.")
            return False
        
        try:
            logger.info(f"🔄 {len(metadata_list)}개 메타데이터 임베딩 시작")
            
            # 임베딩용 텍스트 생성
            texts = []
            ids = []
            metadatas = []
            
            for metadata in metadata_list:
                # 임베딩용 텍스트 생성
                embedding_text = self.create_embeddings_text(metadata)
                texts.append(embedding_text)
                
                # 고유 ID 생성
                file_info = metadata.get('_file_info', {})
                doc_id = file_info.get('file_id', str(uuid.uuid4()))
                ids.append(doc_id)
                
                # 메타데이터 (검색용)
                search_metadata = {
                    'project_name': metadata.get('project_info', {}).get('project_name', ''),
                    'drawing_type': metadata.get('project_info', {}).get('drawing_type', ''),
                    'title': metadata.get('drawing_metadata', {}).get('title', ''),
                    'discipline': metadata.get('project_info', {}).get('discipline', ''),
                    'filename': file_info.get('filename', ''),
                    'file_id': file_info.get('file_id', ''),
                    'embedding_timestamp': datetime.now().isoformat()
                }
                metadatas.append(search_metadata)
            
            # 텍스트 임베딩 생성
            logger.info("📊 텍스트 임베딩 생성 중...")
            embeddings = self.embeddings.encode(texts, task="retrieval.passage")
            
            # ChromaDB에 저장
            logger.info("💾 ChromaDB에 저장 중...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            logger.info(f"✅ {len(metadata_list)}개 메타데이터 임베딩 완료")
            return True
            
        except Exception as e:
            logger.error(f"메타데이터 임베딩 실패: {e}")
            return False
    
    def embed_and_store_metadata(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, int]:
        """메타데이터를 임베딩하고 벡터 DB에 저장 (결과 반환)"""
        results = {
            'success_count': 0,
            'error_count': 0,
            'total_count': len(metadata_list)
        }
        
        if not metadata_list:
            logger.warning("임베딩할 메타데이터가 없습니다.")
            return results
        
        try:
            success = self.embed_metadata(metadata_list)
            if success:
                results['success_count'] = len(metadata_list)
            else:
                results['error_count'] = len(metadata_list)
        except Exception as e:
            logger.error(f"메타데이터 임베딩 및 저장 실패: {e}")
            results['error_count'] = len(metadata_list)
        
        return results
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """벡터 검색"""
        top_k = top_k or self.top_k
        
        try:
            # 쿼리 임베딩
            query_embedding = self.embeddings.encode([query], task="retrieval.query")[0]
            
            # 벡터 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            # 결과 포맷팅
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    
                    # ChromaDB는 기본적으로 squared L2 distance를 사용
                    # 코사인 유사도 계산: similarity = 1 - (distance / 2)
                    # 또는 단순히 거리의 역수 사용
                    if distance > 0:
                        # 거리가 클수록 유사도는 낮아짐
                        score = max(0, 1.0 / (1.0 + distance))
                    else:
                        score = 1.0
                    
                    result = {
                        'id': doc_id,
                        'score': score,
                        'distance': distance,  # 디버깅용 추가
                        'metadata': results['metadatas'][0][i],
                        'content': results['documents'][0][i]
                    }
                    
                    # 임계값 필터링 (임계값을 낮춰서 적용)
                    if result['score'] >= max(0.1, self.similarity_threshold * 0.5):  # 임계값의 절반으로 완화
                        search_results.append(result)
            
            logger.info(f"🔍 검색 완료: {len(search_results)}개 결과 (임계값: {self.similarity_threshold})")
            return search_results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    def search_similar_metadata(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """유사한 메타데이터 검색 (별칭 메서드)"""
        results = self.search(query, top_k)
        
        # 결과 형식을 맞춤
        formatted_results = []
        for result in results:
            formatted_result = {
                'similarity': result['score'],
                'metadata': result['metadata'],
                'content': result['content'],
                'id': result['id']
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 정보 (업데이트)"""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'total_vectors': count,  # 'document_count' 대신 'total_vectors' 사용
                'db_path': self.db_path,
                'embedding_model': self.embeddings.config.model_name if self.embeddings else 'Unknown'
            }
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {e}")
            return {
                'collection_name': self.collection_name,
                'total_vectors': 0,
                'db_path': self.db_path,
                'embedding_model': 'Unknown'
            }
    
    def clear_collection(self):
        """컬렉션 데이터 초기화"""
        try:
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "건축 도면 메타데이터 벡터 저장소"}
            )
            logger.info(f"✅ 컬렉션 '{self.collection_name}' 초기화 완료")
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {e}")

def create_metadata_vector_db(metadata_dir: str, force_recreate: bool = False) -> MetadataVectorDB:
    """메타데이터 벡터 DB 생성"""
    try:
        # 벡터 DB 초기화
        vector_db = MetadataVectorDB()
        
        # 기존 데이터 초기화 (옵션)
        if force_recreate:
            logger.info("🗑️  기존 벡터 DB 데이터 초기화")
            vector_db.clear_collection()
        
        # 메타데이터 로드
        metadata_list = vector_db.load_metadata_files(metadata_dir)
        
        if not metadata_list:
            logger.warning("임베딩할 메타데이터가 없습니다.")
            return vector_db
        
        # 임베딩 및 저장
        success = vector_db.embed_metadata(metadata_list)
        
        if success:
            stats = vector_db.get_collection_stats()
            logger.info(f"📊 벡터 DB 생성 완료: {stats}")
        
        return vector_db
        
    except Exception as e:
        logger.error(f"메타데이터 벡터 DB 생성 실패: {e}")
        raise

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 메타데이터 디렉토리
    metadata_dir = "uploads/01_행복도시 6-3생활권M3BL 실시설계도면2차 건축도면/metadata"
    
    try:
        # 벡터 DB 생성
        vector_db = create_metadata_vector_db(metadata_dir, force_recreate=True)
        
        # 테스트 검색
        test_queries = [
            "지하주차장 도면",
            "주동 입면도",
            "면적 계산",
            "창호 설계"
        ]
        
        print(f"\n🔍 테스트 검색:")
        for query in test_queries:
            results = vector_db.search(query, top_k=3)
            print(f"\n검색어: '{query}'")
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result['metadata']['title']} (유사도: {result['score']:.3f})")
            else:
                print("  검색 결과 없음")
        
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
