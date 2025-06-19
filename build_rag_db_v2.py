#!/usr/bin/env python3
"""
개선된 RAG 데이터베이스 구축 스크립트 v2
메타데이터 폴더 구조를 지원하고, 프로젝트별 별도 DB 생성
"""

import json
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
import sys
import re
from typing import List, Dict, Any, Optional

# .env 설정 로드
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from env_config import get_env_config
    env_config = get_env_config()
    print(f"📋 .env 기반 설정 로드됨 - LLM: {env_config.llm_provider_config.provider}, 임베딩: {env_config.embedding_config.provider}")
except ImportError:
    print("⚠️  env_config를 불러올 수 없습니다. 기본 설정을 사용합니다.")
    env_config = None

# 상수 정의
UPLOADS_ROOT_DIR = Path("uploads")
CHROMA_DB_PATH = "./chroma_db"

class RAGDatabaseBuilder:
    """RAG 데이터베이스 구축을 위한 클래스"""
    
    def __init__(self, db_path: str = CHROMA_DB_PATH):
        self.db_path = db_path
        self.client = None
        self.embedding_function = None
        self._init_client()
    
    def _init_client(self):
        """ChromaDB 클라이언트 초기화"""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 환경 설정에 따른 임베딩 함수 초기화
        if env_config and env_config.embedding_config.provider == "openai":
            print(f"🔧 OpenAI 임베딩 모델 사용: {env_config.embedding_config.openai_model}")
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=env_config.llm_provider_config.openai_api_key,
                model_name=env_config.embedding_config.openai_model
            )
        else:
            # fallback: SentenceTransformer 사용
            fallback_model = "all-MiniLM-L6-v2"
            print(f"🔧 SentenceTransformer 임베딩 모델 사용: {fallback_model}")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=fallback_model
            )
        
        print(f"✅ ChromaDB 클라이언트 초기화 완료: {self.db_path}")
    
    def _normalize_collection_name(self, project_name: str) -> str:
        """컬렉션 이름을 ChromaDB 규칙에 맞게 정규화"""
        # 한글을 영어로 변환하는 간단한 매핑
        korean_to_english = {
            "부산": "busan",
            "장안": "jangan", 
            "프로젝트": "project",
            "정보": "info",
            "도면": "drawing"
        }
        
        # 한글을 영어로 변환
        translated_name = project_name
        for korean, english in korean_to_english.items():
            translated_name = translated_name.replace(korean, english)
        
        # 기본 변환
        collection_name = f"{translated_name}".replace(" ", "_").replace("-", "_").lower()
        
        # ASCII 문자와 숫자, 언더스코어만 허용
        collection_name = "".join(c if (c.isascii() and c.isalnum()) or c == "_" else "_" for c in collection_name)
        
        # 연속된 언더스코어 제거
        collection_name = re.sub(r'_+', '_', collection_name)
        
        # 길이 제한 (3-63자)
        collection_name = collection_name[:63]
        
        # 시작과 끝이 영문/숫자인지 확인
        if not collection_name or not collection_name[0].isalnum():
            collection_name = "proj_" + collection_name.lstrip('_')
        if not collection_name[-1].isalnum():
            collection_name = collection_name.rstrip('_') + "_coll"
        
        return collection_name
    
    def check_existing_collection(self, collection_name: str) -> str:
        """기존 컬렉션 확인 및 사용자 선택"""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            
            if count > 0:
                print(f"\n⚠️  컬렉션 '{collection_name}'에 이미 {count}개의 문서가 있습니다.")
                print("옵션을 선택하세요:")
                print("1. 기존 데이터 삭제 후 새로 구축 (권장)")
                print("2. 기존 데이터에 추가")
                print("3. 취소")
                
                while True:
                    choice = input("\n선택 (1/2/3): ").strip()
                    
                    if choice == "1":
                        print(f"🗑️  기존 컬렉션 '{collection_name}' 삭제 중...")
                        self.client.delete_collection(collection_name)
                        print("✅ 기존 컬렉션이 삭제되었습니다.")
                        return "recreate"
                    
                    elif choice == "2":
                        print(f"📝 기존 컬렉션에 데이터를 추가합니다.")
                        return "append"
                    
                    elif choice == "3":
                        print("❌ 작업이 취소되었습니다.")
                        return "cancel"
                    
                    else:
                        print("올바른 번호를 선택해주세요 (1, 2, 3)")
            else:
                print(f"✅ 컬렉션 '{collection_name}'가 비어있습니다. 새로 구축합니다.")
                # 빈 컬렉션도 삭제 후 재생성
                self.client.delete_collection(collection_name)
                return "recreate"
                
        except Exception as e:
            # 컬렉션이 존재하지 않는 경우
            print(f"📝 새 컬렉션 '{collection_name}'을 생성합니다.")
            return "create"
    
    def load_metadata_from_folder(self, metadata_folder: Path) -> List[Dict[str, Any]]:
        """메타데이터 폴더에서 모든 JSON 파일 로드"""
        metadata_list = []
        metadata_files = list(metadata_folder.glob("*_metadata.json"))
        
        if not metadata_files:
            print(f"❌ 메타데이터 JSON 파일을 찾을 수 없습니다: {metadata_folder}")
            return []
        
        print(f"📄 발견된 메타데이터 파일: {len(metadata_files)}개")
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    drawing_metadata = json.load(f)
                
                # 파일명에서 도면 정보 추출
                file_stem = metadata_file.stem.replace('_metadata', '')
                drawing_metadata['source_file'] = metadata_file.name
                drawing_metadata['file_stem'] = file_stem
                
                metadata_list.append(drawing_metadata)
                
            except json.JSONDecodeError:
                print(f"    ❌ {metadata_file.name}: JSON 파싱 오류")
            except Exception as e:
                print(f"    ❌ {metadata_file.name}: 처리 오류 - {e}")
        
        return metadata_list
    
    def process_metadata_for_rag(self, metadata_list: List[Dict[str, Any]], project_name: str) -> tuple:
        """메타데이터를 RAG용으로 가공 (정보 보존)"""
        documents = []
        metadatas = []
        ids = []
        
        for i, drawing_metadata in enumerate(metadata_list):
            try:
                file_stem = drawing_metadata.get('file_stem', f'doc_{i+1}')
                
                # Self-Query 형식 또는 기존 형식 처리
                if isinstance(drawing_metadata, dict):
                    # Self-Query 형식인 경우
                    if "page_content" in drawing_metadata or "content" in drawing_metadata:
                        content = drawing_metadata.get("page_content", drawing_metadata.get("content", ""))
                        metadata = drawing_metadata.get("metadata", {})
                    else:
                        # 기존 형식인 경우
                        content = drawing_metadata.get("content", "")
                        metadata = drawing_metadata.copy()
                        # content 키 제거 (메타데이터에서)
                        metadata.pop("content", None)
                    
                    # 새로운 처리 메서드 사용 (정보 보존)
                    enhanced_content, cleaned_metadata, doc_id = self._prepare_document_for_chroma(
                        content, metadata, project_name, file_stem, i
                    )
                    
                    documents.append(enhanced_content)
                    metadatas.append(cleaned_metadata)
                    ids.append(doc_id)
                    
                    print(f"    ✅ 처리 완료: {file_stem}")
                
                else:
                    print(f"    ⚠️  {file_stem}: 올바르지 않은 메타데이터 형식")
                    
            except Exception as e:
                print(f"    ❌ 항목 {i} ({file_stem}): 처리 오류 - {e}")
                import traceback
                traceback.print_exc()
        
        return documents, metadatas, ids
    
    def build_project_rag(self, project_name: str, metadata_folder_path: Optional[Path] = None) -> bool:
        """단일 프로젝트의 RAG 데이터베이스 구축"""
        print(f"\n🏗️  프로젝트 '{project_name}' RAG DB 구축 시작...")
        
        # 메타데이터 폴더 경로 설정
        if metadata_folder_path is None:
            metadata_folder_path = UPLOADS_ROOT_DIR / project_name / "metadata"
        else:
            metadata_folder_path = Path(metadata_folder_path)
        
        print(f"📁 메타데이터 폴더: {metadata_folder_path}")
        
        if not metadata_folder_path.exists():
            print(f"❌ 메타데이터 폴더를 찾을 수 없습니다: {metadata_folder_path}")
            return False
        
        # 컬렉션 이름 정규화
        collection_name = self._normalize_collection_name(project_name)
        print(f"🗃️  컬렉션명: {collection_name}")
        
        # 기존 컬렉션 확인 및 처리
        action = self.check_existing_collection(collection_name)
        if action == "cancel":
            return False
        
        # 메타데이터 로드
        metadata_list = self.load_metadata_from_folder(metadata_folder_path)
        if not metadata_list:
            return False
        
        # RAG용 데이터 가공
        documents, metadatas, ids = self.process_metadata_for_rag(metadata_list, project_name)
        
        if not documents:
            print(f"  ❌ 처리할 수 있는 유효한 메타데이터가 없습니다.")
            return False
        
        print(f"  📄 처리된 문서: {len(documents)}개")
        
        # 컬렉션 생성 또는 가져오기
        try:
            if action in ["recreate", "create"]:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
            else:  # append
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            
            print(f"  ✅ 컬렉션 '{collection_name}' 준비 완료")
        except Exception as e:
            print(f"  ❌ 컬렉션 생성/가져오기 실패 ({collection_name}): {e}")
            return False
        
        # 문서 추가
        try:
            print(f"  📝 {len(documents)}개 문서를 데이터베이스에 추가 중...")
            
            # 배치 처리 (한 번에 너무 많이 처리하지 않도록)
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                
                print(f"    📄 {min(i+batch_size, len(documents))}/{len(documents)} 문서 처리 완료")
            
            final_count = collection.count()
            print(f"  ✅ RAG 데이터베이스 구축 완료! 총 문서 수: {final_count}")
            return True
            
        except Exception as e:
            print(f"  ❌ 문서 추가 실패: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """모든 컬렉션 목록 반환"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"❌ 컬렉션 목록 조회 실패: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 반환"""
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()
            return {"name": collection_name, "count": count}
        except Exception as e:
            print(f"❌ 컬렉션 '{collection_name}' 정보 조회 실패: {e}")
            return {}

    def _prepare_document_for_chroma(self, content: str, metadata: Dict[str, Any], project_name: str, file_stem: str, index: int = 0) -> tuple[str, Dict[str, Any], str]:
        """ChromaDB에 저장할 문서와 메타데이터를 준비 (정보 보존)"""
        
        # 메인 컨텐츠 준비
        if isinstance(content, dict):
            main_content = content.get("page_content", content.get("content", ""))
            doc_metadata = content.get("metadata", {})
        else:
            main_content = str(content) if content else ""
            doc_metadata = metadata
        
        # 풍부한 컨텐츠 생성 (검색 성능 향상)
        enhanced_content = self._create_enhanced_content(main_content, doc_metadata, project_name)
        
        # ChromaDB 호환 메타데이터 생성 (정보 보존)
        chroma_metadata = self._clean_metadata_for_chroma(doc_metadata)
        
        # 고유 ID 생성
        doc_id = self._generate_document_id(doc_metadata, project_name, file_stem, index)
        
        return enhanced_content, chroma_metadata, doc_id
    
    def _create_enhanced_content(self, main_content: str, metadata: Dict[str, Any], project_name: str) -> str:
        """검색 성능을 위한 풍부한 컨텐츠 생성 (모든 정보 활용)"""
        content_parts = []
        
        # 메인 컨텐츠
        if main_content:
            content_parts.append(main_content)
        
        # 프로젝트 정보
        content_parts.append(f"프로젝트: {project_name}")
        
        # 도면 기본 정보
        if metadata.get("drawing_title"):
            content_parts.append(f"도면명: {metadata['drawing_title']}")
        
        if metadata.get("drawing_number") and metadata["drawing_number"] != "정보 없음":
            content_parts.append(f"도면번호: {metadata['drawing_number']}")
        
        if metadata.get("drawing_type"):
            content_parts.append(f"도면유형: {metadata['drawing_type']}")
        
        if metadata.get("drawing_category"):
            content_parts.append(f"도면분류: {metadata['drawing_category']}")
        
        # 구조/건축 정보
        if metadata.get("structure_type") and metadata["structure_type"] != "정보 없음":
            content_parts.append(f"구조형식: {metadata['structure_type']}")
        
        if metadata.get("main_use") and metadata["main_use"] != "정보 없음":
            content_parts.append(f"주용도: {metadata['main_use']}")
        
        # 면적 정보
        area_info = []
        if metadata.get("building_area"):
            area_info.append(f"건축면적 {metadata['building_area']}㎡")
        if metadata.get("total_floor_area"):
            area_info.append(f"연면적 {metadata['total_floor_area']}㎡")
        if metadata.get("land_area"):
            area_info.append(f"대지면적 {metadata['land_area']}㎡")
        
        if area_info:
            content_parts.append(" ".join(area_info))
        
        # 층수 정보
        floor_info = []
        if metadata.get("floors_above"):
            floor_info.append(f"지상 {metadata['floors_above']}층")
        if metadata.get("floors_below"):
            floor_info.append(f"지하 {metadata['floors_below']}층")
        
        if floor_info:
            content_parts.append(" ".join(floor_info))
        
        # 부가 정보
        if metadata.get("parking_spaces"):
            content_parts.append(f"주차대수: {metadata['parking_spaces']}대")
        
        if metadata.get("apartment_units"):
            content_parts.append(f"세대수: {metadata['apartment_units']}세대")
        
        # 설계/시공 정보
        if metadata.get("design_firm") and metadata["design_firm"] != "정보 없음":
            content_parts.append(f"설계사: {metadata['design_firm']}")
        
        if metadata.get("construction_firm") and metadata["construction_firm"] != "정보 없음":
            content_parts.append(f"시공사: {metadata['construction_firm']}")
        
        # Legacy 데이터 활용 (상세 정보)
        if metadata.get("legacy_data"):
            legacy_data = metadata["legacy_data"]
            
            # 도면 상세 정보
            if legacy_data.get("draw_info"):
                for key, value in legacy_data["draw_info"].items():
                    if value and str(value).strip() and str(value) != "정보 없음":
                        content_parts.append(f"{key}: {value}")
            
            # 주요 키워드 (상위 30개)
            if legacy_data.get("word_counts"):
                top_words = sorted(legacy_data["word_counts"].items(), 
                                 key=lambda x: x[1], reverse=True)[:30]
                # 의미있는 키워드만 필터링
                meaningful_words = []
                for word, count in top_words:
                    if (len(word.strip()) > 1 and 
                        word.strip() not in ["", "NOTE", "LEVEL"] and
                        count > 1):
                        meaningful_words.append(word)
                
                if meaningful_words:
                    content_parts.append(f"주요 키워드: {', '.join(meaningful_words)}")
        
        return "\n".join(content_parts)
    
    def _clean_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB에 저장 가능한 형태로 메타데이터 정리 (정보 보존)"""
        cleaned = {}
        
        # 기본 필드들 (ChromaDB 호환)
        basic_fields = [
            "drawing_number", "drawing_title", "drawing_type", "drawing_category",
            "project_name", "project_address", "file_name", "page_number",
            "has_tables", "has_images", "structure_type", "main_use", 
            "design_firm", "construction_firm", "extracted_at", "extraction_method"
        ]
        
        # 숫자 필드들
        numeric_fields = [
            "land_area", "building_area", "total_floor_area", "building_height",
            "floors_above", "floors_below", "parking_spaces", "apartment_units",
            "building_coverage_ratio", "floor_area_ratio"
        ]
        
        # 기본 필드 처리
        for field in basic_fields:
            value = metadata.get(field)
            if value is not None and value != "정보 없음" and value != "":
                cleaned[field] = str(value)
        
        # 숫자 필드 처리
        for field in numeric_fields:
            value = metadata.get(field)
            if value is not None and value != 0:
                cleaned[field] = float(value) if isinstance(value, (int, float)) else value
        
        # Boolean 필드 처리
        for field in ["has_tables", "has_images"]:
            value = metadata.get(field)
            if value is not None:
                cleaned[field] = bool(value)
        
        # 리스트 필드 처리 (room_list 등)
        if metadata.get("room_list"):
            room_list = metadata["room_list"]
            if isinstance(room_list, list) and room_list:
                cleaned["room_count"] = len(room_list)
                cleaned["room_types"] = ", ".join(str(room) for room in room_list[:10])  # 상위 10개
        
        # Legacy 데이터에서 유용한 통계 정보 추출
        if metadata.get("legacy_data"):
            legacy_data = metadata["legacy_data"]
            
            # 이미지 정보
            if legacy_data.get("image_paths"):
                cleaned["image_count"] = len(legacy_data["image_paths"])
            
            # 키워드 통계
            if legacy_data.get("word_counts"):
                word_counts = legacy_data["word_counts"]
                cleaned["unique_keywords"] = len(word_counts)
                cleaned["total_keyword_frequency"] = sum(word_counts.values())
                
                # 최고 빈도 키워드들
                top_5_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                if top_5_words:
                    cleaned["top_keywords"] = ", ".join([word for word, count in top_5_words])
            
            # 도면 상세 정보
            if legacy_data.get("draw_info"):
                for key, value in legacy_data["draw_info"].items():
                    if value and str(value).strip() and str(value) != "정보 없음":
                        cleaned[f"draw_{key.lower()}"] = str(value)
        
        return cleaned
    
    def _generate_document_id(self, metadata: Dict[str, Any], project_name: str, file_stem: str, index: int) -> str:
        """고유 문서 ID 생성"""
        # 도면 번호 사용 (있는 경우)
        drawing_num = metadata.get("drawing_number", "")
        if drawing_num and drawing_num != "정보 없음":
            base_id = f"{project_name}_{drawing_num}"
        else:
            base_id = f"{project_name}_{file_stem}"
        
        # 페이지 번호 추가
        page_num = metadata.get("page_number", 1)
        if page_num > 1:
            base_id += f"_p{page_num}"
        
        # 인덱스 추가 (중복 방지)
        if index > 0:
            base_id += f"_{index}"
        
        # 안전한 ID로 변환
        safe_id = re.sub(r'[^\w\-_]', '_', base_id)
        return safe_id


def build_all_projects_rag():
    """모든 프로젝트의 RAG 데이터베이스 구축"""
    builder = RAGDatabaseBuilder()
    
    print("🔍 프로젝트 폴더 스캔 중...")
    
    if not UPLOADS_ROOT_DIR.exists():
        print(f"❌ uploads 폴더를 찾을 수 없습니다: {UPLOADS_ROOT_DIR}")
        return
    
    # 프로젝트 폴더 찾기 (metadata 하위폴더가 있는 폴더)
    project_folders = []
    for item in UPLOADS_ROOT_DIR.iterdir():
        if item.is_dir():
            metadata_folder = item / "metadata"
            if metadata_folder.exists() and any(metadata_folder.glob("*_metadata.json")):
                project_folders.append(item)
    
    if not project_folders:
        print("❌ 메타데이터 폴더가 있는 프로젝트를 찾을 수 없습니다.")
        return
    
    print(f"📁 발견된 프로젝트: {len(project_folders)}개")
    for folder in project_folders:
        print(f"  - {folder.name}")
    
    # 각 프로젝트별로 RAG DB 구축
    success_count = 0
    for project_folder in project_folders:
        project_name = project_folder.name
        metadata_folder = project_folder / "metadata"
        
        if builder.build_project_rag(project_name, metadata_folder):
            success_count += 1
        else:
            print(f"❌ 프로젝트 '{project_name}' RAG DB 구축 실패")
    
    print(f"\n🎉 RAG 데이터베이스 구축 완료!")
    print(f"✅ 성공: {success_count}/{len(project_folders)}개 프로젝트")
    
    # 최종 컬렉션 목록 출력
    collections = builder.list_collections()
    if collections:
        print(f"\n📚 생성된 컬렉션 목록:")
        for collection_name in collections:
            info = builder.get_collection_info(collection_name)
            if info:
                print(f"  - {collection_name}: {info.get('count', 0)}개 문서")


def build_specific_project_rag(project_name: str):
    """특정 프로젝트의 RAG 데이터베이스만 구축"""
    builder = RAGDatabaseBuilder()
    
    project_folder = UPLOADS_ROOT_DIR / project_name
    if not project_folder.exists():
        print(f"❌ 프로젝트 폴더를 찾을 수 없습니다: {project_folder}")
        return False
    
    metadata_folder = project_folder / "metadata"
    if not metadata_folder.exists():
        print(f"❌ 메타데이터 폴더를 찾을 수 없습니다: {metadata_folder}")
        return False
    
    return builder.build_project_rag(project_name, metadata_folder)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 데이터베이스 구축")
    parser.add_argument("--project", type=str, help="특정 프로젝트만 구축 (프로젝트 이름)")
    parser.add_argument("--list", action="store_true", help="기존 컬렉션 목록 출력")
    
    args = parser.parse_args()
    
    if args.list:
        builder = RAGDatabaseBuilder()
        collections = builder.list_collections()
        print("📚 기존 컬렉션 목록:")
        for collection_name in collections:
            info = builder.get_collection_info(collection_name)
            if info:
                print(f"  - {collection_name}: {info.get('count', 0)}개 문서")
    
    elif args.project:
        print(f"🚀 프로젝트 '{args.project}' RAG 데이터베이스 구축을 시작합니다...")
        if build_specific_project_rag(args.project):
            print("🎉 RAG 데이터베이스 구축이 완료되었습니다.")
        else:
            print("❌ RAG 데이터베이스 구축에 실패했습니다.")
    
    else:
        print("🚀 모든 프로젝트 RAG 데이터베이스 구축을 시작합니다...")
        build_all_projects_rag()
