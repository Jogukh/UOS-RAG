import json
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
import sys

# .env 설정 로드
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from env_config import get_env_config
    env_config = get_env_config()
    print(f"📋 .env 기반 설정 로드됨 - 모델: {env_config.model_config.model_name}")
except ImportError:
    print("⚠️  env_config를 불러올 수 없습니다. 기본 설정을 사용합니다.")
    env_config = None

# 상수 정의 (.env에서 가져오거나 기본값 사용)
UPLOADS_ROOT_DIR = Path("uploads")
METADATA_BASE_FILENAME = "project_metadata"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def build_rag_database_for_project(project_name, project_metadata_file_path):
    """
    단일 프로젝트의 메타데이터 파일을 읽어 해당 프로젝트용 ChromaDB 컬렉션을 구축합니다.
    """
    # ChromaDB 클라이언트 초기화 (영구 저장)
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Sentence Transformer 임베딩 함수 설정
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # 프로젝트별 컬렉션 이름 생성 (특수문자 제거/정규화)
    collection_name = f"drawings_{project_name}".replace(" ", "_").replace("-", "_").lower()
    # ChromaDB 컬렉션 이름 규칙에 맞게 ASCII 문자만 허용
    collection_name = "".join(c if (c.isascii() and c.isalnum()) or c == "_" else "_" for c in collection_name)
    # 연속된 _ 제거 및 길이 제한
    collection_name = "_".join(filter(None, collection_name.split("_")))[:63]  # 3-512 문자 제한
    
    # 시작과 끝이 영문/숫자인지 확인
    if not collection_name[0].isalnum():
        collection_name = "drawing_" + collection_name
    if not collection_name[-1].isalnum():
        collection_name = collection_name.rstrip('_') + "_db"
    
    print(f"  컬렉션 이름: {collection_name}")

    # 컬렉션 생성 또는 가져오기
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"} # 코사인 유사도 사용
        )
    except Exception as e:
        print(f"  오류: 컬렉션 생성/가져오기 실패 ({collection_name}): {e}")
        return False

    # 프로젝트 메타데이터 파일 로드
    try:
        with open(project_metadata_file_path, 'r', encoding='utf-8') as f:
            project_metadata = json.load(f)
    except FileNotFoundError:
        print(f"  오류: 메타데이터 파일을 찾을 수 없습니다 ({project_metadata_file_path})")
        return False
    except json.JSONDecodeError:
        print(f"  오류: 메타데이터 파일이 유효한 JSON 형식이 아닙니다 ({project_metadata_file_path})")
        return False

    # extract_metadata.py에서 생성된 구조에 맞게 데이터 접근
    drawings_list = project_metadata.get("drawings", [])
    project_info = project_metadata.get("project_info", {})
    
    if not drawings_list:
        print(f"  경고: 프로젝트 '{project_name}'에 도면 정보가 없습니다.")
        return False

    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    doc_count = 0

    print(f"  총 {len(drawings_list)}개의 도면 처리 중...")

    for i, drawing_info in enumerate(drawings_list):
        if not isinstance(drawing_info, dict):
            print(f"    경고: 도면 #{i}의 형식이 올바르지 않습니다. 건너뜁니다.")
            continue

        # 고유 ID 생성 (도면 번호 또는 파일명_페이지번호 조합)
        drawing_number = drawing_info.get("drawing_number", "근거부족")
        file_name = drawing_info.get("file_name", "unknown_file")
        page_number = drawing_info.get("page_number", i+1)
        
        if drawing_number != "근거 부족" and drawing_number != "근거부족":
            unique_id = f"{project_name}_{drawing_number}_{page_number}"
        else:
            unique_id = f"{project_name}_{file_name}_p{page_number}"

        # 문서(텍스트 청크) 생성 - 추출된 메타데이터 기반
        text_chunk_parts = [
            f"프로젝트: {project_name}",
            f"파일명: {file_name}",
            f"페이지: {page_number}",
            f"도면번호: {drawing_number}",
            f"도면제목: {drawing_info.get('drawing_title', '정보 없음')}",
            f"도면유형: {drawing_info.get('drawing_type', '정보 없음')}",
            f"축척: {drawing_info.get('scale', '정보 없음')}"
        ]
        
        # 면적 정보 추가
        area_info = drawing_info.get("area_info", {})
        if area_info:
            area_parts = []
            for area_type, area_value in area_info.items():
                area_parts.append(f"{area_type}: {area_value}")
            if area_parts:
                text_chunk_parts.append(f"면적정보: {', '.join(area_parts)}")

        # 공간 목록 추가
        room_list = drawing_info.get("room_list", [])
        if room_list:
            text_chunk_parts.append(f"주요공간: {', '.join(room_list)}")

        # 층 정보 추가
        level_info = drawing_info.get("level_info", [])
        if level_info:
            text_chunk_parts.append(f"층정보: {', '.join(level_info)}")

        # 치수 정보 추가 (일부만)
        dimensions = drawing_info.get("dimensions", [])
        if dimensions:
            dim_preview = dimensions[:5]  # 처음 5개만
            text_chunk_parts.append(f"주요치수: {', '.join(dim_preview)}")

        # 텍스트 스니펫 추가 (검색 품질 향상)
        raw_text_snippet = drawing_info.get("raw_text_snippet", "")
        if raw_text_snippet:
            text_chunk_parts.append(f"텍스트내용: {raw_text_snippet}")

        # 최종 문서 텍스트 생성
        document_text = ". ".join(filter(None, text_chunk_parts))
        
        # 메타데이터 구성 (ChromaDB 저장용)
        metadata = {
            "project_name": project_name,
            "file_name": file_name,
            "page_number": str(page_number),
            "drawing_number": drawing_number,
            "drawing_title": drawing_info.get("drawing_title", ""),
            "drawing_type": drawing_info.get("drawing_type", ""),
            "scale": drawing_info.get("scale", ""),
            "area_info_json": json.dumps(area_info, ensure_ascii=False), # JSON 문자열로 저장
            "room_list_str": ",".join(room_list), # 리스트를 문자열로
            "level_info_str": ",".join(level_info), # 리스트를 문자열로
            "full_path": drawing_info.get("full_path", ""),
            "extracted_at": drawing_info.get("extracted_at", "")
        }

        documents_to_add.append(document_text)
        metadatas_to_add.append(metadata)
        ids_to_add.append(unique_id)
        doc_count += 1
        
        # 메모리 관리를 위해 100개 단위로 DB에 추가
        if len(documents_to_add) >= 100:
            print(f"    {len(documents_to_add)}개 문서를 컬렉션에 추가 중...")
            try:
                collection.add(
                    documents=documents_to_add,
                    metadatas=metadatas_to_add,
                    ids=ids_to_add
                )
                documents_to_add, metadatas_to_add, ids_to_add = [], [], []
            except Exception as e:
                print(f"    오류: 문서 추가 실패: {e}")
                return False

    # 남은 문서 추가
    if documents_to_add:
        print(f"    남은 {len(documents_to_add)}개 문서를 컬렉션에 추가 중...")
        try:
            collection.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
        except Exception as e:
            print(f"    오류: 남은 문서 추가 실패: {e}")
            return False

    print(f"  ✅ 프로젝트 '{project_name}': {doc_count}개 도면이 RAG DB에 추가됨")
    print(f"     컬렉션 '{collection_name}' 총 문서 수: {collection.count()}")
    return True

def build_all_projects_rag():
    """
    uploads 폴더의 모든 프로젝트에 대해 RAG 데이터베이스를 구축합니다.
    """
    if not UPLOADS_ROOT_DIR.exists():
        print(f"오류: uploads 폴더({UPLOADS_ROOT_DIR})를 찾을 수 없습니다.")
        return

    processed_projects = 0
    failed_projects = 0

    # uploads 폴더 및 하위 프로젝트 폴더 순회
    for project_dir_item in UPLOADS_ROOT_DIR.iterdir():
        if project_dir_item.is_dir():
            project_name = project_dir_item.name
            
            # 메타데이터 파일 경로 구성 (extract_metadata.py 저장 규칙과 일치)
            safe_project_name_for_file = "".join(c if c.isalnum() else "_" for c in project_name)
            metadata_filename = f"{METADATA_BASE_FILENAME}_{safe_project_name_for_file}.json"
            project_metadata_file_path = project_dir_item / metadata_filename
            
            print(f"\n🏗️  프로젝트 '{project_name}' RAG DB 구축 중...")
            print(f"  메타데이터 파일: {project_metadata_file_path}")

            if not project_metadata_file_path.exists():
                # _default_project의 경우 uploads 폴더 바로 아래에 있을 수 있음
                if project_name == "_default_project":
                    metadata_filename_default = f"{METADATA_BASE_FILENAME}__default_project.json"
                    project_metadata_file_path = UPLOADS_ROOT_DIR / metadata_filename_default
                    if not project_metadata_file_path.exists():
                        print(f"  ⚠️  메타데이터 파일을 찾을 수 없습니다. 건너뜁니다.")
                        failed_projects += 1
                        continue
                else:
                    print(f"  ⚠️  메타데이터 파일을 찾을 수 없습니다. 건너뜁니다.")
                    failed_projects += 1
                    continue
            
            # 프로젝트별 RAG DB 구축
            success = build_rag_database_for_project(project_name, project_metadata_file_path)
            if success:
                processed_projects += 1
            else:
                failed_projects += 1

    print(f"\n📊 RAG DB 구축 완료:")
    print(f"  ✅ 성공한 프로젝트: {processed_projects}개")
    print(f"  ❌ 실패한 프로젝트: {failed_projects}개")
    print(f"  📁 ChromaDB 저장 위치: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    print("🚀 프로젝트별 RAG 데이터베이스 구축을 시작합니다...")
    build_all_projects_rag()
    print("🎉 RAG 데이터베이스 구축이 완료되었습니다.")
