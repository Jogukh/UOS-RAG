#!/usr/bin/env python3
"""
Self-Query 호환 RAG 데이터베이스 구축 도구
기존 메타데이터를 Self-Query 형식으로 변환하여 ChromaDB에 저장
"""

import json
import chromadb
from chromadb.utils import embedding_functions
import os
from pathlib import Path
import sys
from datetime import datetime

# 프로젝트 모듈 추가
sys.path.append(str(Path(__file__).parent / "src"))
from convert_to_self_query import convert_to_self_query_format

# 설정
UPLOADS_ROOT_DIR = Path("uploads")
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def create_collection_name(project_name):
    """프로젝트명을 ChromaDB 호환 컬렉션명으로 변환"""
    # 영문자, 숫자, 언더스코어만 허용
    clean_name = "".join(c if (c.isascii() and c.isalnum()) or c == "_" else "_" for c in project_name.lower())
    # 연속 언더스코어 제거
    clean_name = "_".join(filter(None, clean_name.split("_")))
    # drawings_ 접두사 추가
    collection_name = f"drawings_{clean_name}"
    # 길이 제한 (63자)
    return collection_name[:63]

def build_self_query_rag_database(project_metadata_file):
    """Self-Query 호환 RAG 데이터베이스 구축"""
    
    # 메타데이터 파일 확인
    if not Path(project_metadata_file).exists():
        print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {project_metadata_file}")
        return False
    
    # 메타데이터 로드
    try:
        with open(project_metadata_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)
    except Exception as e:
        print(f"❌ 메타데이터 파일 로드 실패: {e}")
        return False
    
    project_name = project_data.get('project_name', 'unknown')
    drawings = project_data.get('drawings', [])
    
    if not drawings:
        print(f"❌ 도면 데이터가 없습니다")
        return False
    
    print(f"🏗️  프로젝트 '{project_name}' RAG DB 구축 시작...")
    print(f"📄 도면 수: {len(drawings)}개")
    
    # ChromaDB 초기화
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # 임베딩 함수 설정
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # 컬렉션 생성
    collection_name = create_collection_name(project_name)
    print(f"📦 컬렉션명: {collection_name}")
    
    try:
        # 기존 컬렉션 삭제 후 재생성
        try:
            client.delete_collection(collection_name)
            print(f"🗑️  기존 컬렉션 삭제됨")
        except:
            pass
        
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"description": f"{project_name} 프로젝트 Self-Query 호환 도면 데이터"}
        )
        
    except Exception as e:
        print(f"❌ 컬렉션 생성 실패: {e}")
        return False
    
    # 도면 데이터를 Self-Query 형식으로 변환 및 추가
    documents = []
    metadatas = []
    ids = []
    
    for i, drawing in enumerate(drawings):
        try:
            # Self-Query 형식으로 변환
            converted = convert_to_self_query_format(drawing)
            
            # 데이터 추가
            documents.append(converted['content'])
            metadatas.append(converted['metadata'])
            
            # ID 생성 (프로젝트_도면번호_페이지)
            drawing_number = converted['metadata'].get('drawing_number', f'DOC-{i+1:03d}')
            page_number = converted['metadata'].get('page_number', '1')
            doc_id = f"{project_name}_{drawing_number}_{page_number}"
            ids.append(doc_id)
            
            print(f"✅ 변환됨: {converted['metadata'].get('drawing_title', 'Unknown')} (완성도: {converted['metadata'].get('completion_score', 0)}%)")
            
        except Exception as e:
            print(f"⚠️  도면 #{i+1} 변환 실패: {e}")
            continue
    
    if not documents:
        print(f"❌ 변환된 도면이 없습니다")
        return False
    
    # ChromaDB에 추가
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"🎉 RAG 데이터베이스 구축 완료!")
        print(f"   📊 저장된 문서 수: {len(documents)}개")
        print(f"   📁 컬렉션: {collection_name}")
        print(f"   💾 저장 위치: {CHROMA_DB_PATH}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 저장 실패: {e}")
        return False

def build_all_projects():
    """모든 프로젝트의 RAG 데이터베이스 구축"""
    print("🚀 Self-Query 호환 RAG 데이터베이스 구축 시작...")
    
    success_count = 0
    total_count = 0
    
    # uploads 폴더에서 프로젝트 메타데이터 파일 찾기
    for project_dir in UPLOADS_ROOT_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        
        # project_metadata_*.json 파일 찾기
        metadata_files = list(project_dir.glob("project_metadata_*.json"))
        
        for metadata_file in metadata_files:
            total_count += 1
            print(f"\n🔄 처리 중: {metadata_file}")
            
            if build_self_query_rag_database(metadata_file):
                success_count += 1
            else:
                print(f"❌ 실패: {metadata_file}")
    
    print(f"\n📊 RAG DB 구축 완료:")
    print(f"   ✅ 성공: {success_count}개")
    print(f"   ❌ 실패: {total_count - success_count}개")
    print(f"   💾 저장 위치: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    build_all_projects()
