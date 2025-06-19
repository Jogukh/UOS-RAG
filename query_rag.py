import chromadb
from chromadb.utils import embedding_functions
import argparse
import os
import sys
from pathlib import Path

# .env 설정 로드
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from env_config import get_env_config
    env_config = get_env_config()
    print(f"📋 .env 기반 설정 로드됨 - 모델: {env_config.model_config.model_name}")
    HAS_ENV_CONFIG = True
except ImportError:
    print("⚠️  env_config를 불러올 수 없습니다. 기본 설정을 사용합니다.")
    env_config = None
    HAS_ENV_CONFIG = False

# LangSmith 추적 기능 추가
try:
    from langsmith_integration import (
        langsmith_tracker, 
        trace_llm_call, 
        trace_workflow_step, 
        trace_tool_call
    )
    HAS_LANGSMITH = True
    print("📊 LangSmith 추적 기능 로드됨")
except ImportError:
    print("⚠️  LangSmith 통합 모듈을 불러올 수 없습니다.")
    HAS_LANGSMITH = False
    # Mock decorators
    def trace_llm_call(name=None, run_type="llm"):
        def decorator(func):
            return func
        return decorator
    def trace_workflow_step(name=None, run_type="chain"):
        def decorator(func):
            return func
        return decorator
    def trace_tool_call(name=None):
        def decorator(func):
            return func
        return decorator

# Ollama API 사용
try:
    import requests
    import json
    HAS_OLLAMA = True
except ImportError as e:
    print(f"requests를 import하는 데 실패했습니다: {e}")
    print("질의응답 시 LLM 답변 생성 기능이 제한될 수 있습니다.")
    HAS_OLLAMA = False

# 상수 정의 (.env에서 가져오거나 기본값 사용)
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_PATH = env_config.model_config.model_name if HAS_ENV_CONFIG else "Qwen/Qwen2.5-7B-Instruct"
PROMPT_FILE_PATH = Path(__file__).parent / "src" / "prompt.md"

# 프롬프트 매니저 import
from src.prompt_manager import get_prompt_manager

def load_prompt_template(file_path: Path) -> str:
    """지정된 파일 경로에서 프롬프트 템플릿을 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"프롬프트 파일({file_path})을 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
        return get_default_prompt()
    except Exception as e:
        print(f"프롬프트 파일을 읽는 중 오류 발생: {e}. 기본 프롬프트를 사용합니다.")
        return get_default_prompt()

def get_default_prompt() -> str:
    """기본 프롬프트 반환 - 중앙 관리 프롬프트 매니저 사용"""
    prompt_manager = get_prompt_manager()
    return prompt_manager.get_prompt("rag_query").template

def get_available_collections():
    """사용 가능한 ChromaDB 컬렉션 목록을 반환합니다."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        print(f"컬렉션 목록 가져오기 실패: {e}")
        return []

@trace_tool_call(name="vector_search")
def query_rag_database(query_text, n_results=3, project_name=None):
    """
    사용자 질의를 바탕으로 프로젝트별 RAG 데이터베이스를 검색합니다.
    """
    # ChromaDB 클라이언트 초기화
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Sentence Transformer 임베딩 함수 설정
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # 프로젝트별 컬렉션 이름 생성
    if project_name:
        collection_name = f"drawings_{project_name}".replace(" ", "_").replace("-", "_").lower()
        collection_name = "".join(c if c.isalnum() or c == "_" else "_" for c in collection_name)
        
        # 지정된 프로젝트 컬렉션 검색
        try:
            collection = client.get_collection(
                name=collection_name
            )
            print(f"프로젝트 '{project_name}' 컬렉션에서 검색 중...")
        except Exception as e:
            print(f"프로젝트 '{project_name}' 컬렉션을 찾을 수 없습니다: {e}")
            print("사용 가능한 컬렉션 목록:")
            available_collections = get_available_collections()
            for col in available_collections:
                print(f"  - {col}")
            return None
    else:
        # 모든 프로젝트에서 검색 (여러 컬렉션 통합 검색)
        available_collections = get_available_collections()
        drawings_collections = [col for col in available_collections if col.startswith("drawings_")]
        
        if not drawings_collections:
            print("검색 가능한 도면 컬렉션이 없습니다.")
            return None
        
        print(f"모든 프로젝트({len(drawings_collections)}개 컬렉션)에서 검색 중...")
        
        # 각 컬렉션에서 검색하고 결과 통합
        all_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        for col_name in drawings_collections:
            try:
                collection = client.get_collection(
                    name=col_name
                )
                
                results = collection.query(
                    query_texts=[query_text],
                    n_results=min(n_results, 10),  # 컬렉션별 최대 검색 수 제한
                    include=['metadatas', 'documents', 'distances']
                )
                
                if results and results.get('ids') and results['ids'][0]:
                    all_results["ids"][0].extend(results["ids"][0])
                    all_results["documents"][0].extend(results["documents"][0])
                    all_results["metadatas"][0].extend(results["metadatas"][0])
                    all_results["distances"][0].extend(results["distances"][0])
                    
            except Exception as e:
                print(f"컬렉션 {col_name} 검색 중 오류: {e}")
                continue
        
        # 거리 기준으로 정렬하고 상위 n_results개만 반환
        if all_results["distances"][0]:
            combined_results = list(zip(
                all_results["ids"][0],
                all_results["documents"][0], 
                all_results["metadatas"][0],
                all_results["distances"][0]
            ))
            combined_results.sort(key=lambda x: x[3])  # 거리 기준 정렬
            combined_results = combined_results[:n_results]
            
            # 결과 재구성
            all_results = {
                "ids": [[item[0] for item in combined_results]],
                "documents": [[item[1] for item in combined_results]],
                "metadatas": [[item[2] for item in combined_results]],
                "distances": [[item[3] for item in combined_results]]
            }
        
        return all_results

    # 단일 컬렉션 검색
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        return results
    except Exception as e:
        print(f"DB 질의 중 오류 발생: {e}")
        return None

@trace_tool_call(name="initialize_llm")
def initialize_llm():
    """Ollama API 기반 텍스트 LLM을 초기화합니다."""
    if not HAS_OLLAMA:
        print("requests가 없어 LLM 초기화를 건너뜁니다.")
        return None
    
    # Ollama 서버 연결 테스트
    ollama_url = "http://localhost:11434"
    model_name = LLM_MODEL_PATH  # gemma3:12b-it-qat
    
    try:
        # Ollama 서버 상태 확인
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if model_name in model_names:
                print(f"✅ Ollama 모델 '{model_name}'이 사용 가능합니다.")
                return {
                    "ollama_url": ollama_url,
                    "model_name": model_name,
                    "type": "ollama"
                }
            else:
                print(f"❌ 모델 '{model_name}'을 찾을 수 없습니다.")
                print(f"사용 가능한 모델들: {model_names}")
                return None
        else:
            print(f"❌ Ollama 서버에 연결할 수 없습니다. 상태 코드: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Ollama 서버가 실행되지 않았습니다. 'ollama serve' 명령으로 서버를 시작하세요.")
        return None
    except Exception as e:
        print(f"❌ LLM 초기화 중 오류 발생: {e}")
        return None

@trace_llm_call(name="ollama_generate_answer", run_type="llm")
def generate_answer_with_llm(llm_components, query_text, retrieved_documents_text):
    """검색된 문서를 바탕으로 Ollama API를 사용하여 답변을 생성합니다."""
    if not llm_components:
        return "LLM이 초기화되지 않아 답변을 생성할 수 없습니다."

    if llm_components.get("type") != "ollama":
        return "Ollama API가 아닌 다른 LLM 타입은 지원되지 않습니다."
    
    ollama_url = llm_components["ollama_url"]
    model_name = llm_components["model_name"]
    
    # 프롬프트 파일에서 템플릿 로드
    prompt_template = load_prompt_template(PROMPT_FILE_PATH)
    
    # 프롬프트에 변수 채우기
    formatted_prompt = prompt_template.format(
        retrieved_documents_text=retrieved_documents_text, 
        query_text=query_text
    )
    
    # Ollama API 요청 데이터
    request_data = {
        "model": model_name,
        "prompt": formatted_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.8,
            "num_predict": 2048
        }
    }

    try:
        # Ollama API 호출
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=request_data,
            timeout=60  # 60초로 늘림
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get("response", "").strip()
            return response_text
        else:
            print(f"❌ Ollama API 오류. 상태 코드: {response.status_code}")
            print(f"응답: {response.text}")
            return f"LLM API 호출 실패: {response.status_code}"

    except requests.exceptions.Timeout:
        return "LLM 응답 시간이 초과되었습니다."
    except Exception as e:
        print(f"❌ LLM 답변 생성 중 오류 발생: {e}")
        return f"LLM 답변 생성 중 오류가 발생했습니다: {str(e)}"

def display_results(results, llm_analyzer=None, original_query=None):
    """검색 결과를 사용자에게 표시하고, LLM을 사용하여 답변을 생성합니다."""
    if not results or not results.get('ids') or not results['ids'][0]:
        print("검색 결과가 없습니다.")
        if llm_analyzer and original_query:
            print("\\n--- LLM 일반 답변 ---")
            llm_answer = generate_answer_with_llm(llm_analyzer, original_query, "관련 정보가 없습니다.")
            print(f"LLM 답변: {llm_answer}")
        return

    print("\\n--- 검색 결과 ---")
    retrieved_context = ""
    
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else "N/A"
        document_content = results['documents'][0][i] if results.get('documents') and results['documents'][0] else ""
        metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}

        retrieved_context += f"\\n문서 ID: {doc_id}\\n"
        retrieved_context += f"내용: {document_content}\\n---\\n"

        print(f"\\n결과 {i+1}:")
        print(f"  ID: {doc_id}")
        print(f"  유사도 (거리): {distance:.4f}")
        print(f"  프로젝트: {metadata.get('project_name', 'N/A')}")
        print(f"  파일명: {metadata.get('file_name', 'N/A')}")
        print(f"  페이지: {metadata.get('page_number', 'N/A')}")
        print(f"  도면번호: {metadata.get('drawing_number', 'N/A')}")
        print(f"  도면제목: {metadata.get('drawing_title', 'N/A')}")
        print(f"  도면유형: {metadata.get('drawing_type', 'N/A')}")

    if llm_analyzer and original_query:
        print("\\n--- LLM 답변 (검색 기반) ---")
        llm_answer = generate_answer_with_llm(llm_analyzer, original_query, retrieved_context)
        print(f"LLM 답변: {llm_answer}")
        if results.get('ids') and results['ids'][0]:
            print(f"참고한 주요 문서 ID(들): {', '.join(results['ids'][0][:3])}")

@trace_workflow_step(name="rag_query_workflow", run_type="chain")
def execute_rag_workflow(query_text, n_results, project_name, use_llm=True):
    """전체 RAG 워크플로우를 실행하고 추적합니다."""
    workflow_metadata = {
        "query": query_text,
        "n_results": n_results,
        "project": project_name,
        "use_llm": use_llm
    }
    
    # LangSmith 세션 시작
    if HAS_LANGSMITH and langsmith_tracker.is_enabled():
        session_id = langsmith_tracker.start_session("RAG_Query", workflow_metadata)
        print(f"📊 LangSmith 추적 세션 시작: {session_id}")
    
    try:
        # LLM 초기화
        llm_analyzer_instance = None
        if use_llm:
            print("LLM 모델을 초기화하는 중...")
            llm_analyzer_instance = initialize_llm()
            if not llm_analyzer_instance:
                print("경고: LLM 모델 초기화에 실패하여 LLM 답변 없이 검색만 수행합니다.")
        
        # RAG 검색 실행
        search_results = query_rag_database(query_text, n_results, project_name)
        
        # 결과 표시 및 LLM 답변 생성
        if search_results:
            display_results(search_results, llm_analyzer_instance, query_text)
        elif llm_analyzer_instance and use_llm:
            print("\\n--- LLM 일반 답변 (검색 결과 없음) ---")
            llm_answer = generate_answer_with_llm(llm_analyzer_instance, query_text, "관련 정보를 찾을 수 없었습니다.")
            print(f"LLM 답변: {llm_answer}")
        
        return search_results
        
    except Exception as e:
        print(f"❌ RAG 워크플로우 실행 중 오류: {e}")
        return None
    finally:
        # LangSmith 세션 종료
        if HAS_LANGSMITH and langsmith_tracker.is_enabled():
            langsmith_tracker.end_session('session_id' if 'session_id' in locals() else 'default')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="건축 도면 RAG DB 질의 시스템 (프로젝트별)")
    parser.add_argument("query", type=str, nargs='?', help="검색할 질의 내용")
    parser.add_argument("-n", "--n_results", type=int, default=3, help="반환할 검색 결과 수 (기본값: 3)")
    parser.add_argument("-p", "--project", type=str, default=None, help="검색할 프로젝트 이름 (선택 사항, 미지정 시 모든 프로젝트 검색)")
    parser.add_argument("--no_llm", action="store_true", help="LLM 답변 생성을 비활성화합니다.")
    parser.add_argument("--list_projects", action="store_true", help="사용 가능한 프로젝트 목록을 표시합니다.")
    
    args = parser.parse_args()

    if args.list_projects:
        print("사용 가능한 프로젝트 컬렉션:")
        available_collections = get_available_collections()
        drawings_collections = [col for col in available_collections if col.startswith("drawings_")]
        for col in drawings_collections:
            project_name = col.replace("drawings_", "")
            print(f"  - {project_name}")
        exit(0)

    # query가 제공되지 않았을 때 사용법 예시 표시
    if not args.query:
        print("❌ 질의 텍스트가 필요합니다!")
        print("\n📋 사용법 예시:")
        print("python query_rag.py \"부산장안지구 아파트 도면에서 화장실 배치도를 찾아줘\"")
        print("python query_rag.py \"전기 배선도\" -p 부산장안지구 -n 5")
        print("python query_rag.py \"건축 도면\" --no_llm")
        print("python query_rag.py --list_projects")
        print("\n더 자세한 도움말은 'python query_rag.py --help'를 참조하세요.")
        exit(1)

    print(f"질의: \"{args.query}\"")
    if args.project:
        print(f"대상 프로젝트: {args.project}")
    else:
        print("모든 프로젝트에서 검색")
    
    # 새로운 추적 가능한 워크플로우 실행
    execute_rag_workflow(
        query_text=args.query,
        n_results=args.n_results,
        project_name=args.project,
        use_llm=not args.no_llm
    )
