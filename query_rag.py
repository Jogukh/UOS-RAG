import chromadb
from chromadb.utils import embedding_functions
import argparse
import os
import sys
from pathlib import Path

# vLLM 기반 텍스트 LLM 사용
try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError as e:
    print(f"vLLM을 import하는 데 실패했습니다: {e}")
    print("질의응답 시 LLM 답변 생성 기능이 제한될 수 있습니다.")
    HAS_VLLM = False

# 상수 정의
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"  # 텍스트 전용 모델
PROMPT_FILE_PATH = Path(__file__).parent / "src" / "prompt.md"

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
    """기본 프롬프트 반환"""
    return """당신은 건축 도면 및 프로젝트 문서를 분석하고 질문에 답변하는 AI 어시스턴트입니다.
주어진 건축 도면 정보를 바탕으로 다음 질문에 명확하고 간결하게 답변해주세요.

제공된 정보:
---
{retrieved_documents_text}
---

질문: {query_text}

답변 생성 시 다음 사항을 유의해주세요:
1. 답변은 반드시 제공된 정보에 근거해야 합니다.
2. 관련된 정보가 없다면, 추측하지 말고 "제공된 정보로는 답변할 수 없습니다."라고 명확히 밝혀주세요.
3. 가능하다면 답변에 관련된 핵심 도면 정보(예: 문서 ID, 도면명)를 간략히 언급해주세요.

답변:"""

def get_available_collections():
    """사용 가능한 ChromaDB 컬렉션 목록을 반환합니다."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        print(f"컬렉션 목록 가져오기 실패: {e}")
        return []

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
                name=collection_name,
                embedding_function=sentence_transformer_ef
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
                    name=col_name,
                    embedding_function=sentence_transformer_ef
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

def initialize_llm():
    """vLLM 기반 텍스트 LLM을 초기화합니다."""
    if not HAS_VLLM:
        print("vLLM이 없어 LLM 초기화를 건너뜁니다.")
        return None
    try:
        llm = LLM(
            model=LLM_MODEL_PATH,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            max_model_len=32768,
            dtype="bfloat16",
            trust_remote_code=True
        )
        
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.8,
            max_tokens=2048,
            repetition_penalty=1.02,
            stop=["<|endoftext|>", "<|im_end|>"]
        )
        
        print(f"LLM 모델 '{LLM_MODEL_PATH}'이 성공적으로 로드되었습니다.")
        return {"llm": llm, "sampling_params": sampling_params}
        
    except Exception as e:
        print(f"LLM 초기화 중 오류 발생: {e}")
        return None

def generate_answer_with_llm(llm_components, query_text, retrieved_documents_text):
    """검색된 문서를 바탕으로 vLLM을 사용하여 답변을 생성합니다."""
    if not llm_components:
        return "LLM이 초기화되지 않아 답변을 생성할 수 없습니다."

    llm = llm_components["llm"]
    sampling_params = llm_components["sampling_params"]
    
    # 프롬프트 파일에서 템플릿 로드
    prompt_template = load_prompt_template(PROMPT_FILE_PATH)
    
    # 프롬프트에 변수 채우기
    formatted_prompt = prompt_template.format(
        retrieved_documents_text=retrieved_documents_text, 
        query_text=query_text
    )
    
    # Qwen2.5 채팅 형식으로 감싸기
    chat_prompt = f"""<|im_start|>system
당신은 건축 도면 분석 전문가입니다. 주어진 도면 정보를 바탕으로 정확하고 유용한 답변을 제공해주세요.<|im_end|>

<|im_start|>user
{formatted_prompt}<|im_end|>

<|im_start|>assistant
"""

    try:
        outputs = llm.generate([chat_prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        # 프롬프트 부분 제거하고 답변만 추출
        if len(response_text) > len(formatted_prompt):
            response_text = response_text[len(formatted_prompt):].strip()

        return response_text

    except Exception as e:
        print(f"LLM 답변 생성 중 오류 발생: {e}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="건축 도면 RAG DB 질의 시스템 (프로젝트별)")
    parser.add_argument("query", type=str, help="검색할 질의 내용")
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

    llm_analyzer_instance = None
    if not args.no_llm:
        print("LLM 모델을 초기화하는 중...")
        llm_analyzer_instance = initialize_llm()
        if not llm_analyzer_instance:
            print("경고: LLM 모델 초기화에 실패하여 LLM 답변 없이 검색만 수행합니다.")
    else:
        print("LLM 답변 생성이 비활성화되었습니다.")

    print(f"질의: \"{args.query}\"")
    if args.project:
        print(f"대상 프로젝트: {args.project}")
    else:
        print("모든 프로젝트에서 검색")
        
    search_results = query_rag_database(args.query, args.n_results, args.project)

    if search_results:
        display_results(search_results, llm_analyzer_instance, args.query)
    elif llm_analyzer_instance and not args.no_llm:
        print("\\n--- LLM 일반 답변 (검색 결과 없음) ---")
        llm_answer = generate_answer_with_llm(llm_analyzer_instance, args.query, "관련 정보를 찾을 수 없었습니다.")
        print(f"LLM 답변: {llm_answer}")
