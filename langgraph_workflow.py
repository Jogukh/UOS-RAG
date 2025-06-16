#!/usr/bin/env python3
"""
LangGraph 기반 건축 도면 분석 워크플로우
Sequ    def __init__(self, llm_model: str = "Qwen/Qwen3-Reranker-4B"):ntial Thinking, Context7, Tavily 등 MCP 도구들을 활용한 체인 구성
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Annotated

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import tool
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.sqlite import SqliteSaver
    HAS_LANGGRAPH = True
except ImportError as e:
    print(f"LangChain/LangGraph가 설치되지 않았습니다: {e}")
    print("pip install langchain langchain-core langchain-community langgraph 로 설치해주세요.")
    HAS_LANGGRAPH = False

logger = logging.getLogger(__name__)

# 워크플로우 상태 정의
class ArchitecturalWorkflowState(TypedDict):
    """건축 도면 분석 워크플로우 상태"""
    # 입력 데이터
    project_name: str
    pdf_files: List[str]
    query: Optional[str]
    
    # 분석 결과
    extracted_texts: List[Dict[str, Any]]
    metadata_results: List[Dict[str, Any]]  
    relationships: List[Dict[str, Any]]
    rag_db_status: Dict[str, Any]
    
    # 메시지 히스토리
    messages: List[Any]
    
    # 현재 단계
    current_step: str
    completed_steps: List[str]
    
    # 오류 정보
    errors: List[str]
    
    # 최종 결과
    final_results: Dict[str, Any]

class ArchitecturalAnalysisWorkflow:
    """LangGraph 기반 건축 도면 분석 워크플로우"""
    
    def __init__(self, llm_model: str = "Qwen/Qwen2.5-3B-Instruct"):
        """
        Args:
            llm_model: 사용할 LLM 모델 (vLLM 기반, GPU 사용)
        """
        self.llm_model = llm_model
        self.llm = None
        self.sampling_params = None
        self.workflow = None
        self.app = None
        
        if HAS_LANGGRAPH:
            self._initialize_llm()
            self._build_workflow()
            
    def _initialize_llm(self):
        """GPU를 사용하여 vLLM 초기화"""
        try:
            # GPU 사용 확인
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA가 사용 불가능합니다. GPU가 필요합니다.")
            
            print(f"🔥 GPU 사용 가능: {torch.cuda.get_device_name(0)}")
            
            # vLLM import
            from vllm import LLM, SamplingParams
            
            # vLLM GPU 설정으로 초기화 (Qwen3-Reranker-4B에 최적화)
            self.llm = LLM(
                model=self.llm_model,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,  # GPU 메모리 90% 사용
                max_model_len=4096,  # 4B 모델에 적합한 컨텍스트 길이
                dtype="bfloat16",  # bfloat16으로 메모리 효율성 향상
                trust_remote_code=True,
                enforce_eager=True,  # GPU 사용 강제
            )
            
            self.sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.8,
                max_tokens=2048,
                repetition_penalty=1.02,
                stop=["<|endoftext|>", "<|im_end|>"]
            )
            
            print("✅ vLLM GPU 초기화 완료")
            logger.info(f"vLLM 모델 '{self.llm_model}' GPU 초기화 완료")
        except Exception as e:
            print(f"❌ vLLM 초기화 실패: {e}")
            logger.error(f"vLLM 초기화 실패: {e}")
            
    def _build_workflow(self):
        """LangGraph 워크플로우 구성"""
        
        # 워크플로우 그래프 생성
        workflow = StateGraph(ArchitecturalWorkflowState)
        
        # 노드 추가
        workflow.add_node("extract_texts", self.extract_texts_node)
        workflow.add_node("extract_metadata", self.extract_metadata_node)
        workflow.add_node("infer_relationships", self.infer_relationships_node)
        workflow.add_node("build_rag_db", self.build_rag_db_node)
        workflow.add_node("query_system", self.query_system_node)
        workflow.add_node("generate_report", self.generate_report_node)
        
        # 엣지 정의 (단계별 실행 순서)
        workflow.set_entry_point("extract_texts")
        workflow.add_edge("extract_texts", "extract_metadata")
        workflow.add_edge("extract_metadata", "infer_relationships")
        workflow.add_edge("infer_relationships", "build_rag_db")
        workflow.add_conditional_edges(
            "build_rag_db",
            self._should_query,
            {
                "query": "query_system",
                "report": "generate_report"
            }
        )
        workflow.add_edge("query_system", "generate_report")
        workflow.add_edge("generate_report", END)
        
        # 체크포인터 설정 (상태 저장)
        checkpointer = SqliteSaver.from_conn_string(":memory:")
        
        # 앱 컴파일
        self.app = workflow.compile(checkpointer=checkpointer)
        
        logger.info("LangGraph 워크플로우 구성 완료")
    
    def _should_query(self, state: ArchitecturalWorkflowState) -> str:
        """쿼리 실행 여부 결정"""
        if state.get("query"):
            return "query"
        return "report"
    
    async def extract_texts_node(self, state: ArchitecturalWorkflowState) -> ArchitecturalWorkflowState:
        """1단계: PDF 텍스트 추출"""
        
        messages = state.get("messages", [])
        messages.append(SystemMessage(content="PDF 텍스트 추출을 시작합니다."))
        
        try:
            # analyze_uploads_new.py 실행 로직 (실제 구현에서는 모듈 import)
            import subprocess
            result = subprocess.run(
                ["python", "analyze_uploads_new.py"], 
                capture_output=True, 
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                # uploads_analysis_results.json 로드
                results_file = Path(__file__).parent.parent / "uploads_analysis_results.json"
                if results_file.exists():
                    with open(results_file, 'r', encoding='utf-8') as f:
                        extracted_texts = json.load(f)
                else:
                    extracted_texts = []
                
                messages.append(AIMessage(content=f"텍스트 추출 완료: {len(extracted_texts)}개 프로젝트"))
                
                return {
                    **state,
                    "extracted_texts": extracted_texts,
                    "current_step": "extract_texts",
                    "completed_steps": state.get("completed_steps", []) + ["extract_texts"],
                    "messages": messages
                }
            else:
                error_msg = f"텍스트 추출 실패: {result.stderr}"
                messages.append(AIMessage(content=error_msg))
                
                return {
                    **state,
                    "errors": state.get("errors", []) + [error_msg],
                    "messages": messages
                }
                
        except Exception as e:
            error_msg = f"텍스트 추출 중 오류: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            
            return {
                **state,
                "errors": state.get("errors", []) + [error_msg],
                "messages": messages
            }
    
    async def extract_metadata_node(self, state: ArchitecturalWorkflowState) -> ArchitecturalWorkflowState:
        """2단계: 메타데이터 추출"""
        
        messages = state.get("messages", [])
        messages.append(SystemMessage(content="메타데이터 추출을 시작합니다."))
        
        try:
            # extract_metadata.py 실행
            import subprocess
            result = subprocess.run(
                ["python", "extract_metadata.py"], 
                capture_output=True, 
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                # 메타데이터 파일들 확인
                uploads_dir = Path(__file__).parent.parent / "uploads"
                metadata_files = list(uploads_dir.glob("**/project_metadata_*.json"))
                
                metadata_results = []
                for file_path in metadata_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata_results.append(json.load(f))
                
                messages.append(AIMessage(content=f"메타데이터 추출 완료: {len(metadata_results)}개 프로젝트"))
                
                return {
                    **state,
                    "metadata_results": metadata_results,
                    "current_step": "extract_metadata", 
                    "completed_steps": state.get("completed_steps", []) + ["extract_metadata"],
                    "messages": messages
                }
            else:
                error_msg = f"메타데이터 추출 실패: {result.stderr}"
                messages.append(AIMessage(content=error_msg))
                
                return {
                    **state,
                    "errors": state.get("errors", []) + [error_msg],
                    "messages": messages
                }
                
        except Exception as e:
            error_msg = f"메타데이터 추출 중 오류: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            
            return {
                **state,
                "errors": state.get("errors", []) + [error_msg],
                "messages": messages
            }
    
    async def infer_relationships_node(self, state: ArchitecturalWorkflowState) -> ArchitecturalWorkflowState:
        """3단계: 관계 추론"""
        
        messages = state.get("messages", [])
        messages.append(SystemMessage(content="도면 간 관계 추론을 시작합니다."))
        
        try:
            # infer_relationships.py 실행 (LLM 사용)
            import subprocess
            result = subprocess.run(
                ["python", "infer_relationships.py", "--use-llm", "--max-drawings-for-llm", "100"], 
                capture_output=True, 
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                # 관계 파일들 확인
                uploads_dir = Path(__file__).parent.parent / "uploads"
                relationship_files = list(uploads_dir.glob("**/*_drawing_relationships.json"))
                
                relationships = []
                for file_path in relationship_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        relationships.append(json.load(f))
                
                messages.append(AIMessage(content=f"관계 추론 완료: {len(relationships)}개 프로젝트"))
                
                return {
                    **state,
                    "relationships": relationships,
                    "current_step": "infer_relationships",
                    "completed_steps": state.get("completed_steps", []) + ["infer_relationships"],
                    "messages": messages
                }
            else:
                error_msg = f"관계 추론 실패: {result.stderr}"
                messages.append(AIMessage(content=error_msg))
                
                return {
                    **state,
                    "errors": state.get("errors", []) + [error_msg],
                    "messages": messages
                }
                
        except Exception as e:
            error_msg = f"관계 추론 중 오류: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            
            return {
                **state,
                "errors": state.get("errors", []) + [error_msg],
                "messages": messages
            }
    
    async def build_rag_db_node(self, state: ArchitecturalWorkflowState) -> ArchitecturalWorkflowState:
        """4단계: RAG 데이터베이스 구축"""
        
        messages = state.get("messages", [])
        messages.append(SystemMessage(content="RAG 데이터베이스 구축을 시작합니다."))
        
        try:
            # build_rag_db.py 실행
            import subprocess
            result = subprocess.run(
                ["python", "build_rag_db.py"], 
                capture_output=True, 
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                rag_db_status = {
                    "status": "success",
                    "output": result.stdout,
                    "timestamp": datetime.now().isoformat()
                }
                
                messages.append(AIMessage(content="RAG 데이터베이스 구축 완료"))
                
                return {
                    **state,
                    "rag_db_status": rag_db_status,
                    "current_step": "build_rag_db",
                    "completed_steps": state.get("completed_steps", []) + ["build_rag_db"],
                    "messages": messages
                }
            else:
                error_msg = f"RAG DB 구축 실패: {result.stderr}"
                messages.append(AIMessage(content=error_msg))
                
                return {
                    **state,
                    "errors": state.get("errors", []) + [error_msg],
                    "messages": messages
                }
                
        except Exception as e:
            error_msg = f"RAG DB 구축 중 오류: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            
            return {
                **state,
                "errors": state.get("errors", []) + [error_msg],
                "messages": messages
            }
    
    async def query_system_node(self, state: ArchitecturalWorkflowState) -> ArchitecturalWorkflowState:
        """5단계: 질의응답 시스템"""
        
        messages = state.get("messages", [])
        query = state.get("query", "")
        
        if not query:
            return {
                **state,
                "current_step": "query_system",
                "completed_steps": state.get("completed_steps", []) + ["query_system"],
                "messages": messages
            }
        
        messages.append(SystemMessage(content=f"질의응답을 실행합니다: {query}"))
        
        try:
            # query_rag.py 실행
            import subprocess
            result = subprocess.run(
                ["python", "query_rag.py", query], 
                capture_output=True, 
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            if result.returncode == 0:
                query_result = result.stdout
                messages.append(AIMessage(content=f"질의응답 완료:\n{query_result}"))
                
                return {
                    **state,
                    "final_results": {
                        **state.get("final_results", {}),
                        "query_result": query_result
                    },
                    "current_step": "query_system",
                    "completed_steps": state.get("completed_steps", []) + ["query_system"],
                    "messages": messages
                }
            else:
                error_msg = f"질의응답 실패: {result.stderr}"
                messages.append(AIMessage(content=error_msg))
                
                return {
                    **state,
                    "errors": state.get("errors", []) + [error_msg],
                    "messages": messages
                }
                
        except Exception as e:
            error_msg = f"질의응답 중 오류: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            
            return {
                **state,
                "errors": state.get("errors", []) + [error_msg],
                "messages": messages
            }
    
    async def generate_report_node(self, state: ArchitecturalWorkflowState) -> ArchitecturalWorkflowState:
        """6단계: 최종 보고서 생성"""
        
        messages = state.get("messages", [])
        messages.append(SystemMessage(content="최종 보고서를 생성합니다."))
        
        try:
            # 워크플로우 실행 결과 요약
            report = {
                "project_name": state.get("project_name", "Unknown"),
                "execution_time": datetime.now().isoformat(),
                "completed_steps": state.get("completed_steps", []),
                "errors": state.get("errors", []),
                "extracted_texts_count": len(state.get("extracted_texts", [])),
                "metadata_results_count": len(state.get("metadata_results", [])),
                "relationships_count": len(state.get("relationships", [])),
                "rag_db_status": state.get("rag_db_status", {}),
                "query_result": state.get("final_results", {}).get("query_result", ""),
                "success": len(state.get("errors", [])) == 0
            }
            
            # 보고서를 마크다운으로 저장
            report_md = self._generate_markdown_report(report)
            report_file = Path(__file__).parent.parent / f"workflow_reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report_file.parent.mkdir(exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_md)
            
            messages.append(AIMessage(content=f"최종 보고서 생성 완료: {report_file}"))
            
            return {
                **state,
                "final_results": {
                    **state.get("final_results", {}),
                    "report": report,
                    "report_file": str(report_file)
                },
                "current_step": "generate_report",
                "completed_steps": state.get("completed_steps", []) + ["generate_report"],
                "messages": messages
            }
            
        except Exception as e:
            error_msg = f"보고서 생성 중 오류: {str(e)}"
            messages.append(AIMessage(content=error_msg))
            
            return {
                **state,
                "errors": state.get("errors", []) + [error_msg],
                "messages": messages
            }
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """마크다운 보고서 생성"""
        
        md_content = f"""# 건축 도면 분석 워크플로우 보고서

## 프로젝트 정보
- **프로젝트명**: {report['project_name']}
- **실행 시간**: {report['execution_time']}
- **성공 여부**: {'✅ 성공' if report['success'] else '❌ 실패'}

## 실행 단계
{chr(10).join([f"- ✅ {step}" for step in report['completed_steps']])}

## 결과 요약
- **추출된 텍스트**: {report['extracted_texts_count']}개 프로젝트
- **메타데이터 결과**: {report['metadata_results_count']}개 프로젝트  
- **관계 추론 결과**: {report['relationships_count']}개 프로젝트
- **RAG DB 상태**: {report['rag_db_status'].get('status', 'Unknown')}

## 질의응답 결과
```
{report.get('query_result', '질의 없음')}
```

## 오류 내역
{chr(10).join([f"- ❌ {error}" for error in report['errors']]) if report['errors'] else '오류 없음'}

---
*LangGraph 기반 자동 생성 보고서*
"""
        return md_content
    
    async def run_workflow(self, 
                          project_name: str,
                          pdf_files: List[str] = None,
                          query: str = None) -> Dict[str, Any]:
        """워크플로우 실행"""
        
        if not HAS_LANGGRAPH:
            raise RuntimeError("LangGraph가 설치되지 않았습니다.")
        
        # 초기 상태 설정
        initial_state = ArchitecturalWorkflowState(
            project_name=project_name,
            pdf_files=pdf_files or [],
            query=query,
            extracted_texts=[],
            metadata_results=[],
            relationships=[],
            rag_db_status={},
            messages=[],
            current_step="",
            completed_steps=[],
            errors=[],
            final_results={}
        )
        
        # 워크플로우 실행
        config = {"configurable": {"thread_id": f"workflow_{datetime.now().timestamp()}"}}
        
        try:
            final_state = await self.app.ainvoke(initial_state, config)
            return final_state
            
        except Exception as e:
            logger.error(f"워크플로우 실행 중 오류: {e}")
            return {
                **initial_state,
                "errors": [str(e)],
                "final_results": {"error": str(e)}
            }

# CLI 인터페이스
async def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph 기반 건축 도면 분석 워크플로우")
    parser.add_argument("--project", type=str, default="Default Project", help="프로젝트명")
    parser.add_argument("--query", type=str, help="질의할 내용 (선택)")
    parser.add_argument("--llm-model", type=str, default="qwen2.5:7b", help="사용할 LLM 모델")
    
    args = parser.parse_args()
    
    # 워크플로우 실행
    workflow = ArchitecturalAnalysisWorkflow(llm_model=args.llm_model)
    
    print(f"🚀 LangGraph 워크플로우 시작: {args.project}")
    
    result = await workflow.run_workflow(
        project_name=args.project,
        query=args.query
    )
    
    if result.get("final_results", {}).get("report_file"):
        print(f"📝 보고서 생성됨: {result['final_results']['report_file']}")
    
    if result.get("errors"):
        print("❌ 오류 발생:")
        for error in result["errors"]:
            print(f"  - {error}")
    else:
        print("✅ 워크플로우 완료!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
