"""
LangGraph 워크플로우 상태 정의 및 관리
"""
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from datetime import datetime
import operator
from enum import Enum

class WorkflowStage(Enum):
    """워크플로우 단계 정의"""
    INITIALIZED = "initialized"
    PDF_EXTRACTING = "pdf_extracting"
    PDF_EXTRACTED = "pdf_extracted"
    METADATA_EXTRACTING = "metadata_extracting"
    METADATA_EXTRACTED = "metadata_extracted"
    RELATIONSHIP_INFERRING = "relationship_inferring"
    RELATIONSHIP_INFERRED = "relationship_inferred"
    RAG_BUILDING = "rag_building"
    RAG_BUILT = "rag_built"
    QUERY_READY = "query_ready"
    ERROR = "error"
    COMPLETED = "completed"

class WorkflowState(TypedDict):
    """LangGraph 워크플로우 전체 상태"""
    # 메타데이터
    workflow_id: str
    stage: WorkflowStage
    created_at: datetime
    updated_at: datetime
    
    # 입력 데이터
    uploads_dir: str
    target_files: List[str]
    
    # 처리 결과
    extracted_texts: Dict[str, str]  # 파일명: 추출된 텍스트
    metadata: Dict[str, Dict[str, Any]]  # 파일명: 메타데이터
    relationships: List[Dict[str, Any]]  # 추론된 관계들
    rag_collection_name: str
    
    # 웹 검색 결과 (Tavily)
    web_search_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # 오류 및 재시도
    errors: Annotated[List[Dict[str, Any]], operator.add]
    retry_count: int
    max_retries: int
    
    # 로그 및 추적
    execution_log: Annotated[List[Dict[str, Any]], operator.add]
    performance_metrics: Dict[str, Any]
    
    # 설정
    config: Dict[str, Any]

class NodeExecutionLog(TypedDict):
    """노드 실행 로그"""
    node_name: str
    timestamp: datetime
    duration_seconds: float
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    success: bool
    error_message: Optional[str]
    retry_attempt: int

class PerformanceMetrics(TypedDict):
    """성능 메트릭"""
    total_execution_time: float
    files_processed: int
    metadata_extracted_count: int
    relationships_inferred_count: int
    web_searches_performed: int
    errors_encountered: int
    retries_performed: int
    
def create_initial_state(
    uploads_dir: str,
    target_files: List[str],
    config: Dict[str, Any]
) -> WorkflowState:
    """초기 워크플로우 상태 생성"""
    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return WorkflowState(
        workflow_id=workflow_id,
        stage=WorkflowStage.INITIALIZED,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        uploads_dir=uploads_dir,
        target_files=target_files,
        extracted_texts={},
        metadata={},
        relationships=[],
        rag_collection_name="",
        web_search_results=[],
        errors=[],
        retry_count=0,
        max_retries=config.get("max_retries", 3),
        execution_log=[],
        performance_metrics={
            "total_execution_time": 0.0,
            "files_processed": 0,
            "metadata_extracted_count": 0,
            "relationships_inferred_count": 0,
            "web_searches_performed": 0,
            "errors_encountered": 0,
            "retries_performed": 0
        },
        config=config
    )

def update_state_stage(state: WorkflowState, new_stage: WorkflowStage) -> WorkflowState:
    """워크플로우 단계 업데이트"""
    return {
        **state,
        "stage": new_stage,
        "updated_at": datetime.now()
    }

def add_execution_log(
    state: WorkflowState,
    node_name: str,
    duration: float,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    success: bool,
    error_message: Optional[str] = None,
    retry_attempt: int = 0
) -> Dict[str, Any]:
    """실행 로그 추가"""
    log_entry = NodeExecutionLog(
        node_name=node_name,
        timestamp=datetime.now(),
        duration_seconds=duration,
        input_data=input_data,
        output_data=output_data,
        success=success,
        error_message=error_message,
        retry_attempt=retry_attempt
    )
    
    return {"execution_log": [log_entry]}

def add_error(state: WorkflowState, error_info: Dict[str, Any]) -> Dict[str, Any]:
    """오류 정보 추가"""
    error_entry = {
        "timestamp": datetime.now(),
        "stage": state["stage"],
        **error_info
    }
    
    return {"errors": [error_entry]}
