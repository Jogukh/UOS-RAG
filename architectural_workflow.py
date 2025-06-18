#!/usr/bin/env python3
"""
LangGraph 기반 건축 도면 분석 워크플로우
.env 파일 기반 설정 사용
PDF 및 DWG/DXF 파일 통합 분석 지원
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import json
from pathlib import Path
import operator
import sys

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

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# DWG 분석 모듈 import
try:
    from src.dwg_parser import DWGParser
    from src.dwg_metadata_extractor import DWGMetadataExtractor
    from src.langsmith_integration import trace_llm_call, LangSmithTracker
    HAS_DWG_MODULES = True
except ImportError:
    print("⚠️  DWG 분석 모듈을 불러올 수 없습니다.")
    HAS_DWG_MODULES = False

class WorkflowState(TypedDict):
    """워크플로우 상태 정의"""
    # 입력 데이터
    project_path: str
    analysis_type: str  # "full", "metadata_only", "relationships_only", "rag_only", "dwg_only", "pdf_only"
    
    # 처리 단계별 상태
    step: str
    current_task: str
    progress: float
    
    # 데이터 상태
    pdf_texts: Dict[str, Any]
    dwg_data: Dict[str, Any]  # DWG 분석 데이터 추가
    metadata: Dict[str, Any]
    relationships: Dict[str, Any]
    rag_db_status: bool
    
    # 사고 과정 (Sequential Thinking)
    thoughts: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]
    
    # 결과 및 로그
    results: Dict[str, Any]
    logs: List[str]
    errors: List[str]

class ArchitecturalAnalysisWorkflow:
    """LangGraph 기반 건축 도면 분석 워크플로우"""
    
    def __init__(self):
        self.workflow = StateGraph(WorkflowState)
        self._setup_workflow()
        
    def _setup_workflow(self):
        """워크플로우 그래프 구성"""
        
        # 노드 추가
        self.workflow.add_node("initialize", self.initialize_analysis)
        self.workflow.add_node("analyze_requirements", self.analyze_requirements)
        self.workflow.add_node("extract_pdf_text", self.extract_pdf_text)
        self.workflow.add_node("extract_dwg_data", self.extract_dwg_data)  # DWG 분석 노드 추가
        self.workflow.add_node("extract_metadata", self.extract_metadata)
        self.workflow.add_node("infer_relationships", self.infer_relationships)
        self.workflow.add_node("build_rag_db", self.build_rag_db)
        self.workflow.add_node("validate_results", self.validate_results)
        self.workflow.add_node("generate_report", self.generate_report)
        
        # 엣지 정의 (조건부 라우팅 포함)
        self.workflow.set_entry_point("initialize")
        
        self.workflow.add_edge("initialize", "analyze_requirements")
        self.workflow.add_conditional_edges(
            "analyze_requirements",
            self.route_next_step,
            {
                "extract_text": "extract_pdf_text",
                "extract_dwg": "extract_dwg_data",  # DWG 분석 경로 추가
                "metadata_only": "extract_metadata",
                "relationships_only": "infer_relationships",
                "rag_only": "build_rag_db"
            }
        )
        
        self.workflow.add_edge("extract_pdf_text", "extract_metadata")
        self.workflow.add_edge("extract_dwg_data", "extract_metadata")  # DWG → 메타데이터 경로
        self.workflow.add_edge("extract_metadata", "infer_relationships")
        self.workflow.add_edge("infer_relationships", "build_rag_db")
        self.workflow.add_edge("build_rag_db", "validate_results")
        self.workflow.add_edge("validate_results", "generate_report")
        self.workflow.add_edge("generate_report", END)
        
        # 워크플로우 컴파일
        self.app = self.workflow.compile()
    
    def initialize_analysis(self, state: WorkflowState) -> WorkflowState:
        """분석 초기화"""
        state["step"] = "initialize"
        state["current_task"] = "워크플로우 초기화"
        state["progress"] = 0.0
        state["thoughts"] = []
        state["decisions"] = []
        state["results"] = {}
        state["logs"] = [f"[{datetime.now()}] 워크플로우 초기화 시작"]
        state["errors"] = []
        
        # 프로젝트 경로 검증
        project_path = Path(state["project_path"])
        if not project_path.exists():
            state["errors"].append(f"프로젝트 경로가 존재하지 않음: {project_path}")
            
        state["logs"].append(f"[{datetime.now()}] 프로젝트 경로: {project_path}")
        state["progress"] = 10.0
        
        return state
    
    def analyze_requirements(self, state: WorkflowState) -> WorkflowState:
        """요구사항 분석 (Sequential Thinking 활용)"""
        state["step"] = "analyze_requirements"
        state["current_task"] = "요구사항 분석 및 실행 계획 수립"
        
        # 분석 유형에 따른 사고 과정 시뮬레이션
        analysis_type = state["analysis_type"]
        
        thought_process = {
            "thought_1": f"분석 유형: {analysis_type}",
            "thought_2": "필요한 단계들을 순서대로 계획",
            "thought_3": "각 단계별 예상 시간과 리소스 계산",
            "decision": f"{analysis_type} 분석에 최적화된 워크플로우 선택"
        }
        
        state["thoughts"].append(thought_process)
        state["decisions"].append({
            "decision": "워크플로우 경로 결정",
            "rationale": f"{analysis_type} 분석에 필요한 단계들만 실행",
            "next_steps": self._get_next_steps(analysis_type)
        })
        
        state["logs"].append(f"[{datetime.now()}] 요구사항 분석 완료: {analysis_type}")
        state["progress"] = 20.0
        
        return state
    
    def extract_pdf_text(self, state: WorkflowState) -> WorkflowState:
        """PDF 텍스트 추출"""
        state["step"] = "extract_pdf_text"
        state["current_task"] = "PDF 파일에서 텍스트 추출"
        
        try:
            # 실제 PDF 텍스트 추출 로직 (analyze_uploads_new.py 호출)
            import subprocess
            result = subprocess.run(
                ["python", "analyze_uploads_new.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                state["pdf_texts"] = {"status": "success", "message": "PDF 텍스트 추출 완료"}
                state["logs"].append(f"[{datetime.now()}] PDF 텍스트 추출 성공")
            else:
                state["errors"].append(f"PDF 텍스트 추출 실패: {result.stderr}")
                
        except Exception as e:
            state["errors"].append(f"PDF 텍스트 추출 중 오류: {str(e)}")
        
        state["progress"] = 40.0
        return state
    
    def extract_metadata(self, state: WorkflowState) -> WorkflowState:
        """메타데이터 추출 - PDF와 DWG 데이터 통합"""
        state["step"] = "extract_metadata"
        state["current_task"] = "통합 메타데이터 추출"
        
        try:
            # PDF 메타데이터 추출
            if state.get("pdf_texts", {}).get("status") == "success":
                # 실제 PDF 메타데이터 추출 로직 (extract_metadata.py 호출)
                import subprocess
                result = subprocess.run(
                    ["python", "extract_metadata.py"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    state["logs"].append(f"[{datetime.now()}] PDF 메타데이터 추출 성공")
                else:
                    state["errors"].append(f"PDF 메타데이터 추출 실패: {result.stderr}")
            
            # DWG 메타데이터는 이미 extract_dwg_data에서 처리됨
            dwg_data = state.get("dwg_data", {})
            if dwg_data.get("status") == "success":
                # DWG 메타데이터를 통합 메타데이터에 병합
                extracted_dwg_data = dwg_data.get("data", {})
                
                integrated_metadata = {
                    "pdf_metadata": state.get("metadata", {}),
                    "dwg_metadata": extracted_dwg_data,
                    "integration_timestamp": datetime.now().isoformat(),
                    "total_dwg_files": dwg_data.get("files_processed", 0)
                }
                
                state["metadata"] = integrated_metadata
                state["logs"].append(f"[{datetime.now()}] DWG 메타데이터 통합 완료")
            
            # 통합 메타데이터 상태 설정
            if not state.get("metadata"):
                state["metadata"] = {"status": "no_data", "message": "추출할 메타데이터가 없습니다"}
            
        except Exception as e:
            state["errors"].append(f"메타데이터 추출 중 오류: {str(e)}")
            state["metadata"] = {"status": "error", "error": str(e)}
        
        state["progress"] = 60.0
        return state
    
    def infer_relationships(self, state: WorkflowState) -> WorkflowState:
        """관계 추론"""
        state["step"] = "infer_relationships"
        state["current_task"] = "도면 간 관계 추론"
        
        try:
            # 실제 관계 추론 로직 (infer_relationships.py 호출)
            import subprocess
            result = subprocess.run(
                ["python", "infer_relationships.py", "--use-llm"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                state["relationships"] = {"status": "success", "message": "관계 추론 완료"}
                state["logs"].append(f"[{datetime.now()}] 관계 추론 성공")
            else:
                state["errors"].append(f"관계 추론 실패: {result.stderr}")
                
        except Exception as e:
            state["errors"].append(f"관계 추론 중 오류: {str(e)}")
        
        state["progress"] = 80.0
        return state
    
    def build_rag_db(self, state: WorkflowState) -> WorkflowState:
        """RAG 데이터베이스 구축 - PDF와 DWG 데이터 통합"""
        state["step"] = "build_rag_db"
        state["current_task"] = "통합 RAG 데이터베이스 구축"
        
        try:
            # PDF RAG 데이터 구축
            if state.get("pdf_texts", {}).get("status") == "success":
                import subprocess
                result = subprocess.run(
                    ["python", "build_rag_db.py"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    state["logs"].append(f"[{datetime.now()}] PDF RAG 데이터베이스 구축 성공")
                else:
                    state["errors"].append(f"PDF RAG 구축 실패: {result.stderr}")
            
            # DWG RAG 데이터 구축
            dwg_data = state.get("dwg_data", {})
            if dwg_data.get("status") == "success":
                # DWG RAG 콘텐츠를 RAG 데이터베이스에 추가
                extracted_dwg_data = dwg_data.get("data", {})
                
                rag_contents = []
                for file_path, metadata in extracted_dwg_data.items():
                    rag_content = metadata.get("rag_content", "")
                    if rag_content:
                        rag_contents.append({
                            "source": file_path,
                            "content": rag_content,
                            "metadata": metadata,
                            "type": "dwg_analysis"
                        })
                
                if rag_contents:
                    # RAG 데이터베이스에 DWG 콘텐츠 추가
                    self._add_dwg_to_rag_db(rag_contents, state)
                    state["logs"].append(f"[{datetime.now()}] DWG RAG 콘텐츠 {len(rag_contents)}개 추가")
            
            state["rag_db_status"] = True
            state["logs"].append(f"[{datetime.now()}] 통합 RAG 데이터베이스 구축 완료")
            
        except Exception as e:
            state["errors"].append(f"RAG 데이터베이스 구축 중 오류: {str(e)}")
            state["rag_db_status"] = False
        
        state["progress"] = 85.0
        return state
    
    @trace_llm_call("workflow_add_dwg_to_rag", "chain")
    def _add_dwg_to_rag_db(self, rag_contents: List[Dict[str, Any]], state: WorkflowState):
        """DWG 콘텐츠를 RAG 데이터베이스에 추가"""
        try:
            # ChromaDB에 DWG 데이터 추가하는 로직
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_or_create_collection(
                name="architectural_drawings",
                metadata={"description": "통합 건축 도면 분석 데이터"}
            )
            
            for i, content in enumerate(rag_contents):
                collection.add(
                    documents=[content["content"]],
                    metadatas=[{
                        "source": content["source"],
                        "type": content["type"],
                        "timestamp": datetime.now().isoformat()
                    }],
                    ids=[f"dwg_{i}_{datetime.now().timestamp()}"]
                )
                
            state["logs"].append(f"[{datetime.now()}] ChromaDB에 DWG 데이터 추가 완료")
            
        except Exception as e:
            state["errors"].append(f"DWG RAG 데이터 추가 실패: {str(e)}")
    
    def validate_results(self, state: WorkflowState) -> WorkflowState:
        """결과 검증"""
        state["step"] = "validate_results"
        state["current_task"] = "결과 검증 및 품질 확인"
        
        # 각 단계별 결과 검증
        validation_results = {}
        
        if state.get("pdf_texts"):
            validation_results["pdf_texts"] = state["pdf_texts"]["status"] == "success"
        
        if state.get("metadata"):
            validation_results["metadata"] = state["metadata"]["status"] == "success"
            
        if state.get("relationships"):
            validation_results["relationships"] = state["relationships"]["status"] == "success"
            
        validation_results["rag_db"] = state.get("rag_db_status", False)
        
        state["results"]["validation"] = validation_results
        state["results"]["overall_success"] = all(validation_results.values())
        
        state["logs"].append(f"[{datetime.now()}] 결과 검증 완료")
        state["progress"] = 95.0
        
        return state
    
    def generate_report(self, state: WorkflowState) -> WorkflowState:
        """최종 보고서 생성"""
        state["step"] = "generate_report" 
        state["current_task"] = "최종 보고서 생성"
        
        # 워크플로우 실행 보고서 생성
        report = self._create_workflow_report(state)
        
        # 보고서 저장
        report_path = Path("workflow_reports") / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        state["results"]["report_path"] = str(report_path)
        state["logs"].append(f"[{datetime.now()}] 보고서 생성 완료: {report_path}")
        state["progress"] = 100.0
        
        return state
    
    def route_next_step(self, state: WorkflowState) -> str:
        """다음 단계 라우팅 - DWG 분석 경로 추가"""
        analysis_type = state["analysis_type"]
        project_path = Path(state["project_path"])
        
        # 파일 유형별 존재 확인
        has_pdf = any(project_path.rglob("*.pdf"))
        has_dwg = any(project_path.rglob("*.dwg")) or any(project_path.rglob("*.dxf"))
        
        if analysis_type == "full":
            # 전체 분석 - PDF와 DWG 모두 처리
            if has_pdf and has_dwg:
                return "extract_text"  # PDF 먼저 처리
            elif has_pdf:
                return "extract_text"
            elif has_dwg:
                return "extract_dwg"
            else:
                return "metadata_only"
                
        elif analysis_type == "dwg_only":
            return "extract_dwg"
        elif analysis_type == "pdf_only":
            return "extract_text"
        elif analysis_type == "metadata_only":
            return "metadata_only"
        elif analysis_type == "relationships_only":
            return "relationships_only"
        elif analysis_type == "rag_only":
            return "rag_only"
        else:
            # 기본값: 파일 유형에 따라 자동 결정
            if has_dwg:
                return "extract_dwg"
            elif has_pdf:
                return "extract_text"
            else:
                return "metadata_only"
    
    def _get_next_steps(self, analysis_type: str) -> List[str]:
        """분석 유형별 다음 단계 목록"""
        steps_map = {
            "full": ["extract_pdf_text", "extract_metadata", "infer_relationships", "build_rag_db"],
            "metadata_only": ["extract_metadata"],
            "relationships_only": ["infer_relationships"],
            "rag_only": ["build_rag_db"],
            "dwg_only": ["extract_dwg_data"],
            "pdf_only": ["extract_pdf_text"]
        }
        return steps_map.get(analysis_type, steps_map["full"])
    
    def _create_workflow_report(self, state: WorkflowState) -> str:
        """워크플로우 실행 보고서 생성"""
        
        report = f"""# 건축 도면 분석 워크플로우 실행 보고서

## 실행 정보
- **실행 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **프로젝트 경로**: {state['project_path']}
- **분석 유형**: {state['analysis_type']}
- **최종 진행률**: {state['progress']:.1f}%

## 실행 결과
"""
        
        # 검증 결과
        if "validation" in state["results"]:
            report += "\n### 단계별 성공 여부\n"
            for step, success in state["results"]["validation"].items():
                status = "✅ 성공" if success else "❌ 실패"
                report += f"- **{step}**: {status}\n"
        
        # 사고 과정 (Sequential Thinking 결과)
        if state["thoughts"]:
            report += "\n### 분석 사고 과정\n"
            for i, thought in enumerate(state["thoughts"], 1):
                report += f"\n#### 사고 과정 {i}\n"
                for key, value in thought.items():
                    report += f"- **{key}**: {value}\n"
        
        # 의사결정 과정
        if state["decisions"]:
            report += "\n### 주요 의사결정\n"
            for i, decision in enumerate(state["decisions"], 1):
                report += f"\n#### 의사결정 {i}\n"
                report += f"- **결정**: {decision['decision']}\n"
                report += f"- **근거**: {decision['rationale']}\n"
                if 'next_steps' in decision:
                    report += f"- **다음 단계**: {', '.join(decision['next_steps'])}\n"
        
        # 실행 로그
        report += "\n### 실행 로그\n"
        for log in state["logs"]:
            report += f"- {log}\n"
        
        # 오류 로그
        if state["errors"]:
            report += "\n### 오류 로그\n"
            for error in state["errors"]:
                report += f"- ❌ {error}\n"
        
        return report
    
    def run_workflow(self, project_path: str, analysis_type: str = "full") -> Dict[str, Any]:
        """워크플로우 실행"""
        
        initial_state = WorkflowState(
            project_path=project_path,
            analysis_type=analysis_type,
            step="",
            current_task="",
            progress=0.0,
            pdf_texts={},
            dwg_data={},
            metadata={},
            relationships={},
            rag_db_status=False,
            thoughts=[],
            decisions=[],
            results={},
            logs=[],
            errors=[]
        )
        
        # 워크플로우 실행
        final_state = self.app.invoke(initial_state)
        
        return final_state

    @trace_llm_call("workflow_extract_dwg_data", "chain")
    def extract_dwg_data(self, state: WorkflowState) -> WorkflowState:
        """DWG/DXF 파일 데이터 추출"""
        state["step"] = "extract_dwg_data"
        state["current_task"] = "DWG/DXF 파일 분석 및 메타데이터 추출"
        
        if not HAS_DWG_MODULES:
            state["errors"].append("DWG 분석 모듈이 설치되지 않았습니다.")
            state["progress"] = 40.0
            return state
        
        try:
            project_path = Path(state["project_path"])
            dwg_files = []
            
            # DWG/DXF 파일 찾기 (XREF 폴더 제외)
            for ext in ['*.dwg', '*.dxf']:
                found_files = project_path.rglob(ext)
                for dwg_file in found_files:
                    # XREF 폴더 제외 - 경로에 XREF가 포함된 경우 건너뛰기
                    if 'XREF' not in str(dwg_file).upper():
                        dwg_files.append(dwg_file)
            
            if not dwg_files:
                state["logs"].append(f"[{datetime.now()}] DWG/DXF 파일을 찾을 수 없습니다 (XREF 폴더 제외).")
                state["dwg_data"] = {"status": "no_files", "files": []}
                state["progress"] = 40.0
                return state
            
            # DWG 메타데이터 추출기 초기화
            dwg_extractor = DWGMetadataExtractor()
            
            extracted_data = {}
            
            for dwg_file in dwg_files:
                state["logs"].append(f"[{datetime.now()}] DWG 파일 분석 시작: {dwg_file.name}")
                
                try:
                    # DWG 메타데이터 추출 (프로젝트 경로도 전달)
                    metadata = dwg_extractor.extract_from_dwg_file(
                        str(dwg_file), 
                        str(project_path)  # project_base_path 전달
                    )
                    
                    if metadata:
                        extracted_data[str(dwg_file)] = metadata
                        
                        # 메타데이터 JSON 파일 저장 (RAG 콘텐츠 대신)
                        output_dir = dwg_file.parent / "metadata"
                        output_dir.mkdir(exist_ok=True)
                        
                        output_path = output_dir / f"{dwg_file.stem}_metadata.json"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
                        
                        state["logs"].append(f"[{datetime.now()}] DWG 분석 완료: {dwg_file.name}")
                    else:
                        state["errors"].append(f"DWG 메타데이터 추출 실패: {dwg_file.name}")
                        
                except Exception as e:
                    state["errors"].append(f"DWG 파일 처리 중 오류 ({dwg_file.name}): {str(e)}")
            
            state["dwg_data"] = {
                "status": "success",
                "files_processed": len(extracted_data),
                "total_files": len(dwg_files),
                "data": extracted_data
            }
            
            # 사고 과정 기록
            state["thoughts"].append({
                "thought": f"DWG 파일 {len(dwg_files)}개 중 {len(extracted_data)}개 처리 완료",
                "analysis": "DWG 파일에서 구조적 데이터와 메타데이터를 성공적으로 추출",
                "next_action": "추출된 데이터를 통합 메타데이터 시스템에 연동"
            })
            
            state["logs"].append(f"[{datetime.now()}] DWG 데이터 추출 완료: {len(extracted_data)}/{len(dwg_files)} 파일")
            
        except Exception as e:
            state["errors"].append(f"DWG 데이터 추출 중 오류: {str(e)}")
            state["dwg_data"] = {"status": "error", "error": str(e)}
        
        state["progress"] = 40.0
        return state

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph 기반 건축 도면 분석 워크플로우")
    parser.add_argument("project_path", help="분석할 프로젝트 경로")
    parser.add_argument("--analysis-type", choices=["full", "metadata_only", "relationships_only", "rag_only", "dwg_only", "pdf_only"], 
                       default="full", help="분석 유형")
    
    args = parser.parse_args()
    
    # 워크플로우 실행
    workflow = ArchitecturalAnalysisWorkflow()
    result = workflow.run_workflow(args.project_path, args.analysis_type)
    
    # 결과 출력
    print(f"워크플로우 실행 완료!")
    print(f"진행률: {result['progress']:.1f}%")
    print(f"전체 성공 여부: {result['results'].get('overall_success', False)}")
    
    if result.get("results", {}).get("report_path"):
        print(f"상세 보고서: {result['results']['report_path']}")

if __name__ == "__main__":
    main()
