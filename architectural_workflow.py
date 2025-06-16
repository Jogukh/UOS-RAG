#!/usr/bin/env python3
"""
LangGraph 기반 건축 도면 분석 워크플로우
Sequential Thinking과 MCP 도구들을 활용한 체인 구성
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import json
from pathlib import Path
import operator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# MCP 도구들은 별도 호출 (코드에는 구현하지 않음)
# - Sequential Thinking MCP
# - Context7 MCP  
# - Tavily MCP

class WorkflowState(TypedDict):
    """워크플로우 상태 정의"""
    # 입력 데이터
    project_path: str
    analysis_type: str  # "full", "metadata_only", "relationships_only", "rag_only"
    
    # 처리 단계별 상태
    step: str
    current_task: str
    progress: float
    
    # 데이터 상태
    pdf_texts: Dict[str, Any]
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
                "metadata_only": "extract_metadata",
                "relationships_only": "infer_relationships",
                "rag_only": "build_rag_db"
            }
        )
        
        self.workflow.add_edge("extract_pdf_text", "extract_metadata")
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
        
        # 여기서 Sequential Thinking MCP 도구를 호출하여 분석 계획 수립
        # (실제 MCP 호출은 외부에서 수행)
        
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
        """메타데이터 추출"""
        state["step"] = "extract_metadata"
        state["current_task"] = "도면 메타데이터 추출"
        
        try:
            # 실제 메타데이터 추출 로직 (extract_metadata.py 호출)
            import subprocess
            result = subprocess.run(
                ["python", "extract_metadata.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                state["metadata"] = {"status": "success", "message": "메타데이터 추출 완료"}
                state["logs"].append(f"[{datetime.now()}] 메타데이터 추출 성공")
            else:
                state["errors"].append(f"메타데이터 추출 실패: {result.stderr}")
                
        except Exception as e:
            state["errors"].append(f"메타데이터 추출 중 오류: {str(e)}")
        
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
        """RAG 데이터베이스 구축"""
        state["step"] = "build_rag_db"
        state["current_task"] = "RAG 데이터베이스 구축"
        
        try:
            # 실제 RAG DB 구축 로직 (build_rag_db.py 호출)
            import subprocess
            result = subprocess.run(
                ["python", "build_rag_db.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                state["rag_db_status"] = True
                state["logs"].append(f"[{datetime.now()}] RAG DB 구축 성공")
            else:
                state["rag_db_status"] = False
                state["errors"].append(f"RAG DB 구축 실패: {result.stderr}")
                
        except Exception as e:
            state["rag_db_status"] = False
            state["errors"].append(f"RAG DB 구축 중 오류: {str(e)}")
        
        state["progress"] = 90.0
        return state
    
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
        """다음 단계 라우팅"""
        analysis_type = state["analysis_type"]
        
        if analysis_type == "full":
            return "extract_text"
        elif analysis_type == "metadata_only":
            return "metadata_only"
        elif analysis_type == "relationships_only":
            return "relationships_only"
        elif analysis_type == "rag_only":
            return "rag_only"
        else:
            return "extract_text"  # 기본값
    
    def _get_next_steps(self, analysis_type: str) -> List[str]:
        """분석 유형별 다음 단계 목록"""
        steps_map = {
            "full": ["extract_pdf_text", "extract_metadata", "infer_relationships", "build_rag_db"],
            "metadata_only": ["extract_metadata"],
            "relationships_only": ["infer_relationships"],
            "rag_only": ["build_rag_db"]
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
        
        # MCP 도구 활용 참고사항
        report += """
## MCP 도구 활용 참고사항

이 워크플로우는 다음 MCP 도구들과 연계하여 실행됩니다:

### Sequential Thinking MCP
- 복잡한 분석 과정을 체계적으로 사고
- 단계별 의사결정 지원
- 분석 품질 향상

### Context7 MCP  
- LangChain, LangGraph 최신 문서 참조
- 건축 도메인 지식 활용
- 기술적 구현 가이드

### Tavily MCP
- 실시간 웹 검색으로 최신 정보 수집
- 건축 기준 및 규정 확인
- 기술 동향 파악

### 활용 방법
1. Sequential Thinking으로 분석 계획 수립
2. Context7으로 기술 문서 조회
3. Tavily로 최신 정보 보완
4. LangGraph 워크플로우 실행
"""
        
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

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph 기반 건축 도면 분석 워크플로우")
    parser.add_argument("project_path", help="분석할 프로젝트 경로")
    parser.add_argument("--analysis-type", choices=["full", "metadata_only", "relationships_only", "rag_only"], 
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
