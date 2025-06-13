#!/usr/bin/env python3
"""
LangGraph 기반 VLM 패턴 분석 워크플로우 (수정된 버전)
벡터 분석과 VLM 분석을 체계적으로 통합한 건축 도면 패턴 인식 시스템
"""

import json
import logging
import time
import operator
from typing import Dict, Any, List, Optional, Annotated
from pathlib import Path
from typing_extensions import TypedDict
from dataclasses import dataclass, asdict

# LangGraph imports
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("Warning: LangGraph not available. Using simple workflow.")

# Project imports
try:
    from architectural_vector_analyzer import ArchitecturalVectorAnalyzer
    from qwen_vlm_analyzer_fixed import QwenVLMAnalyzer
    HAS_ANALYZERS = True
except ImportError as e:
    HAS_ANALYZERS = False
    print(f"Warning: Analyzers not available: {e}")

import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


# 워크플로우 상태 정의 (Annotated 사용으로 동시 업데이트 문제 해결)
class PatternAnalysisState(TypedDict):
    """패턴 분석 워크플로우 상태"""
    # 입력
    file_path: str
    page_number: int
    analysis_options: Dict[str, Any]
    
    # 처리 중간 결과
    pdf_page: Optional[Any]  # fitz.Page
    page_image: Optional[Any]  # PIL.Image
    vector_data: Dict[str, Any]
    text_data: List[Dict[str, Any]]
    
    # VLM 분석 결과
    vlm_pattern_analysis: Dict[str, Any]
    vlm_spatial_analysis: Dict[str, Any]
    vlm_element_analysis: Dict[str, Any]
    
    # 벡터 분석 결과
    vector_walls: List[Dict[str, Any]]
    vector_doors: List[Dict[str, Any]]
    vector_windows: List[Dict[str, Any]]
    vector_spaces: List[Dict[str, Any]]
    
    # 통합 결과
    combined_patterns: Dict[str, Any]
    confidence_scores: Dict[str, float]
    final_analysis: Dict[str, Any]
    
    # 메타데이터 (Annotated로 동시 업데이트 허용)
    processing_steps: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]
    status: str


@dataclass
class PatternResult:
    """패턴 인식 결과"""
    pattern_type: str
    confidence: float
    coordinates: List[tuple]
    properties: Dict[str, Any]
    source: str  # 'vector' or 'vlm' or 'combined'
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VLMPatternWorkflow:
    """LangGraph 기반 VLM 패턴 분석 워크플로우"""
    
    def __init__(self, 
                 vector_analyzer: Optional[ArchitecturalVectorAnalyzer] = None,
                 vlm_analyzer: Optional[QwenVLMAnalyzer] = None,
                 use_vllm: bool = True):
        """초기화"""
        
        # 분석기 초기화 (vLLM 사용 옵션 포함)
        self.vector_analyzer = vector_analyzer or ArchitecturalVectorAnalyzer()
        self.vlm_analyzer = vlm_analyzer or QwenVLMAnalyzer(use_vllm=use_vllm)
        
        # vLLM 사용 여부 저장
        self.use_vllm = use_vllm
        
        # 체크포인터 설정
        self.checkpointer = MemorySaver() if HAS_LANGGRAPH else None
        
        # 워크플로우 구축
        if HAS_LANGGRAPH:
            self.workflow = self._build_workflow()
        else:
            self.workflow = None
            logger.warning("LangGraph not available, using fallback workflow")
        
        # 분석 프롬프트 (간단하고 효율적인 버전)
        self.analysis_schema = {
            "walls": {
                "type": "array",
                "items": {
                    "coordinates": "list of [x1,y1,x2,y2]",
                    "thickness": "number",
                    "confidence": "number 0-1"
                }
            },
            "doors": {
                "type": "array", 
                "items": {
                    "coordinates": "list of [x,y,width,height]",
                    "type": "string",
                    "confidence": "number 0-1"
                }
            },
            "windows": {
                "type": "array",
                "items": {
                    "coordinates": "list of [x,y,width,height]",
                    "confidence": "number 0-1"
                }
            },
            "spaces": {
                "type": "array",
                "items": {
                    "name": "string",
                    "boundary": "list of [x,y] points",
                    "area": "number",
                    "confidence": "number 0-1"
                }
            }
        }
    
    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        
        workflow = StateGraph(PatternAnalysisState)
        
        # 노드 추가 (이름 중복 방지)
        workflow.add_node("load_page", self._load_page_node)
        workflow.add_node("extract_vectors", self._extract_vectors_node)
        workflow.add_node("convert_to_image", self._convert_to_image_node)
        workflow.add_node("run_vlm_pattern_analysis", self._vlm_pattern_analysis_node)
        workflow.add_node("run_vlm_spatial_analysis", self._vlm_spatial_analysis_node)
        workflow.add_node("run_vector_pattern_analysis", self._vector_pattern_analysis_node)
        workflow.add_node("combine_results", self._combine_results_node)
        workflow.add_node("validate_patterns", self._validate_patterns_node)
        
        # 엣지 연결 (병렬 처리 가능한 구조)
        workflow.add_edge(START, "load_page")
        workflow.add_edge("load_page", "extract_vectors")
        workflow.add_edge("load_page", "convert_to_image")
        
        # 병렬 분석
        workflow.add_edge("extract_vectors", "run_vector_pattern_analysis")
        workflow.add_edge("convert_to_image", "run_vlm_pattern_analysis")
        workflow.add_edge("run_vlm_pattern_analysis", "run_vlm_spatial_analysis")
        
        # 결과 통합
        workflow.add_edge("run_vector_pattern_analysis", "combine_results")
        workflow.add_edge("run_vlm_spatial_analysis", "combine_results")
        workflow.add_edge("combine_results", "validate_patterns")
        workflow.add_edge("validate_patterns", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _load_page_node(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """PDF 페이지 로드 노드"""
        logger.info(f"Loading page {state['page_number']} from {state['file_path']}")
        
        step_start = time.time()
        
        try:
            doc = fitz.open(state['file_path'])
            page = doc[state['page_number']]
            
            return {
                "pdf_page": page,
                "status": "page_loaded",
                "processing_steps": [{
                    "step": "load_page",
                    "duration": time.time() - step_start,
                    "status": "success"
                }]
            }
            
        except Exception as e:
            error_msg = f"Failed to load page: {str(e)}"
            logger.error(error_msg)
            
            return {
                "errors": [error_msg],
                "status": "load_failed",
                "processing_steps": [{
                    "step": "load_page",
                    "duration": time.time() - step_start,
                    "status": "failed",
                    "error": error_msg
                }]
            }
    
    def _extract_vectors_node(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """벡터 데이터 추출 노드"""
        logger.info("Extracting vector data")
        
        step_start = time.time()
        
        try:
            page = state.get("pdf_page")
            if not page:
                raise ValueError("No PDF page available")
            
            # 벡터 데이터 추출
            vector_data = self.vector_analyzer.extract_vector_data(page)
            text_data = self.vector_analyzer.extract_text_data(page)
            
            return {
                "vector_data": vector_data,
                "text_data": text_data,
                "processing_steps": [{
                    "step": "extract_vectors",
                    "duration": time.time() - step_start,
                    "status": "success",
                    "details": {
                        "lines_count": len(vector_data.get('lines', [])),
                        "curves_count": len(vector_data.get('curves', [])),
                        "text_blocks": len(text_data)
                    }
                }]
            }
            
        except Exception as e:
            error_msg = f"Vector extraction failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "vector_data": {"lines": [], "curves": [], "rectangles": []},
                "text_data": [],
                "errors": [error_msg],
                "processing_steps": [{
                    "step": "extract_vectors",
                    "duration": time.time() - step_start,
                    "status": "failed",
                    "error": error_msg
                }]
            }
    
    def _convert_to_image_node(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """페이지를 이미지로 변환 노드"""
        logger.info("Converting page to image")
        
        step_start = time.time()
        
        try:
            page = state.get("pdf_page")
            if not page:
                raise ValueError("No PDF page available")
            
            # 이미지 변환 (적당한 해상도로)
            mat = fitz.Matrix(1.5, 1.5)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.pil_tobytes(format="PNG")
            
            image = Image.open(BytesIO(img_data)).convert('RGB')
            
            return {
                "page_image": image,
                "processing_steps": [{
                    "step": "convert_to_image",
                    "duration": time.time() - step_start,
                    "status": "success",
                    "details": {
                        "image_size": image.size,
                        "image_mode": image.mode
                    }
                }]
            }
            
        except Exception as e:
            error_msg = f"Image conversion failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "page_image": None,
                "errors": [error_msg],
                "processing_steps": [{
                    "step": "convert_to_image",
                    "duration": time.time() - step_start,
                    "status": "failed",
                    "error": error_msg
                }]
            }
    
    def _vlm_pattern_analysis_node(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """VLM 패턴 분석 노드"""
        logger.info("Running VLM pattern analysis")
        
        step_start = time.time()
        
        try:
            image = state.get("page_image")
            if not image:
                raise ValueError("No page image available")
            
            # 간단하고 빠른 프롬프트
            prompt = """당신은 건축 도면 분석 전문가입니다.

이 건축 도면에서 다음 요소들을 식별하고 JSON 형식으로만 답하세요:

{
  "walls": [
    {"coordinates": [x1,y1,x2,y2], "thickness": 100, "confidence": 0.9}
  ],
  "doors": [
    {"coordinates": [x,y,width,height], "type": "hinged", "confidence": 0.8}
  ],
  "windows": [
    {"coordinates": [x,y,width,height], "confidence": 0.7}
  ],
  "spaces": [
    {"name": "거실", "boundary": [[x1,y1],[x2,y2]...], "area": 2000, "confidence": 0.8}
  ]
}

좌표는 이미지의 픽셀 좌표로 제공하세요."""
            
            # VLM 분석 실행
            if not self.vlm_analyzer.model:
                self.vlm_analyzer.load_model()
            
            result = self.vlm_analyzer.analyze_image(image, custom_prompt=prompt)
            
            return {
                "vlm_pattern_analysis": result,
                "processing_steps": [{
                    "step": "vlm_pattern_analysis",
                    "duration": time.time() - step_start,
                    "status": "success"
                }]
            }
            
        except Exception as e:
            error_msg = f"VLM pattern analysis failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "vlm_pattern_analysis": {},
                "errors": [error_msg],
                "processing_steps": [{
                    "step": "vlm_pattern_analysis",
                    "duration": time.time() - step_start,
                    "status": "failed",
                    "error": error_msg
                }]
            }
    
    def _vlm_spatial_analysis_node(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """VLM 공간 분석 노드"""
        logger.info("Running VLM spatial analysis")
        
        step_start = time.time()
        
        try:
            image = state.get("page_image")
            if not image:
                raise ValueError("No page image available")
            
            # 공간 관계 분석 프롬프트
            spatial_prompt = """이 건축 도면의 공간 배치와 동선을 분석하여 JSON으로만 답하세요:

{
  "rooms": [
    {
      "name": "거실",
      "area_estimate": "25㎡",
      "connections": ["현관", "주방"],
      "access_points": ["현관문", "베란다문"]
    }
  ],
  "circulation": {
    "main_entrance": "남쪽 하단",
    "corridors": ["현관 복도"],
    "flow_pattern": "현관-거실-주방 중심 동선"
  }
}"""
            
            result = self.vlm_analyzer.analyze_image(image, custom_prompt=spatial_prompt)
            
            return {
                "vlm_spatial_analysis": result,
                "processing_steps": [{
                    "step": "vlm_spatial_analysis", 
                    "duration": time.time() - step_start,
                    "status": "success"
                }]
            }
            
        except Exception as e:
            error_msg = f"VLM spatial analysis failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "vlm_spatial_analysis": {},
                "errors": [error_msg],
                "processing_steps": [{
                    "step": "vlm_spatial_analysis",
                    "duration": time.time() - step_start,
                    "status": "failed",
                    "error": error_msg
                }]
            }
    
    def _vector_pattern_analysis_node(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """벡터 패턴 분석 노드"""
        logger.info("Running vector pattern analysis")
        
        step_start = time.time()
        
        try:
            vector_data = state.get("vector_data", {})
            text_data = state.get("text_data", [])
            page_num = state.get("page_number", 0)
            
            # 벡터 기반 요소 검출
            walls = self.vector_analyzer.detect_walls(vector_data, page_num)
            doors = self.vector_analyzer.detect_doors(vector_data, page_num)
            windows = self.vector_analyzer.detect_windows(vector_data, page_num)
            spaces = self.vector_analyzer.detect_spaces(vector_data, text_data, page_num)
            
            return {
                "vector_walls": [wall.to_dict() for wall in walls],
                "vector_doors": doors,
                "vector_windows": windows,
                "vector_spaces": [space.to_dict() for space in spaces],
                "processing_steps": [{
                    "step": "vector_pattern_analysis",
                    "duration": time.time() - step_start,
                    "status": "success",
                    "details": {
                        "walls_found": len(walls),
                        "doors_found": len(doors),
                        "windows_found": len(windows),
                        "spaces_found": len(spaces)
                    }
                }]
            }
            
        except Exception as e:
            error_msg = f"Vector pattern analysis failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "vector_walls": [],
                "vector_doors": [],
                "vector_windows": [],
                "vector_spaces": [],
                "errors": [error_msg],
                "processing_steps": [{
                    "step": "vector_pattern_analysis",
                    "duration": time.time() - step_start,
                    "status": "failed",
                    "error": error_msg
                }]
            }
    
    def _combine_results_node(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """결과 통합 노드"""
        logger.info("Combining analysis results")
        
        step_start = time.time()
        
        try:
            # VLM 결과 파싱
            vlm_patterns = self._parse_vlm_results(state.get("vlm_pattern_analysis", {}))
            
            # 벡터 결과 가져오기
            vector_results = {
                "walls": state.get("vector_walls", []),
                "doors": state.get("vector_doors", []),
                "windows": state.get("vector_windows", []),
                "spaces": state.get("vector_spaces", [])
            }
            
            # 결과 통합 및 검증
            combined_patterns = self._merge_patterns(vlm_patterns, vector_results)
            
            # 신뢰도 계산
            confidence_scores = self._calculate_confidence_scores(combined_patterns, state)
            
            return {
                "combined_patterns": combined_patterns,
                "confidence_scores": confidence_scores,
                "processing_steps": [{
                    "step": "combine_results",
                    "duration": time.time() - step_start,
                    "status": "success",
                    "details": {
                        "total_patterns": sum(len(patterns) for patterns in combined_patterns.values()),
                        "avg_confidence": sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
                    }
                }]
            }
            
        except Exception as e:
            error_msg = f"Result combination failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "combined_patterns": {"walls": [], "doors": [], "windows": [], "spaces": []},
                "confidence_scores": {},
                "errors": [error_msg],
                "processing_steps": [{
                    "step": "combine_results",
                    "duration": time.time() - step_start,
                    "status": "failed",
                    "error": error_msg
                }]
            }
    
    def _validate_patterns_node(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """패턴 검증 및 최종 분석 노드"""
        logger.info("Validating and finalizing patterns")
        
        step_start = time.time()
        
        try:
            combined_patterns = state.get("combined_patterns", {})
            confidence_scores = state.get("confidence_scores", {})
            
            # 패턴 검증 및 필터링
            validated_patterns = self._validate_and_filter_patterns(combined_patterns)
            
            # 최종 분석 생성
            final_analysis = {
                "summary": {
                    "total_walls": len(validated_patterns.get("walls", [])),
                    "total_doors": len(validated_patterns.get("doors", [])),
                    "total_windows": len(validated_patterns.get("windows", [])),
                    "total_spaces": len(validated_patterns.get("spaces", [])),
                    "overall_confidence": sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
                },
                "patterns": validated_patterns,
                "confidence_scores": confidence_scores,
                "spatial_analysis": state.get("vlm_spatial_analysis", {}),
                "processing_metadata": {
                    "total_steps": len(state.get("processing_steps", [])),
                    "total_time": sum(step.get("duration", 0) for step in state.get("processing_steps", [])),
                    "errors_count": len(state.get("errors", []))
                }
            }
            
            return {
                "final_analysis": final_analysis,
                "status": "completed",
                "processing_steps": [{
                    "step": "validate_patterns",
                    "duration": time.time() - step_start,
                    "status": "success"
                }]
            }
            
        except Exception as e:
            error_msg = f"Pattern validation failed: {str(e)}"
            logger.error(error_msg)
            
            return {
                "final_analysis": {"error": error_msg},
                "status": "validation_failed",
                "errors": [error_msg],
                "processing_steps": [{
                    "step": "validate_patterns",
                    "duration": time.time() - step_start,
                    "status": "failed",
                    "error": error_msg
                }]
            }
    
    def _parse_vlm_results(self, vlm_result: Dict[str, Any]) -> Dict[str, List]:
        """VLM 결과 파싱"""
        try:
            # VLM 결과에서 JSON 추출 시도
            result_text = vlm_result.get("analysis", "") if isinstance(vlm_result, dict) else str(vlm_result)
            
            # JSON 형태의 결과 찾기
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                return {
                    "walls": parsed.get("walls", []),
                    "doors": parsed.get("doors", []),
                    "windows": parsed.get("windows", []),
                    "spaces": parsed.get("spaces", [])
                }
            
        except Exception as e:
            logger.warning(f"Failed to parse VLM results: {e}")
        
        return {"walls": [], "doors": [], "windows": [], "spaces": []}
    
    def _merge_patterns(self, vlm_patterns: Dict[str, List], vector_patterns: Dict[str, List]) -> Dict[str, List]:
        """VLM과 벡터 패턴 결과 병합"""
        merged = {}
        
        for pattern_type in ["walls", "doors", "windows", "spaces"]:
            vlm_items = vlm_patterns.get(pattern_type, [])
            vector_items = vector_patterns.get(pattern_type, [])
            
            # 간단한 병합 (중복 제거 로직 포함 가능)
            all_items = []
            
            # VLM 결과 추가
            for item in vlm_items:
                all_items.append({
                    **item,
                    "source": "vlm",
                    "confidence": item.get("confidence", 0.7)
                })
            
            # 벡터 결과 추가
            for item in vector_items:
                all_items.append({
                    **item,
                    "source": "vector",
                    "confidence": item.get("confidence", 0.8)
                })
            
            merged[pattern_type] = all_items
        
        return merged
    
    def _calculate_confidence_scores(self, patterns: Dict[str, List], state: PatternAnalysisState) -> Dict[str, float]:
        """신뢰도 점수 계산"""
        confidence_scores = {}
        
        for pattern_type, items in patterns.items():
            if items:
                avg_confidence = sum(item.get("confidence", 0) for item in items) / len(items)
                confidence_scores[pattern_type] = avg_confidence
            else:
                confidence_scores[pattern_type] = 0.0
        
        # 전체 처리 신뢰도
        error_count = len(state.get("errors", []))
        step_count = len(state.get("processing_steps", []))
        
        if step_count > 0:
            processing_confidence = max(0, 1.0 - (error_count / step_count))
            confidence_scores["overall_processing"] = processing_confidence
        
        return confidence_scores
    
    def _validate_and_filter_patterns(self, patterns: Dict[str, List]) -> Dict[str, List]:
        """패턴 검증 및 필터링"""
        validated = {}
        
        for pattern_type, items in patterns.items():
            # 신뢰도 기준으로 필터링
            min_confidence = 0.5  # 최소 신뢰도 50%
            
            filtered_items = [
                item for item in items 
                if item.get("confidence", 0) >= min_confidence
            ]
            
            validated[pattern_type] = filtered_items
        
        return validated
    
    def analyze_page(self, file_path: str, page_number: int = 0, 
                    analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """페이지 분석 실행"""
        
        if analysis_options is None:
            analysis_options = {
                "use_vlm": True,
                "use_vector": True,
                "image_dpi": 150
            }
        
        initial_state = {
            "file_path": file_path,
            "page_number": page_number,
            "analysis_options": analysis_options,
            "pdf_page": None,
            "page_image": None,
            "vector_data": {},
            "text_data": [],
            "vlm_pattern_analysis": {},
            "vlm_spatial_analysis": {},
            "vlm_element_analysis": {},
            "vector_walls": [],
            "vector_doors": [],
            "vector_windows": [],
            "vector_spaces": [],
            "combined_patterns": {},
            "confidence_scores": {},
            "final_analysis": {},
            "processing_steps": [],
            "errors": [],
            "status": "initialized"
        }
        
        if HAS_LANGGRAPH and self.workflow:
            # LangGraph 워크플로우 실행
            try:
                config = {"configurable": {"thread_id": f"page_{page_number}"}}
                result = self.workflow.invoke(initial_state, config)
                return result
            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                return {"error": str(e), "status": "workflow_failed"}
        else:
            # Fallback: 순차 실행
            return self._run_fallback_workflow(initial_state)
    
    def _run_fallback_workflow(self, state: PatternAnalysisState) -> Dict[str, Any]:
        """LangGraph 없을 때의 대체 워크플로우"""
        logger.info("Running fallback workflow")
        
        # 순차적으로 각 노드 실행
        state = {**state, **self._load_page_node(state)}
        state = {**state, **self._extract_vectors_node(state)}
        state = {**state, **self._convert_to_image_node(state)}
        state = {**state, **self._vlm_pattern_analysis_node(state)}
        state = {**state, **self._vlm_spatial_analysis_node(state)}
        state = {**state, **self._vector_pattern_analysis_node(state)}
        state = {**state, **self._combine_results_node(state)}
        state = {**state, **self._validate_patterns_node(state)}
        
        return state


def main():
    """테스트용 메인 함수"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 VLM Pattern Workflow Test (Fixed Version)")
    print("=" * 60)
    
    if not HAS_ANALYZERS:
        print("❌ Required analyzers not available")
        return
    
    # 테스트 파일
    test_file = r"C:\Users\user\Documents\VLM\uploads\architectural-plan.pdf"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    # 워크플로우 실행
    try:
        workflow = VLMPatternWorkflow()
        
        start_time = time.time()
        result = workflow.analyze_page(test_file, page_number=0)
        end_time = time.time()
        
        print(f"\n⏱️ Analysis completed in {end_time - start_time:.2f} seconds")
        
        if result.get("status") == "completed":
            final_analysis = result.get("final_analysis", {})
            summary = final_analysis.get("summary", {})
            
            print(f"\n📊 Analysis Summary:")
            print(f"   🧱 Walls: {summary.get('total_walls', 0)}")
            print(f"   🚪 Doors: {summary.get('total_doors', 0)}")
            print(f"   🪟 Windows: {summary.get('total_windows', 0)}")
            print(f"   🏠 Spaces: {summary.get('total_spaces', 0)}")
            print(f"   🎯 Overall Confidence: {summary.get('overall_confidence', 0):.2f}")
            
            # 처리 단계 정보
            processing_steps = result.get("processing_steps", [])
            print(f"\n🔄 Processing Steps: {len(processing_steps)}")
            for step in processing_steps:
                status_icon = "✅" if step.get("status") == "success" else "❌"
                print(f"   {status_icon} {step.get('step')}: {step.get('duration', 0):.2f}s")
        
        else:
            print(f"❌ Analysis failed: {result.get('status')}")
            if result.get("errors"):
                for error in result.get("errors"):
                    print(f"   Error: {error}")
        
        # 결과 저장
        output_file = "vlm_pattern_workflow_fixed_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
