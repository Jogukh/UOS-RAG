#!/usr/bin/env python3
"""
도면 간의 의미 관계를 추론하고 프로젝트별로 그래프를 저장하는 스크립트
LLM(Qwen2.5-7B-Instruct) 기반 관계 추론 + 규칙 기반 추론 결합
.env 파일 기반 설정 사용
"""

import json
import os
import re # 정규표현식 모듈 추가
from pathlib import Path # pathlib 추가
from typing import Dict, List, Any, Tuple
import networkx as nx
import sys

# 로컬 모듈 import
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from llm_relationship_inferencer import LLMDrawingRelationshipInferencer
    from env_config import get_env_config
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  LLM 관계 추론기를 불러올 수 없습니다: {e}")
    print("   규칙 기반 추론만 사용합니다.")
    LLM_AVAILABLE = False

# 기본 업로드 폴더 및 메타데이터 파일 기본 이름
UPLOADS_ROOT_DIR = Path("uploads")
METADATA_BASE_FILENAME = "project_metadata"
RELATIONSHIP_JSON_SUFFIX = "_relationships.json"
GRAPH_GML_SUFFIX = "_graph.gml"

def load_project_metadata_file(metadata_file_path: Path) -> Dict[str, Any]:
    """지정된 경로의 프로젝트 메타데이터 JSON 파일을 로드합니다."""
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"오류: 메타데이터 파일({metadata_file_path})을 찾을 수 없습니다.")
        return {}
    except json.JSONDecodeError:
        print(f"오류: 메타데이터 파일({metadata_file_path})의 형식이 잘못되었습니다.")
        return {}

def infer_relationships_for_project(project_metadata: Dict[str, Any], max_drawings_for_llm: int = 30, use_llm: bool = False) -> nx.Graph:
    """단일 프로젝트의 메타데이터를 기반으로 도면 간의 관계를 추론하여 그래프를 생성합니다."""
    graph = nx.Graph()
    drawings = project_metadata.get("drawings", [])
    project_info = project_metadata.get("project_info", {})
    project_name = project_info.get("project_name", "UnknownProject")

    if not drawings:
        print(f"프로젝트 '{project_name}'에 대한 도면 정보가 메타데이터에 없습니다.")
        return graph

    # 모든 도면을 노드로 추가
    # 노드 ID는 "파일이름_페이지번호" 또는 고유한 "drawing_number"를 사용
    for i, drawing_info in enumerate(drawings):
        # 고유 ID 생성: drawing_number가 유효하면 사용, 아니면 파일명과 페이지번호 조합
        node_id = drawing_info.get("drawing_number")
        if not node_id or node_id == "근거 부족":
            node_id = f"{drawing_info.get('file_name', 'unknown_file')}_p{drawing_info.get('page_number', i)}"
        else:
            # 동일 drawing_number가 여러 페이지에 걸쳐 있을 수 있으므로, 페이지 정보 추가 (선택적)
            # 여기서는 drawing_number가 페이지별로 고유하다고 가정하거나, 대표 페이지만 사용한다고 가정
            # 또는 drawing_number + "_p" + page_number 로 고유화
            page_num_for_id = drawing_info.get('page_number')
            if page_num_for_id:
                 node_id = f"{node_id}_p{page_num_for_id}"

        attributes = {
            "file_name": drawing_info.get("file_name", "N/A"),
            "page_number": drawing_info.get("page_number", "N/A"),
            "drawing_number_raw": drawing_info.get("drawing_number", "N/A"), # 원본 도면번호
            "drawing_title": drawing_info.get("drawing_title", "N/A"),
            "drawing_type": drawing_info.get("drawing_type", "N/A"),
            "level_info": ", ".join(drawing_info.get("level_info", [])), # 리스트를 문자열로
            "project_name": project_name,
            "full_path": drawing_info.get("full_path", "N/A")
            # 필요한 다른 메타데이터 추가
        }
        graph.add_node(node_id, **attributes)

    # 관계 추론 로직 (extract_metadata.py의 analyze_project_relationships와 유사/확장)
    # project_metadata.get("relationships") 에 이미 일부 관계가 있을 수 있음.
    # 여기서는 해당 관계를 그래프에 추가하고, 필요시 더 복잡한 관계 추론 가능.

    # 1. extract_metadata.py에서 생성된 관계 활용
    pre_analyzed_relationships = project_metadata.get("relationships", [])
    for rel in pre_analyzed_relationships:
        rel_type = rel.get("type", "related")
        rel_description = rel.get("description", "")
        involved_drawings = rel.get("drawings", []) # 도면 번호 리스트

        # 도면 번호를 그래프 노드 ID로 매핑해야 함
        # 위에서 노드 ID 생성 규칙과 일치시켜야 함
        # 여기서는 단순화를 위해, involved_drawings의 도면 번호가 노드 ID와 직접 매칭된다고 가정
        # (실제로는 페이지 정보 등을 포함한 ID로 변환 필요)
        
        # 이 예제에서는 pre_analyzed_relationships의 drawing 리스트가 drawing_number라고 가정하고,
        # 그래프 노드 ID도 drawing_number_pPAGE 형식으로 만들었으므로, 매칭을 위한 로직이 필요.
        # 간단히, pre_analyzed_relationships는 여기서 직접 사용하지 않고, 아래에서 새로 추론.

    # 2. 새로운 관계 추론 (기존 infer_drawing_relationships 로직 참고 및 개선)
    drawing_nodes_with_attrs = list(graph.nodes(data=True))

    for i in range(len(drawing_nodes_with_attrs)):
        for j in range(i + 1, len(drawing_nodes_with_attrs)):
            node1_id, attrs1 = drawing_nodes_with_attrs[i]
            node2_id, attrs2 = drawing_nodes_with_attrs[j]

            # A. 동일 파일, 연속 페이지 관계
            if attrs1.get("file_name") == attrs2.get("file_name") and \
               attrs1.get("page_number") is not None and attrs2.get("page_number") is not None:
                page_diff = abs(attrs1["page_number"] - attrs2["page_number"])
                if page_diff == 1:
                    graph.add_edge(node1_id, node2_id, type="consecutive_pages_in_file", 
                                   description=f"{attrs1['file_name']} 내 연속 페이지")
            
            # B. 동일 도면 유형 (drawing_type 기준)
            type1 = attrs1.get("drawing_type", "unknown")
            type2 = attrs2.get("drawing_type", "unknown")
            if type1 != "근거 부족" and type1 != "unknown" and type1 == type2:
                # 너무 많은 연결을 피하기 위해, 같은 파일 내에서는 이미 페이지 연결로 처리될 수 있음
                if attrs1.get("file_name") != attrs2.get("file_name"):
                     graph.add_edge(node1_id, node2_id, type="same_drawing_type", 
                                   description=f"동일 도면 유형: {type1}")

            # C. 층간 연결 (level_info 활용, 평면도 중심)
            # level_info는 문자열이므로 파싱 필요
            levels1_str = attrs1.get("level_info", "")
            levels2_str = attrs2.get("level_info", "")
            # 간단히 첫번째 숫자만 추출 시도 (예: "1층", "지하2층", "FL+3000")
            level_num1_match = re.search(r'(-?[0-9]+)', levels1_str)
            level_num2_match = re.search(r'(-?[0-9]+)', levels2_str)

            is_plan1 = "평면도" in attrs1.get("drawing_title", "").lower() or \
                       "평면도" in attrs1.get("drawing_type", "").lower()
            is_plan2 = "평면도" in attrs2.get("drawing_title", "").lower() or \
                       "평면도" in attrs2.get("drawing_type", "").lower()

            if level_num1_match and level_num2_match and is_plan1 and is_plan2:
                l1 = int(level_num1_match.group(1))
                l2 = int(level_num2_match.group(1))
                if abs(l1 - l2) == 1: # 인접 층 (숫자 기준)
                    graph.add_edge(node1_id, node2_id, type="adjacent_floor_plan",
                                   description=f"인접층 평면도 ({levels1_str} <-> {levels2_str})")
            
            # D. 도면 번호 유사성 (예: A-101, A-102)
            # attrs1["drawing_number_raw"] 사용
            dn1_raw = attrs1.get("drawing_number_raw", "")
            dn2_raw = attrs2.get("drawing_number_raw", "")
            if dn1_raw != "근거 부족" and dn2_raw != "근거 부족" and dn1_raw != dn2_raw:
                # 예: "XX-A-101", "XX-A-102" -> "XX-A-" 부분이 같고, 숫자만 1차이
                match1 = re.match(r'(.*?)([0-9]+)$', dn1_raw)
                match2 = re.match(r'(.*?)([0-9]+)$', dn2_raw)
                if match1 and match2:
                    prefix1, num1_str = match1.groups()
                    prefix2, num2_str = match2.groups()
                    if prefix1 == prefix2 and abs(int(num1_str) - int(num2_str)) == 1:
                        graph.add_edge(node1_id, node2_id, type="sequential_drawing_number",
                                       description=f"연속 도면 번호 ({dn1_raw} <-> {dn2_raw})")

            # E. 텍스트 내용 기반 참조 관계 (raw_text_snippet 활용)
            # 예: "참조: A-201" 또는 "SEE DWG. A-201"
            text1_snippet = project_metadata.get("drawings", [])[i].get("raw_text_snippet", "") # 원본 메타데이터에서 가져옴
            text2_snippet = project_metadata.get("drawings", [])[j].get("raw_text_snippet", "")
            
            # node2_id (또는 그 일부인 drawing_number)가 text1_snippet에 언급되는지
            target_dn2 = attrs2.get("drawing_number_raw", "")
            if target_dn2 != "근거 부족" and target_dn2 in text1_snippet:
                if f"참조" in text1_snippet or f"SEE" in text1_snippet.upper() or f"REFER" in text1_snippet.upper():
                    graph.add_edge(node1_id, node2_id, type="text_reference", 
                                   description=f"{node1_id} -> {node2_id} (텍스트 참조)")
            
            # node1_id가 text2_snippet에 언급되는지
            target_dn1 = attrs1.get("drawing_number_raw", "")
            if target_dn1 != "근거 부족" and target_dn1 in text2_snippet:
                 if f"참조" in text2_snippet or f"SEE" in text2_snippet.upper() or f"REFER" in text2_snippet.upper():
                    graph.add_edge(node2_id, node1_id, type="text_reference", 
                                   description=f"{node2_id} -> {node1_id} (텍스트 참조)")

    # 3. LLM 기반 관계 추론 추가 (선택적)
    if use_llm and LLM_AVAILABLE and len(drawings) <= max_drawings_for_llm:
        try:
            print(f"🤖 LLM으로 '{project_name}' 프로젝트의 의미적 관계를 추론합니다...")
            
            llm_inferencer = LLMDrawingRelationshipInferencer()
            llm_relationships = llm_inferencer.batch_analyze_relationships(drawings, use_text_analysis=True)
            
            # LLM 결과를 그래프에 추가
            added_llm_edges = 0
            for rel in llm_relationships:
                drawing1_num = rel["drawing1"]
                drawing2_num = rel["drawing2"]
                
                # 도면 번호를 노드 ID로 변환 (페이지 정보 포함)
                node1_id = None
                node2_id = None
                
                for node_id, attrs in graph.nodes(data=True):
                    drawing_number_raw = attrs.get("drawing_number_raw", "")
                    if drawing_number_raw == drawing1_num and node1_id is None:
                        node1_id = node_id
                    elif drawing_number_raw == drawing2_num and node2_id is None:
                        node2_id = node_id
                    
                    # 둘 다 찾았으면 더 이상 검색 불필요
                    if node1_id and node2_id:
                        break
                
                if node1_id and node2_id and not graph.has_edge(node1_id, node2_id):
                    # 관계 강도에 따른 가중치 설정
                    weight = {"강함": 1.0, "보통": 0.7, "약함": 0.3}.get(rel["strength"], 0.5)
                    
                    graph.add_edge(node1_id, node2_id, 
                                   type=f"llm_{rel['type']}", 
                                   description=rel["description"],
                                   weight=weight,
                                   method=rel["method"])
                    added_llm_edges += 1
            
            print(f"✅ LLM 기반으로 {added_llm_edges}개의 추가 관계를 그래프에 추가했습니다.")
            
        except Exception as e:
            print(f"⚠️  LLM 관계 추론 중 오류 발생: {e}")
            print("   규칙 기반 관계만 사용합니다.")
    
    elif use_llm and LLM_AVAILABLE and len(drawings) > max_drawings_for_llm:
        print(f"📊 도면 개수({len(drawings)})가 많아 LLM 추론을 스킵합니다. (비용/시간 절약)")

    return graph

def save_graph_data(graph: nx.Graph, output_dir: Path, base_filename: str):
    """추론된 관계 그래프를 JSON (node-link) 및 GML 형식으로 저장합니다."""
    if not graph.nodes():
        print(f"저장할 그래프 정보가 없습니다 (그래프가 비어있음) for {base_filename}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{base_filename}{RELATIONSHIP_JSON_SUFFIX}"
    gml_path = output_dir / f"{base_filename}{GRAPH_GML_SUFFIX}"

    try:
        data = nx.node_link_data(graph)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"   ilişkiler JSON olarak kaydedildi: {json_path}")
    except Exception as e:
        print(f"  JSON 저장 중 오류 ({json_path}): {e}")

    try:
        nx.write_gml(graph, str(gml_path)) # write_gml은 문자열 경로를 받음
        print(f"  그래프 GML로 저장됨: {gml_path}")
    except Exception as e:
        print(f"  GML 저장 중 오류 ({gml_path}): {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="도면 관계 추론 스크립트")
    parser.add_argument("--use-llm", action="store_true", help="LLM 기반 관계 추론 사용 (기본: 비활성화)")
    parser.add_argument("--max-drawings-for-llm", type=int, default=30, help="LLM 사용할 최대 도면 개수 (기본: 30)")
    
    args = parser.parse_args()
    
    # LLM 사용 여부 설정
    use_llm_inference = args.use_llm and LLM_AVAILABLE
    
    if not args.use_llm:
        print("🔧 LLM 관계 추론이 비활성화되었습니다. 규칙 기반 추론만 사용합니다.")
    elif use_llm_inference:
        print("🤖 LLM 관계 추론이 활성화되었습니다.")
    else:
        print("⚠️  LLM 라이브러리를 불러올 수 없어 규칙 기반 추론만 사용합니다.")
    
    print("프로젝트별 도면 관계 추론을 시작합니다...")
    
    processed_projects_count = 0
    # uploads 폴더 및 하위 프로젝트 폴더 순회
    for project_dir_item in UPLOADS_ROOT_DIR.iterdir():
        if project_dir_item.is_dir(): # 각 프로젝트 폴더
            project_name = project_dir_item.name
            # _default_project는 uploads 폴더 자체를 의미할 수 있음
            # extract_metadata.py 저장 규칙에 따라 메타데이터 파일 경로 설정
            safe_project_name_for_file = "".join(c if c.isalnum() else "_" for c in project_name)
            metadata_filename = f"{METADATA_BASE_FILENAME}_{safe_project_name_for_file}.json"
            project_metadata_file_path = project_dir_item / metadata_filename
            
            print(f"\n처리 중인 프로젝트: {project_name} (메타데이터 파일: {project_metadata_file_path})")

            if not project_metadata_file_path.exists():
                # _default_project의 경우, uploads 폴더 바로 아래에 있을 수 있음
                if project_name == "_default_project": # 이 이름은 extract_metadata.py와 일치해야 함
                    metadata_filename_default = f"{METADATA_BASE_FILENAME}__default_project.json"
                    project_metadata_file_path = UPLOADS_ROOT_DIR / metadata_filename_default
                    if not project_metadata_file_path.exists():
                         print(f"  메타데이터 파일을 찾을 수 없습니다: {project_metadata_file_path}. 건너뜁니다.")
                         continue
                else:
                    print(f"  메타데이터 파일을 찾을 수 없습니다: {project_metadata_file_path}. 건너뜁니다.")
                    continue
            
            project_data = load_project_metadata_file(project_metadata_file_path)
            
            if project_data and project_data.get("drawings"):
                relationship_graph = infer_relationships_for_project(project_data, args.max_drawings_for_llm, use_llm_inference)
                if relationship_graph.number_of_nodes() > 0:
                    print(f"  '{project_name}' 프로젝트 그래프 정보:")
                    print(f"    노드 수: {relationship_graph.number_of_nodes()}")
                    print(f"    간선 수: {relationship_graph.number_of_edges()}")
                    
                    # 저장 파일명에 프로젝트 이름 사용 (덮어쓰기 방지 및 구분)
                    # 출력 디렉토리는 해당 프로젝트 메타데이터 파일이 있던 곳과 동일하게
                    output_directory = project_metadata_file_path.parent
                    base_output_filename = f"{project_name}_drawing"
                    save_graph_data(relationship_graph, output_directory, base_output_filename)
                    processed_projects_count += 1
                else:
                    print(f"  '{project_name}' 프로젝트에 대해 추론된 관계가 없거나 노드가 없습니다.")
            else:
                print(f"  '{project_name}' 프로젝트 메타데이터를 로드할 수 없거나 도면 정보가 없습니다.")

    if processed_projects_count > 0:
        print(f"\n총 {processed_projects_count}개 프로젝트의 관계 추론 및 저장이 완료되었습니다.")
    else:
        print("\n처리할 프로젝트 메타데이터를 찾지 못했거나, 관계 추론에 실패했습니다.")
    print("도면 관계 추론 완료.")
