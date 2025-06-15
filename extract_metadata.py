#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
건축 도면 메타데이터 추출 및 프로젝트 분석 도구 (LLM 기반)
- PDF에서 텍스트 추출
- LLM 기반 도면 정보 메타데이터 생성
- 프로젝트 단위 도면 관계 분석
"""

import json
import os
import re
import io
import sys
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF
from pypdf import PdfReader # PyPDF2 대신 pypdf 사용 (최신 권장)

# LLM 메타데이터 추출기 import
sys.path.append(str(Path(__file__).parent / "src"))
try:
    from llm_metadata_extractor import LLMMetadataExtractor
    HAS_LLM_EXTRACTOR = True
except ImportError as e:
    print(f"⚠️  LLM 메타데이터 추출기를 불러올 수 없습니다: {e}")
    print("   정규표현식 기반 추출만 사용합니다.")
    HAS_LLM_EXTRACTOR = False

class ArchitecturalMetadataExtractor:
    def __init__(self, analysis_file="uploads_analysis_results.json", uploads_root_dir="uploads"):
        self.analysis_file = Path(analysis_file) # Path 객체로 변경
        self.uploads_root_dir = Path(uploads_root_dir) # Path 객체로 변경
        # project_metadata는 프로젝트별로 생성되므로, 클래스 멤버에서 제거하고 process_project에서 반환하도록 변경
        
    def load_analysis_results(self):
        """uploads_analysis_results.json 파일 로드"""
        try:
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 분석 결과 파일 로드 실패 ({self.analysis_file}): {e}")
            return None

    def extract_text_from_pdf_page_content(self, page_text_content):
        """
        analyze_uploads_new.py에서 이미 추출된 페이지별 텍스트를 그대로 사용하거나,
        필요시 추가 정제 로직을 여기에 구현할 수 있습니다.
        현재는 입력된 텍스트를 그대로 반환합니다.
        """
        if isinstance(page_text_content, str):
            return page_text_content.strip()
        return ""

    def extract_drawing_metadata_from_text(self, text_content, file_name, page_number, project_path_str):
        """텍스트에서 도면 메타데이터 추출 (VLM 분석 결과 없이)"""
        metadata = {
            "file_name": file_name,
            "page_number": page_number,
            "full_path": str(Path(project_path_str) / file_name), # 파일 전체 경로 추가
            "drawing_number": "근거 부족",
            "drawing_title": "근거 부족", 
            "drawing_type": "근거 부족", # drawing_title과 유사하게 설정될 수 있음
            "scale": "근거 부족",
            "area_info": {},
            "room_list": [],
            "level_info": [], # 층 정보
            "dimensions": [], # 주요 치수
            # "building_info": {}, # 프로젝트 레벨로 이동 가능
            # "project_info": {}, # 프로젝트 레벨로 이동 가능
            "raw_text_snippet": text_content[:500] + "..." if len(text_content) > 500 else text_content, # 미리보기용 텍스트 일부
            "extracted_at": datetime.now().isoformat()
        }
        
        # 정규 표현식 패턴들 (기존 로직 활용, 필요시 개선)
        # 도면 번호 (Drawing Number)
        # 보다 일반적이고 다양한 형식을 포괄하도록 수정
        drawing_number_patterns = [
            r"DWG\\.?\\s*NO\\.?\\s*[:\\s]*([A-Z0-9\\-_./]+)",  # DWG. NO. A-001, DWG NO: X-X-001
            r"도면번호\\s*[:\\s]*([A-Z0-9\\-_./]+)",          # 도면번호: 가-101
            r"SHEET\\s*NO\\.?\\s*[:\\s]*([A-Z0-9\\-_./]+)",    # SHEET NO. A-100
            r"도\\s*면\\s*명\\s*[:\\s]*.*\\(([A-Z0-9\\-_.]+)\\)", # 도 면 명 : XXX 평면도 (A-101)
            r"\\b([A-Z]{1,3}[-\\s]?[0-9]{2,4}[-\\s]?[A-Z0-9]{0,3})\\b", # A-101, AR-001, STR-1001-A (일반적인 도면 번호 형식)
            r"\\b([A-Z]{1,3}[0-9]{2,4})\\b" # A101, AR001
        ]
        for pattern in drawing_number_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                metadata["drawing_number"] = match.group(1).strip() if match.groups() else match.group(0).strip()
                break
        
        # 도면 제목/유형 (Drawing Title/Type)
        # 좀 더 포괄적인 키워드와, 제목으로 간주될 수 있는 라인 탐색
        title_keywords = [
            "평면도", "입면도", "단면도", "배치도", "주단면도", "종단면도", "횡단면도",
            "창호도", "상세도", "계획도", "설계도", "전개도", "구조도", "설비도",
            "전기설비도", "기계설비도", "소방설비도", "통신설비도"
        ]
        # 제목으로 추정되는 라인 (예: "OOO 평면도", "제1종 지구단위계획 결정도")
        # 보통 페이지 상단이나 특정 박스 안에 위치
        lines = text_content.split('\\n')
        found_title = False
        for line in lines[:15]: # 상위 15줄에서 탐색
            for keyword in title_keywords:
                if keyword in line:
                    # 키워드를 포함하는 전체 라인 또는 의미있는 부분을 제목으로
                    # 예: "단위세대 평면도 (TYPE 84A)"
                    # 너무 길지 않게, 주요 정보만 추출
                    title_candidate = line.strip()
                    # 도면번호나 축척 등 다른 정보가 섞여있으면 분리 시도
                    # 여기서는 일단 키워드가 포함된 라인을 사용
                    metadata["drawing_title"] = title_candidate[:100] # 길이 제한
                    metadata["drawing_type"] = keyword # 핵심 키워드를 타입으로
                    found_title = True
                    break
            if found_title:
                break
        
        if not found_title and metadata["drawing_number"] != "근거 부족":
             metadata["drawing_title"] = f"{metadata['drawing_number']} 관련 도면"


        # 축척 (Scale)
        scale_patterns = [
            r"SCALE\s*[:\s]*([0-9.,]+(?:\s*:\s*[0-9.,]+)?(?:\s*@\s*[A-Z0-9]+)?)", # SCALE : 1/100, SCALE : 1:100, 1/200 @ A3
            r"축척\s*[:\s]*([0-9.,]+(?:\s*:\s*[0-9.,]+)?(?:\s*@\s*[A-Z0-9]+)?)",   # 축척 : 1/100
            r"\bS\s*=\s*([0-9.,]+/[0-9.,]+)\b", # S = 1/100
            r"\b(1\s*[:/]\s*[0-9.,]+)\b(?:\s*\(?[A-Z][0-9]\)?)?" # 1:100, 1/100, 1/200 (A3)
        ]
        for pattern in scale_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                scale_str = match.group(1).strip().replace(" ", "")
                metadata["scale"] = scale_str
                break
        
        # 면적 정보 (Area Information) - m2, ㎡, 평 등 단위 고려
        area_patterns = [
            # 전용면적, 공급면적, 계약면적, 기타면적, 발코니, 서비스면적 등
            r"(전용면적|주거전용|전용)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
            r"(공용면적|주거공용|공용)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
            r"(공급면적|분양면적)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
            r"(계약면적)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
            r"(기타공용면적|기타공용)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
            r"(발코니|발코니면적|서비스면적)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
            r"(대지면적)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
            r"(건축면적)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
            r"(연면적|총면적)\s*[:\s]*([0-9.,]+\.?[0-9]*)\s*(㎡|m2|평|M2|M²)",
        ]
        areas = {}
        # 각 면적 패턴을 개별적으로 검사
        for pattern in area_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match in matches:
                if len(match) >= 3:
                    name, val, unit = match[0], match[1], match[2]
                    area_type_map = {
                        "전용면적": "exclusive_area", "주거전용": "exclusive_area", "전용": "exclusive_area",
                        "공용면적": "public_area", "주거공용": "public_area", "공용": "public_area",
                        "공급면적": "supply_area", "분양면적": "supply_area",
                        "계약면적": "contract_area",
                        "기타공용면적": "other_public_area", "기타공용": "other_public_area",
                        "발코니": "balcony_area", "발코니면적": "balcony_area", "서비스면적": "balcony_area",
                        "대지면적": "site_area", "건축면적": "building_area", "연면적": "total_floor_area", "총면적": "total_floor_area"
                    }
                    # 정규화된 키워드 찾기
                    normalized_key = ""
                    for k_map, v_map in area_type_map.items():
                        if k_map in name:
                            normalized_key = v_map
                            break
                    if normalized_key:
                        areas[normalized_key] = {"value": val, "unit": unit}
        
        if areas:
            metadata["area_info"] = areas

        # 공간 목록 (Room List) - 다양한 표현 고려
        room_keywords = [
            "거실", "침실[0-9]*", "안방", "자녀방", "방[0-9]*", "룸",
            "주방", "키친", "식당", "다이닝룸",
            "욕실[0-9]*", "화장실[0-9]*", "샤워실", "파우더룸",
            "현관", "전실", "홀",
            "발코니[0-9]*", "베란다", "테라스",
            "다용도실", "세탁실", "보일러실",
            "드레스룸", "옷방", "붙박이장",
            "팬트리", "창고", "수납공간",
            "서재", "작업실", "스터디룸",
            "알파룸", "맘스오피스", "가족실",
            "대피공간", "실외기실"
        ]
        rooms_found = set()
        for keyword_pattern in room_keywords:
            # 단어 경계(\\b)를 사용하여 정확도 향상
            matches = re.findall(r'\\b(' + keyword_pattern + r')\\b', text_content)
            for room in matches:
                # "침실1", "침실" 과 같은 경우 "침실"로 정규화 (선택적)
                room_normalized = re.sub(r'[0-9]+$', '', room)
                rooms_found.add(room_normalized)
        metadata["room_list"] = sorted(list(rooms_found))

        # 층 정보 (Level Information) - 지하, 지상, 옥탑 등
        level_patterns = [
            r"([ 지하]+[0-9]+층|[0-9]+층|옥탑층|지붕층|PH)", # 지하1층, 1층, 15층, 옥탑층, PH
            r"FL(?:\\s*\\.\\s*|\\s*)([+-]?[0-9,]+\\.?[0-9]*)",  # FL. +1,230, FL -500
            r"EL(?:\\s*\\.\\s*|\\s*)([+-]?[0-9,]+\\.?[0-9]*)",  # EL. +1,230
            r"LEVEL\\s*([0-9A-Za-z]+)" # LEVEL 1, LEVEL B1
        ]
        levels = set()
        for pattern in level_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            for match_item in matches:
                # match_item이 튜플일 수 있음 (여러 그룹 캡처 시)
                level = match_item if isinstance(match_item, str) else match_item[0]
                levels.add(level.strip())
        metadata["level_info"] = sorted(list(levels))

        # 주요 치수 (Dimensions) - 예: 3,500 x 4,200 / W1200 x H2100
        # 좀 더 건축 도면에 특화된 치수 패턴
        dimension_patterns = [
            r"\\b([0-9,]{3,})\\s*X\\s*([0-9,]{3,})\\b",  # 3,500 X 4,200 (공백 허용)
            r"\\bW([0-9,]+)\\s*X\\s*H([0-9,]+)\\b", # W1200 X H2100
            r"\\b[XY](?:=|:)?\\s*([0-9,.]+)\\b", # X=100.50, Y:2500
            # 일반적인 숫자 시퀀스 (너무 많을 수 있어 주의)
            # r"\\b([0-9]{3,5})\\b" # 3자리 이상 5자리 이하 숫자 (단독 치수)
        ]
        dimensions_found = []
        for pattern in dimension_patterns:
            matches = re.findall(pattern, text_content)
            for match_item in matches:
                if isinstance(match_item, tuple) and len(match_item) == 2:
                    dimensions_found.append(f"{match_item[0].strip()}x{match_item[1].strip()}")
                elif isinstance(match_item, str):
                     dimensions_found.append(match_item.strip())
        metadata["dimensions"] = list(set(dimensions_found))[:15] # 중복 제거 및 개수 제한
        
        return metadata

    def analyze_project_relationships(self, drawings_metadata_list):
        """프로젝트 내 도면 관계 분석 (기존 로직 유지 또는 개선)"""
        relationships = []
        if not drawings_metadata_list:
            return relationships

        # 1. 도면 번호 순서에 따른 연관성 (예: A-101, A-102는 시리즈일 가능성)
        # 정렬을 위해 도면 번호를 파싱 가능한 형태로 변환 시도
        def sort_key_dwg_number(dwg):
            num_part = re.findall(r'\\d+', dwg.get("drawing_number", ""))
            if num_part:
                return int(num_part[-1]) # 마지막 숫자 부분을 기준으로 정렬
            return float('inf') # 숫자 없으면 뒤로

        sorted_drawings = sorted(drawings_metadata_list, key=lambda d: (
            re.sub(r'[^A-Z]', '', d.get("drawing_number","ZZZ")), # 알파벳 부분
            sort_key_dwg_number(d) # 숫자 부분
        ))

        for i in range(len(sorted_drawings) - 1):
            d1 = sorted_drawings[i]
            d2 = sorted_drawings[i+1]
            d1_num = d1.get("drawing_number", "N/A")
            d2_num = d2.get("drawing_number", "N/A")

            # 도면 번호 앞부분이 유사하고 뒷부분 숫자만 1 차이 나는 경우
            d1_match = re.match(r'(.*[^0-9])([0-9]+)$', d1_num)
            d2_match = re.match(r'(.*[^0-9])([0-9]+)$', d2_num)

            if d1_match and d2_match:
                if d1_match.group(1) == d2_match.group(1) and int(d2_match.group(2)) == int(d1_match.group(2)) + 1:
                    relationships.append({
                        "type": "sequential_drawings",
                        "drawings": [d1_num, d2_num],
                        "description": f"연속된 도면: {d1_num} -> {d2_num}"
                    })
        
        # 2. 동일 도면 유형 그룹핑 (기존 로직)
        type_groups = {}
        for drawing in drawings_metadata_list:
            drawing_type = drawing.get('drawing_type', 'unknown_type')
            if drawing_type == "근거 부족": drawing_type = "unknown_type"
            if drawing_type not in type_groups:
                type_groups[drawing_type] = []
            type_groups[drawing_type].append(drawing.get("drawing_number", "N/A"))
        
        for type_name, dwg_numbers in type_groups.items():
            if len(dwg_numbers) > 1:
                relationships.append({
                    "type": "same_type_collection",
                    "drawing_type": type_name,
                    "drawings": dwg_numbers,
                    "description": f"동일 유형 도면 그룹: {type_name} ({len(dwg_numbers)}개)"
                })

        # 3. 참조 관계 (예: "SEE DWG A-DETAIL-001") - 텍스트 내용에서 탐색
        ref_pattern = r"(?:참조|SEE|REFER TO)\\s*(?:DWG\\.?|도면)?\\s*([A-Z0-9\\-_./]+)"
        for drawing in drawings_metadata_list:
            text_snippet = drawing.get("raw_text_snippet", "")
            source_dwg_num = drawing.get("drawing_number", "N/A")
            found_refs = re.findall(ref_pattern, text_snippet, re.IGNORECASE)
            for ref_dwg_num in found_refs:
                # 자기 자신을 참조하는 경우는 제외
                if source_dwg_num != ref_dwg_num.strip():
                    relationships.append({
                        "type": "reference",
                        "source_drawing": source_dwg_num,
                        "referenced_drawing": ref_dwg_num.strip(),
                        "description": f"{source_dwg_num}이(가) {ref_dwg_num.strip()}을(를) 참조"
                    })
        return relationships

    def generate_project_summary_info(self, project_name, project_path, drawings_metadata, relationships):
        """프로젝트 전체 요약 정보 생성"""
        summary = {
            "project_name": project_name,
            "project_path": project_path,
            "total_drawings_processed": len(drawings_metadata),
            "drawing_types_summary": {}, # 도면 유형별 개수
            "total_relationships_found": len(relationships),
            "key_drawing_numbers": [], # 주요 도면 번호 (예: 배치도, 기준층 평면도)
            "overall_area_info": {}, # 프로젝트 전체 면적 정보 (가능하다면)
            "extraction_timestamp": datetime.now().isoformat()
        }

        drawing_type_counts = {}
        for drawing in drawings_metadata:
            dtype = drawing.get("drawing_type", "unknown_type")
            if dtype == "근거 부족": dtype = "unknown_type"
            drawing_type_counts[dtype] = drawing_type_counts.get(dtype, 0) + 1
            
            # 주요 도면 번호 후보 (예시: "배치도", "기준층", "평면도" 키워드 포함)
            title = drawing.get("drawing_title", "").lower()
            if any(k in title for k in ["배치도", "site plan"]):
                summary["key_drawing_numbers"].append(f"{drawing.get('drawing_number', 'N/A')} (배치도)")
            elif any(k in title for k in ["기준층", "typical floor"]) and "평면도" in title:
                 summary["key_drawing_numbers"].append(f"{drawing.get('drawing_number', 'N/A')} (기준층 평면도)")
        
        summary["drawing_types_summary"] = drawing_type_counts
        summary["key_drawing_numbers"] = list(set(summary["key_drawing_numbers"]))[:5] # 상위 5개

        # 전체 면적 정보 집계 (예시: 대지면적, 연면적 등 프로젝트 레벨 정보)
        # 도면 메타데이터에서 해당 정보를 찾아 집계 (가장 처음 발견된 값 또는 가장 큰 값 등 정책 필요)
        # 여기서는 간단히 첫번째 "site_area"와 "total_floor_area"를 사용
        for drawing in drawings_metadata:
            if "site_area" in drawing.get("area_info", {}) and "site_area" not in summary["overall_area_info"]:
                summary["overall_area_info"]["site_area"] = drawing["area_info"]["site_area"]
            if "total_floor_area" in drawing.get("area_info", {}) and "total_floor_area" not in summary["overall_area_info"]:
                summary["overall_area_info"]["total_floor_area"] = drawing["area_info"]["total_floor_area"]
            if "site_area" in summary["overall_area_info"] and "total_floor_area" in summary["overall_area_info"]:
                break # 두 정보 모두 찾으면 중단

        return summary

    def process_project(self, project_name, project_data):
        """개별 프로젝트 처리하여 메타데이터 생성"""
        project_path_str = project_data.get("project_path", str(self.uploads_root_dir / project_name))
        project_specific_metadata = {
            "project_info": {}, # 프로젝트 요약 정보가 여기에 들어감
            "drawings": [],
            "relationships": [],
        }
        
        print(f"\\n📄 Processing Project: {project_name} (Path: {project_path_str})")

        # 'pdf_files_text' 키를 사용 (analyze_uploads_new.py의 결과에 맞춤)
        pdf_files_text_data = project_data.get("pdf_files_text", [])
        if not pdf_files_text_data:
            print(f"   ⚠️ No PDF text data found for project {project_name}. Skipping.")
            # 프로젝트 요약만이라도 생성
            project_specific_metadata["project_info"] = self.generate_project_summary_info(
                project_name, project_path_str, [], []
            )
            return project_specific_metadata


        current_project_drawings_metadata = []
        for pdf_file_entry in pdf_files_text_data:
            file_name = pdf_file_entry.get("file")
            if not file_name:
                print("   ⚠️ PDF file entry missing 'file' name. Skipping.")
                continue

            print(f"   📄 Extracting metadata from: {file_name}")
            
            # 'pages_text' 키를 사용 (analyze_uploads_new.py의 결과에 맞춤)
            pages_text_content = pdf_file_entry.get("pages_text", [])
            if 'error' in pdf_file_entry:
                 print(f"      ❌ Error reported for this file in analysis results: {pdf_file_entry['error']}. Skipping pages.")
                 continue

            for page_content_entry in pages_text_content:
                page_num = page_content_entry.get("page")
                text_content = page_content_entry.get("text_content", "")
                
                if page_num is None:
                    print("      ⚠️ Page entry missing 'page' number. Skipping.")
                    continue

                if not text_content and "warning" in page_content_entry:
                    print(f"      ⚠️ Page {page_num}: {page_content_entry['warning']}")
                
                # 텍스트 기반 메타데이터 추출
                # VLM 분석 결과는 사용하지 않으므로 None 전달
                metadata = self.extract_drawing_metadata_from_text(text_content, file_name, page_num, project_path_str)
                current_project_drawings_metadata.append(metadata)
                
                # print(f"         ✅ Dwg No: {metadata['drawing_number']}, Title: {metadata['drawing_title'][:30]}...")

        project_specific_metadata["drawings"] = current_project_drawings_metadata
        
        # 프로젝트 내 도면 관계 분석
        print(f"   🔍 Analyzing relationships for project: {project_name}...")
        relationships = self.analyze_project_relationships(current_project_drawings_metadata)
        project_specific_metadata["relationships"] = relationships
        
        # 프로젝트 요약 생성
        project_summary = self.generate_project_summary_info(
            project_name,
            project_path_str,
            current_project_drawings_metadata,
            relationships
        )
        project_specific_metadata["project_info"] = project_summary
        
        return project_specific_metadata

    def process_all_projects_and_save(self, output_base_filename="project_metadata"):
        """모든 프로젝트를 처리하고, 각 프로젝트별로 메타데이터 파일을 저장"""
        analysis_results_data = self.load_analysis_results()
        if not analysis_results_data:
            print("❌ No analysis results loaded. Cannot proceed.")
            return

        all_projects_output_files = []

        for project_name, project_data_from_analysis in analysis_results_data.items():
            project_metadata_content = self.process_project(project_name, project_data_from_analysis)
            
            if project_metadata_content:
                # 저장 경로: uploads_dir / project_name / project_metadata_project_name.json
                # 또는 별도의 metadata 폴더에 저장할 수도 있음
                # 여기서는 원본 프로젝트 폴더 내에 저장
                
                # project_data_from_analysis["project_path"] 가 실제 프로젝트 폴더의 절대 경로
                # output_dir = Path(project_data_from_analysis.get("project_path", self.uploads_root_dir / project_name))
                
                # 일관성을 위해 self.uploads_root_dir 기준으로 경로 재구성
                # _default_project의 경우 uploads_root_dir 바로 아래
                if project_name == "_default_project":
                    output_dir = self.uploads_root_dir
                else:
                    output_dir = self.uploads_root_dir / project_name
                
                output_dir.mkdir(parents=True, exist_ok=True) # 저장 폴더 생성
                
                # 파일명에 프로젝트 이름 포함 (중복 방지 및 식별 용이)
                # slugify project_name for filename
                safe_project_name = "".join(c if c.isalnum() else "_" for c in project_name)
                output_file_path = output_dir / f"{output_base_filename}_{safe_project_name}.json"

                try:
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(project_metadata_content, f, indent=2, ensure_ascii=False)
                    print(f"   💾 Project metadata saved to: {output_file_path}")
                    all_projects_output_files.append(str(output_file_path))
                except Exception as e:
                    print(f"   ❌ Failed to save metadata for {project_name}: {e}")
            else:
                print(f"   ⚠️ No metadata generated for project {project_name}.")
        
        if all_projects_output_files:
            print(f"\\n🎉 All projects processed. Metadata files created: {len(all_projects_output_files)}")
            for f_path in all_projects_output_files:
                print(f"     - {f_path}")
        else:
            print("\\n⚠️ No metadata files were generated for any project.")
        return all_projects_output_files

def main():
    """메인 함수"""
    print("🏗️  Architectural Project Metadata Extractor (Text-Based)")
    print("=" * 60)
    
    # uploads_analysis_results.json 파일의 위치와 uploads 폴더의 루트를 정확히 지정
    # 이 스크립트가 VLM 폴더에 있다고 가정
    base_dir = Path(__file__).resolve().parent 
    analysis_json_file = base_dir / "uploads_analysis_results.json"
    uploads_folder = base_dir / "uploads"

    if not analysis_json_file.exists():
        print(f"❌ Critical Error: Analysis results file not found at {analysis_json_file}")
        print("   Please run 'analyze_uploads_new.py' first to generate this file.")
        return
        
    extractor = ArchitecturalMetadataExtractor(
        analysis_file=str(analysis_json_file),
        uploads_root_dir=str(uploads_folder)
    )
    
    # 모든 프로젝트 처리 및 저장
    extractor.process_all_projects_and_save()

    print("\\n💡 Next steps:")
    print("   - Review the generated project_metadata_*.json files in each project's subfolder within 'uploads'.")
    print("   - Use these JSON files as input for 'infer_relationships.py' and 'build_rag_db.py'.")

if __name__ == "__main__":
    main()
