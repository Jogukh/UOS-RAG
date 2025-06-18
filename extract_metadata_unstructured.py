#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
건축 도면 메타데이터 추출 (UnstructuredPDFLoader 기반)
- UnstructuredPDFLoader를 사용한 PDF 표 중심 텍스트 추출
- 표 구조 인식 및 데이터 구조화
- LLM 기반 표 데이터 분석 및 메타데이터 생성
- 건축 도면 특화 표 데이터 처리
"""

import json
import os
import re
import io
import sys
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain document loaders
from langchain_community.document_loaders import UnstructuredPDFLoader

# LLM 메타데이터 추출기 import
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# LangSmith 추적 설정
try:
    from langsmith import traceable
    HAS_LANGSMITH = True
    print("✅ LangSmith 모듈 로드 성공")
except ImportError as e:
    print(f"⚠️  LangSmith 모듈을 불러올 수 없습니다: {e}")
    HAS_LANGSMITH = False
    # Mock decorator
    def traceable(run_type=None, name=None):
        def decorator(func):
            return func
        return decorator

try:
    # 환경 설정 먼저 import
    from env_config import get_env_config, EnvironmentConfig
    # LLM 메타데이터 추출기 import
    from llm_metadata_extractor import LLMMetadataExtractor
    # LangSmith 추적 import (optional)
    if HAS_LANGSMITH:
        from langsmith_integration import langsmith_tracker, trace_llm_call
        print(f"✅ LangSmith 추적 활성화: {langsmith_tracker.is_enabled()}")
    HAS_LLM_EXTRACTOR = True
    print("✅ LLM 메타데이터 추출기 로드 성공")
except ImportError as e:
    print(f"⚠️  LLM 메타데이터 추출기를 불러올 수 없습니다: {e}")
    print("   정규표현식 기반 추출만 사용합니다.")
    HAS_LLM_EXTRACTOR = False

class UnstructuredTableMetadataExtractor:
    """UnstructuredPDFLoader 기반 표 중심 메타데이터 추출기"""
    
    def __init__(self, uploads_root_dir="uploads"):
        self.uploads_root_dir = Path(uploads_root_dir)
        
        # 환경 설정 로드
        self.env_config = get_env_config() if HAS_LLM_EXTRACTOR else None
        
        # LLM 메타데이터 추출기 초기화
        if HAS_LLM_EXTRACTOR:
            try:
                self.llm_extractor = LLMMetadataExtractor()
                print("✅ LLM 메타데이터 추출기 초기화 완료")
            except Exception as e:
                print(f"⚠️ LLM 메타데이터 추출기 초기화 실패: {e}")
                self.llm_extractor = None
        else:
            self.llm_extractor = None
    
    def extract_pdf_with_table_focus(self, pdf_path: Path) -> dict:
        """표 구조에 집중한 PDF 추출"""
        print(f"� [표 중심 추출] 처리 중: {pdf_path.name}")
        
        start_time = time.time()
        
        try:
            # UnstructuredPDFLoader로 추출 (elements 모드로 구조 유지)
            loader = UnstructuredPDFLoader(
                str(pdf_path), 
                mode="elements",
                strategy="fast"  # 속도 우선
            )
            docs = loader.load()
            
            print(f"   📄 추출된 요소 수: {len(docs)}")
            
            # 요소별로 분류하여 표 데이터 특별 처리
            extracted_data = {
                "file_name": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size": pdf_path.stat().st_size,
                "total_elements": len(docs),
                "extraction_time": time.time() - start_time,
                "elements": [],
                "tables": [],
                "structured_data": {
                    "table_count": 0,
                    "text_blocks": 0,
                    "headers": [],
                    "list_items": []
                },
                "text_summary": {
                    "total_text_length": 0,
                    "categories": {},
                    "all_text": "",
                    "table_text": ""
                }
            }
            
            all_text_parts = []
            table_text_parts = []
            
            for i, doc in enumerate(docs):
                element_data = {
                    "element_id": i,
                    "text_content": doc.page_content.strip(),
                    "text_length": len(doc.page_content.strip()),
                    "category": doc.metadata.get('category', 'Unknown'),
                    "page_number": doc.metadata.get('page_number', 1),
                    "coordinates": doc.metadata.get('coordinates', {}),
                    "metadata": doc.metadata
                }
                
                extracted_data["elements"].append(element_data)
                
                # 카테고리별 분류 및 처리
                category = element_data["category"]
                
                if category == "Table":
                    # 표 데이터 특별 처리
                    table_data = self._process_table_element(element_data)
                    extracted_data["tables"].append(table_data)
                    extracted_data["structured_data"]["table_count"] += 1
                    
                    table_text_parts.append(element_data["text_content"])
                    
                elif category == "Title":
                    extracted_data["structured_data"]["headers"].append({
                        "text": element_data["text_content"],
                        "page": element_data["page_number"],
                        "element_id": i
                    })
                    
                elif category == "ListItem":
                    extracted_data["structured_data"]["list_items"].append({
                        "text": element_data["text_content"],
                        "page": element_data["page_number"],
                        "element_id": i
                    })
                
                elif category in ["NarrativeText", "UncategorizedText"]:
                    extracted_data["structured_data"]["text_blocks"] += 1
                
                # 전체 텍스트에 추가
                if element_data["text_content"]:
                    all_text_parts.append(element_data["text_content"])
                    extracted_data["text_summary"]["categories"][category] = \
                        extracted_data["text_summary"]["categories"].get(category, 0) + 1
            
            # 텍스트 결합
            extracted_data["text_summary"]["all_text"] = "\n".join(all_text_parts)
            extracted_data["text_summary"]["table_text"] = "\n".join(table_text_parts)
            extracted_data["text_summary"]["total_text_length"] = len(extracted_data["text_summary"]["all_text"])
            
            return {
                "success": True,
                "data": extracted_data
            }
            
        except Exception as e:
            extraction_time = time.time() - start_time
            print(f"   ❌ 추출 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "extraction_time": extraction_time
            }
    
    def _process_table_element(self, element_data: dict) -> dict:
        """표 요소 특별 처리"""
        text_content = element_data["text_content"]
        
        # 표 구조 분석
        lines = text_content.split('\n')
        rows = []
        
        for line in lines:
            if line.strip():
                # 탭이나 공백으로 구분된 열 데이터 분석
                if '\t' in line:
                    columns = [col.strip() for col in line.split('\t') if col.strip()]
                else:
                    # 공백 기반 열 분리 (2개 이상의 연속 공백)
                    columns = [col.strip() for col in re.split(r'\s{2,}', line) if col.strip()]
                
                if columns:
                    rows.append(columns)
        
        # 표 메타데이터
        table_metadata = {
            "element_id": element_data["element_id"],
            "page_number": element_data["page_number"],
            "raw_text": text_content,
            "rows": rows,
            "row_count": len(rows),
            "max_columns": max(len(row) for row in rows) if rows else 0,
            "structure_type": self._identify_table_type(rows),
            "keywords": self._extract_table_keywords(text_content)
        }
        
        return table_metadata
    
    def _identify_table_type(self, rows: List[List[str]]) -> str:
        """표 유형 식별"""
        if not rows:
            return "empty"
        
        # 첫 번째 행으로 표 유형 추정
        if len(rows) > 0:
            first_row = ' '.join(rows[0]).lower()
            
            if any(keyword in first_row for keyword in ['도면', '번호', '제목', '도명']):
                return "drawing_list"
            elif any(keyword in first_row for keyword in ['면적', '넓이', 'area']):
                return "area_table"
            elif any(keyword in first_row for keyword in ['재료', '마감', 'material']):
                return "material_table"
            elif any(keyword in first_row for keyword in ['치수', '규격', 'dimension']):
                return "dimension_table"
            elif any(keyword in first_row for keyword in ['층', 'floor', '높이']):
                return "floor_table"
            elif any(keyword in first_row for keyword in ['실', '공간', 'room']):
                return "room_table"
        
        return "general"
    
    def _extract_table_keywords(self, text_content: str) -> List[str]:
        """표에서 중요 키워드 추출"""
        # 건축 관련 키워드
        architectural_keywords = [
            '도면', '평면도', '입면도', '단면도', '상세도', '배치도',
            '면적', '넓이', '규모', '치수', '크기',
            '재료', '마감재', '구조', '콘크리트', '철근', '강재',
            '층', '층고', '높이', '레벨',
            '실', '공간', '용도', '기능',
            '주차', '주차장', '주차면',
            '화장실', '계단', '엘리베이터', '복도',
            '발코니', '테라스', '옥상'
        ]
        
        found_keywords = []
        text_lower = text_content.lower()
        
        for keyword in architectural_keywords:
            if keyword in text_content:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def extract_metadata_from_structured_data(self, extracted_data: dict, file_name: str, file_path: str) -> dict:
        """구조화된 데이터에서 메타데이터 생성"""
        
        if HAS_LLM_EXTRACTOR and self.llm_extractor:
            try:
                # 표 데이터를 포함한 컨텍스트 구성
                table_context = self._build_table_context(extracted_data)
                
                # PDF 메타데이터 추출 전용 프롬프트 사용
                llm_metadata = self._extract_pdf_metadata_with_prompt(
                    extracted_data=extracted_data,
                    file_name=file_name,
                    file_path=file_path
                )
                
                # 표 분석 결과 추가
                if llm_metadata:
                    llm_metadata["table_analysis"] = {
                        "table_count": extracted_data["structured_data"]["table_count"],
                        "table_types": [table["structure_type"] for table in extracted_data["tables"]],
                        "key_tables": self._identify_key_tables(extracted_data["tables"]),
                        "table_summary": table_context
                    }
                
                    return {
                        "metadata_source": "LLM_with_pdf_prompt",
                        "metadata": llm_metadata,
                        "extraction_success": True
                    }
                    
            except Exception as e:
                print(f"   ⚠️ LLM 메타데이터 추출 실패: {e}")
        
        # 표 기반 기본 메타데이터 추출
        table_metadata = self._extract_table_based_metadata(extracted_data, file_name, file_path)
        return {
            "metadata_source": "table_analysis",
            "metadata": table_metadata,
            "extraction_success": True
        }
    
    def _extract_pdf_metadata_with_prompt(self, extracted_data: dict, file_name: str, file_path: str) -> dict:
        """PDF 메타데이터 추출 전용 프롬프트 사용"""
        try:
            # PDF 메타데이터 추출 프롬프트 로드
            pdf_prompt = self.llm_extractor.prompt_manager.get_prompt("pdf_metadata_extraction")
            
            if not pdf_prompt:
                print("   ⚠️ PDF 메타데이터 추출 프롬프트를 찾을 수 없습니다.")
                return None
            
            # 표 데이터를 문자열로 변환
            tables_data_str = ""
            if extracted_data["tables"]:
                tables_data_str = "\n".join([
                    f"[표 {i+1}] {table['structure_type']} (행: {table['row_count']}, 열: {table['max_columns']})\n" +
                    "\n".join([" | ".join(row) for row in table["rows"][:10]])  # 처음 10행만
                    for i, table in enumerate(extracted_data["tables"][:3])  # 처음 3개 표만
                ])
            
            # 프롬프트 변수 구성
            prompt_vars = {
                "file_name": file_name,
                "page_number": 1,
                "text_content": extracted_data["text_summary"]["all_text"][:3000],  # 처음 3000자만
                "html_content": "",  # UnstructuredPDFLoader는 HTML 제공 안함
                "tables_data": tables_data_str,
                "has_images": "False"
            }
            
            # 프롬프트 템플릿 적용 (안전한 문자열 교체 사용)
            formatted_prompt = pdf_prompt.template
            for key, value in prompt_vars.items():
                placeholder = "{" + key + "}"
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))
            
            print(f"   🤖 PDF 메타데이터 추출 프롬프트 사용: {file_name}")
            print(f"   📋 프롬프트 변수: file_name={file_name}, text_length={len(prompt_vars['text_content'])}")
            print(f"   📄 실제 PDF 텍스트 내용:")
            print(f"   {'-'*60}")
            print(f"   {prompt_vars['text_content'][:500]}...")
            print(f"   {'-'*60}")
            print(f"   📊 표 데이터: {prompt_vars['tables_data'][:200] if prompt_vars['tables_data'] else 'None'}...")
            
            # LLM 호출
            response = self.llm_extractor.llm.invoke(formatted_prompt)
            response_text = response.content.strip()
            
            print(f"   🤖 LLM 응답: {response_text[:200]}...")  # 응답 일부 출력
            
            # JSON 응답 파싱
            cleaned_response = self.llm_extractor._clean_json_response(response_text)
            print(f"   🧹 정리된 응답: {cleaned_response[:200]}...")  # 정리된 응답 일부 출력
            
            metadata = json.loads(cleaned_response)
            
            # 필수 필드 확인 및 보완
            current_time = datetime.now().isoformat()
            
            # drawingTitle이 없거나 잘못된 경우 파일명에서 추출
            if ("drawingTitle" not in metadata or 
                not metadata["drawingTitle"] or 
                metadata["drawingTitle"] == "설계개요" and file_name != "설계개요.pdf"):
                # 파일명에서 확장자 제거하고 도면명으로 사용
                base_name = Path(file_name).stem
                metadata["drawingTitle"] = base_name
                print(f"   🔧 도면명 fallback 적용: {base_name}")
            
            # fileName을 실제 파일명으로 강제 수정
            metadata["fileName"] = file_name
            print(f"   📝 파일명 확정: {file_name}")
            
            # extractedAt을 현재 시간으로 업데이트
            metadata["extractedAt"] = current_time
            
            # 기본 정보 추가
            metadata["file_info"] = {
                "file_name": file_name,
                "file_path": file_path,
                "extracted_at": current_time,
                "extraction_method": "pdf_metadata_prompt",
                "prompt_used": "pdf_metadata_extraction"
            }
            
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON 파싱 실패: {e}")
            if 'response_text' in locals():
                print(f"   📋 응답 내용: {response_text[:500]}...")
            return None
        except Exception as e:
            import traceback
            print(f"   ⚠️ PDF 메타데이터 추출 오류: {e}")
            print(f"   📋 상세 오류: {traceback.format_exc()[:500]}...")
            return None
    
    def _build_table_context(self, extracted_data: dict) -> str:
        """표 데이터를 LLM이 이해할 수 있는 컨텍스트로 구성"""
        context_parts = []
        
        if extracted_data["tables"]:
            context_parts.append("=== 발견된 표 데이터 ===")
            
            for i, table in enumerate(extracted_data["tables"], 1):
                context_parts.append(f"\n[표 {i}] {table['structure_type']} (페이지 {table['page_number']})")
                context_parts.append(f"행 수: {table['row_count']}, 열 수: {table['max_columns']}")
                
                if table["keywords"]:
                    context_parts.append(f"키워드: {', '.join(table['keywords'])}")
                
                # 표 내용 (처음 5행만)
                context_parts.append("내용:")
                for j, row in enumerate(table["rows"][:5]):
                    context_parts.append(f"  {j+1}: {' | '.join(row)}")
                
                if len(table["rows"]) > 5:
                    context_parts.append(f"  ... (총 {table['row_count']}행)")
        
        return "\n".join(context_parts)
    
    def _identify_key_tables(self, tables: List[dict]) -> List[dict]:
        """핵심 표 식별"""
        key_tables = []
        
        for table in tables:
            # 중요도 점수 계산
            importance_score = 0
            
            # 행 수 기반 점수
            importance_score += min(table["row_count"] / 10, 3)
            
            # 키워드 기반 점수
            important_keywords = ['도면', '면적', '재료', '층', '실', '주차']
            keyword_score = sum(1 for keyword in important_keywords if keyword in table["keywords"])
            importance_score += keyword_score * 2
            
            # 표 유형 기반 점수
            if table["structure_type"] in ["drawing_list", "area_table", "room_table"]:
                importance_score += 3
            
            if importance_score >= 3:
                key_tables.append({
                    "table": table,
                    "importance_score": importance_score
                })
        
        # 중요도 순으로 정렬
        key_tables.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return key_tables[:5]  # 상위 5개만 반환
    
    
    def _extract_table_based_metadata(self, extracted_data: dict, file_name: str, file_path: str) -> dict:
        """표 기반 메타데이터 추출"""
        
        metadata = {
            "basic_info": {
                "drawing_number": "",
                "drawing_title": "",
                "drawing_type": "",
                "project_name": "",
                "scale": "",
                "date": ""
            },
            "table_analysis": {
                "total_tables": extracted_data["structured_data"]["table_count"],
                "table_types": {},
                "drawing_list": [],
                "area_data": [],
                "material_data": [],
                "room_data": []
            },
            "structural_elements": {
                "headers": extracted_data["structured_data"]["headers"],
                "list_items": len(extracted_data["structured_data"]["list_items"]),
                "text_blocks": extracted_data["structured_data"]["text_blocks"]
            },
            "file_info": {
                "file_name": file_name,
                "file_path": file_path,
                "extracted_at": datetime.now().isoformat(),
                "extraction_method": "table_focused"
            }
        }
        
        # 표별 상세 분석
        for table in extracted_data["tables"]:
            table_type = table["structure_type"]
            metadata["table_analysis"]["table_types"][table_type] = \
                metadata["table_analysis"]["table_types"].get(table_type, 0) + 1
            
            # 표 유형별 데이터 추출
            if table_type == "drawing_list":
                drawing_info = self._extract_drawing_list_data(table)
                metadata["table_analysis"]["drawing_list"].extend(drawing_info)
                
            elif table_type == "area_table":
                area_info = self._extract_area_data(table)
                metadata["table_analysis"]["area_data"].extend(area_info)
                
            elif table_type == "material_table":
                material_info = self._extract_material_data(table)
                metadata["table_analysis"]["material_data"].extend(material_info)
                
            elif table_type == "room_table":
                room_info = self._extract_room_data(table)
                metadata["table_analysis"]["room_data"].extend(room_info)
        
        # 기본 정보 유추
        if metadata["table_analysis"]["drawing_list"]:
            # 도면 목록에서 기본 정보 추출
            first_drawing = metadata["table_analysis"]["drawing_list"][0]
            metadata["basic_info"]["drawing_number"] = first_drawing.get("number", "")
            metadata["basic_info"]["drawing_title"] = first_drawing.get("title", "")
            metadata["basic_info"]["drawing_type"] = first_drawing.get("type", "")
        
        # 프로젝트명 추정 (파일명 또는 헤더에서)
        if extracted_data["structured_data"]["headers"]:
            longest_header = max(extracted_data["structured_data"]["headers"], 
                                key=lambda x: len(x["text"]))
            if len(longest_header["text"]) > 10:
                metadata["basic_info"]["project_name"] = longest_header["text"][:50]
        
        return metadata
    
    def _extract_drawing_list_data(self, table: dict) -> List[dict]:
        """도면 목록 표에서 데이터 추출"""
        drawing_list = []
        
        rows = table["rows"]
        if len(rows) < 2:  # 헤더 + 최소 1개 데이터 행
            return drawing_list
        
        # 헤더 행 분석 (보통 첫 번째 행)
        header_row = rows[0]
        
        # 열 매핑 추정
        number_col = None
        title_col = None
        type_col = None
        
        for i, header in enumerate(header_row):
            header_lower = header.lower()
            if any(keyword in header_lower for keyword in ['번호', 'no', 'number']):
                number_col = i
            elif any(keyword in header_lower for keyword in ['제목', '도면명', 'title', 'name']):
                title_col = i
            elif any(keyword in header_lower for keyword in ['유형', '종류', 'type']):
                type_col = i
        
        # 데이터 행 처리
        for row in rows[1:]:
            if len(row) > max(filter(None, [number_col, title_col, type_col]) or [0]):
                drawing_info = {
                    "number": row[number_col] if number_col is not None and number_col < len(row) else "",
                    "title": row[title_col] if title_col is not None and title_col < len(row) else "",
                    "type": row[type_col] if type_col is not None and type_col < len(row) else "",
                    "raw_row": row
                }
                drawing_list.append(drawing_info)
        
        return drawing_list
    
    def _extract_area_data(self, table: dict) -> List[dict]:
        """면적 표에서 데이터 추출"""
        area_data = []
        
        for row in table["rows"][1:]:  # 헤더 제외
            if len(row) >= 2:
                # 일반적으로 첫 번째 열은 공간명, 두 번째 열은 면적
                area_info = {
                    "space_name": row[0] if row[0] else "",
                    "area_value": row[1] if len(row) > 1 else "",
                    "unit": self._extract_area_unit(row[1] if len(row) > 1 else ""),
                    "raw_row": row
                }
                area_data.append(area_info)
        
        return area_data
    
    def _extract_material_data(self, table: dict) -> List[dict]:
        """재료 표에서 데이터 추출"""
        material_data = []
        
        for row in table["rows"][1:]:  # 헤더 제외
            if row:
                material_info = {
                    "material_name": row[0] if row[0] else "",
                    "specification": row[1] if len(row) > 1 else "",
                    "location": row[2] if len(row) > 2 else "",
                    "raw_row": row
                }
                material_data.append(material_info)
        
        return material_data
    
    def _extract_room_data(self, table: dict) -> List[dict]:
        """실 정보 표에서 데이터 추출"""
        room_data = []
        
        for row in table["rows"][1:]:  # 헤더 제외
            if row:
                room_info = {
                    "room_name": row[0] if row[0] else "",
                    "room_code": row[1] if len(row) > 1 else "",
                    "area": row[2] if len(row) > 2 else "",
                    "usage": row[3] if len(row) > 3 else "",
                    "raw_row": row
                }
                room_data.append(room_info)
        
        return room_data
    
    def _extract_area_unit(self, area_text: str) -> str:
        """면적 단위 추출"""
        units = ['㎡', 'm²', 'm2', '평', '평방미터']
        for unit in units:
            if unit in area_text:
                return unit
        return ""
    
    def process_pdf_file(self, pdf_path: Path, project_dir: Path = None) -> dict:
        """단일 PDF 파일 처리 (표 중심)"""
        
        # 1. 표 중심 PDF 추출
        extraction_result = self.extract_pdf_with_table_focus(pdf_path)
        
        if not extraction_result["success"]:
            return {
                "file_name": pdf_path.name,
                "success": False,
                "error": extraction_result["error"],
                "extraction_time": extraction_result.get("extraction_time", 0)
            }
        
        extracted_data = extraction_result["data"]
        
        # 2. 구조화된 데이터에서 메타데이터 추출
        metadata_result = self.extract_metadata_from_structured_data(
            extracted_data, pdf_path.name, str(pdf_path)
        )
        
        # 3. 메타데이터 JSON 파일 저장 (프로젝트 폴더에)
        json_file_path = None
        if metadata_result["extraction_success"] and project_dir:
            json_file_path = self._save_metadata_json(
                metadata=metadata_result["metadata"],
                pdf_file=pdf_path,
                project_dir=project_dir
            )
        
        # 4. 결과 조합
        return {
            "file_name": pdf_path.name,
            "success": True,
            "extraction_data": {
                "total_elements": extracted_data["total_elements"],
                "total_text_length": extracted_data["text_summary"]["total_text_length"],
                "table_count": extracted_data["structured_data"]["table_count"],
                "categories": extracted_data["text_summary"]["categories"],
                "extraction_time": extracted_data["extraction_time"],
                "structured_summary": {
                    "headers": len(extracted_data["structured_data"]["headers"]),
                    "text_blocks": extracted_data["structured_data"]["text_blocks"],
                    "list_items": len(extracted_data["structured_data"]["list_items"])
                }
            },
            "metadata": metadata_result["metadata"],
            "metadata_source": metadata_result["metadata_source"],
            "tables": extracted_data["tables"][:3],  # 처음 3개 표만 포함 (용량 절약)
            "json_file_path": json_file_path
        }
    
    def _save_metadata_json(self, metadata: dict, pdf_file: Path, project_dir: Path) -> str:
        """메타데이터를 JSON 파일로 프로젝트 폴더에 저장"""
        try:
            # JSON 파일명 생성 (PDF 파일명 기반, 간단하게)
            pdf_name = pdf_file.stem  # 확장자 제외
            json_filename = f"{pdf_name}_metadata.json"
            
            # 프로젝트 디렉토리에 직접 저장 (metadata 폴더 없이)
            json_file_path = project_dir / json_filename
            
            # JSON 저장
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"   💾 메타데이터 저장: {json_file_path}")
            return str(json_file_path)
            
        except Exception as e:
            print(f"   ⚠️ 메타데이터 저장 실패: {e}")
            return None
    
    def process_project_directory(self, project_dir: Path) -> dict:
        """프로젝트 디렉토리 내 모든 PDF 파일 처리 (표 중심)"""
        
        print(f"\n🏗️ 표 중심 프로젝트 처리 시작: {project_dir.name}")
        print(f"📂 경로: {project_dir}")
        
        # PDF 파일 찾기
        pdf_files = []
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(Path(root) / file)
        
        if not pdf_files:
            print("   ⚠️ PDF 파일을 찾을 수 없습니다.")
            return {
                "project_name": project_dir.name,
                "project_path": str(project_dir),
                "total_files": 0,
                "processed_files": [],
                "errors": [],
                "processing_time": 0,
                "table_summary": {
                    "total_tables": 0,
                    "table_types": {},
                    "key_findings": []
                },
                "summary": {
                    "success_rate": 0,
                    "total_text_extracted": 0,
                    "average_extraction_time": 0,
                    "metadata_sources": {"LLM_with_tables": 0, "table_analysis": 0}
                }
            }
        
        print(f"   📄 발견된 PDF 파일 수: {len(pdf_files)}")
        
        start_time = time.time()
        processed_files = []
        errors = []
        table_stats = {"total_tables": 0, "table_types": {}, "key_findings": []}
        
        # 처음 5개 파일 처리 (표 중심 분석)
        test_files = pdf_files[:5]
        
        for i, pdf_file in enumerate(test_files, 1):
            print(f"\n   � [{i}/{len(test_files)}] 표 분석 중: {pdf_file.name}")
            
            try:
                result = self.process_pdf_file(pdf_file, project_dir)
                
                if result["success"]:
                    processed_files.append(result)
                    
                    # 표 통계 업데이트
                    table_count = result["extraction_data"]["table_count"]
                    table_stats["total_tables"] += table_count
                    
                    if "tables" in result:
                        for table in result["tables"]:
                            table_type = table["structure_type"]
                            table_stats["table_types"][table_type] = \
                                table_stats["table_types"].get(table_type, 0) + 1
                    
                    print(f"      ✅ 성공 - 텍스트: {result['extraction_data']['total_text_length']:,}자, "
                          f"표: {table_count}개")
                    
                    # JSON 파일 저장 결과 확인
                    if result.get("json_file_path"):
                        print(f"      💾 JSON 저장됨: {Path(result['json_file_path']).name}")
                else:
                    errors.append(result)
                    print(f"      ❌ 실패: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                error_result = {
                    "file_name": pdf_file.name,
                    "success": False,
                    "error": str(e)
                }
                errors.append(error_result)
                print(f"      ❌ 예외 발생: {e}")
        
        processing_time = time.time() - start_time
        
        # 핵심 발견사항 분석
        key_findings = self._analyze_project_findings(processed_files)
        table_stats["key_findings"] = key_findings
        
        # 프로젝트 요약
        total_text_extracted = sum(f["extraction_data"]["total_text_length"] for f in processed_files)
        avg_extraction_time = sum(f["extraction_data"]["extraction_time"] for f in processed_files) / max(len(processed_files), 1)
        
        return {
            "project_name": project_dir.name,
            "project_path": str(project_dir),
            "total_files": len(pdf_files),
            "processed_files": processed_files,
            "errors": errors,
            "processing_time": processing_time,
            "table_summary": table_stats,
            "summary": {
                "success_rate": len(processed_files) / len(test_files) * 100,
                "total_text_extracted": total_text_extracted,
                "average_extraction_time": avg_extraction_time,
                "metadata_sources": {
                    "LLM_with_tables": sum(1 for f in processed_files if f.get("metadata_source") == "LLM_with_tables"),
                    "table_analysis": sum(1 for f in processed_files if f.get("metadata_source") == "table_analysis")
                }
            }
        }
    
    def _analyze_project_findings(self, processed_files: List[dict]) -> List[str]:
        """프로젝트 차원의 핵심 발견사항 분석"""
        findings = []
        
        if not processed_files:
            return findings
        
        # 총 표 개수
        total_tables = sum(f["extraction_data"]["table_count"] for f in processed_files)
        if total_tables > 0:
            findings.append(f"총 {total_tables}개의 표가 발견되었습니다.")
        
        # 가장 많은 표 유형
        all_table_types = {}
        for file_result in processed_files:
            if "tables" in file_result:
                for table in file_result["tables"]:
                    table_type = table["structure_type"]
                    all_table_types[table_type] = all_table_types.get(table_type, 0) + 1
        
        if all_table_types:
            most_common_type = max(all_table_types, key=all_table_types.get)
            findings.append(f"가장 많은 표 유형: {most_common_type} ({all_table_types[most_common_type]}개)")
        
        # 도면 목록이 있는 파일
        files_with_drawings = 0
        for file_result in processed_files:
            if (file_result.get("metadata", {}).get("table_analysis", {}).get("drawing_list")):
                files_with_drawings += 1
        
        if files_with_drawings > 0:
            findings.append(f"{files_with_drawings}개 파일에서 도면 목록 표를 발견했습니다.")
        
        # 면적 데이터가 있는 파일
        files_with_area = 0
        for file_result in processed_files:
            if (file_result.get("metadata", {}).get("table_analysis", {}).get("area_data")):
                files_with_area += 1
        
        if files_with_area > 0:
            findings.append(f"{files_with_area}개 파일에서 면적 데이터를 발견했습니다.")
        
        return findings

def main():
    """메인 실행 함수"""
    
    print("🚀 표 중심 PDF 메타데이터 추출 시스템 시작")
    print("📊 UnstructuredPDFLoader + 표 구조 분석")
    print("=" * 60)
    
    # 표 중심 추출기 초기화
    extractor = UnstructuredTableMetadataExtractor()
    
    # 프로젝트 디렉토리 찾기
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        print("❌ uploads 디렉토리를 찾을 수 없습니다.")
        return
    
    # 프로젝트 디렉토리들 찾기
    project_dirs = [d for d in uploads_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not project_dirs:
        print("❌ 프로젝트 디렉토리를 찾을 수 없습니다.")
        return
    
    print(f"📂 발견된 프로젝트 수: {len(project_dirs)}")
    
    # PDF가 있는 프로젝트 찾기
    pdf_project = None
    for project_dir in project_dirs:
        pdf_count = sum(1 for root, dirs, files in os.walk(project_dir) 
                       for file in files if file.lower().endswith('.pdf'))
        if pdf_count > 0:
            pdf_project = project_dir
            print(f"📄 선택된 프로젝트: {project_dir.name} ({pdf_count}개 PDF 파일)")
            break
    
    if not pdf_project:
        print("❌ PDF 파일이 있는 프로젝트를 찾을 수 없습니다.")
        return
    
    overall_start_time = time.time()
    
    # 프로젝트 처리 (표 중심)
    project_result = extractor.process_project_directory(pdf_project)
    
    overall_time = time.time() - overall_start_time
    
    # 결과 저장
    output_file = "table_focused_metadata_extraction_results.json"
    result_data = {
        "extraction_info": {
            "timestamp": datetime.now().isoformat(),
            "total_time": overall_time,
            "extractor_type": "UnstructuredPDFLoader_TableFocused",
            "llm_enabled": HAS_LLM_EXTRACTOR and extractor.llm_extractor is not None,
            "features": [
                "표 구조 인식",
                "표 유형 분류",
                "도면 목록 추출",
                "면적 데이터 분석",
                "재료 정보 추출"
            ]
        },
        "project_result": project_result
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    # 결과 요약 출력
    print(f"\n📊 처리 완료 - 총 시간: {overall_time:.2f}초")
    print(f"💾 결과 저장: {output_file}")
    
    summary = project_result["summary"]
    table_summary = project_result["table_summary"]
    
    print(f"\n✅ 처리 결과:")
    print(f"  성공률: {summary['success_rate']:.1f}%")
    print(f"  총 추출 텍스트: {summary['total_text_extracted']:,}자")
    print(f"  평균 추출 시간: {summary['average_extraction_time']:.2f}초")
    print(f"  메타데이터 소스: LLM+표 {summary['metadata_sources']['LLM_with_tables']}개, 표분석 {summary['metadata_sources']['table_analysis']}개")
    
    print(f"\n📊 표 분석 결과:")
    print(f"  총 표 개수: {table_summary['total_tables']}개")
    print(f"  표 유형 분포: {table_summary['table_types']}")
    
    if table_summary["key_findings"]:
        print(f"\n🔍 핵심 발견사항:")
        for finding in table_summary["key_findings"]:
            print(f"  • {finding}")
    
    # 표 중심 분석이 완료되었으므로, 벡터 DB 연동 제안
    print(f"\n🚀 다음 단계 제안:")
    print(f"  1. 추출된 표 데이터를 벡터 DB에 임베딩")
    print(f"  2. 표 구조를 유지한 RAG 시스템 구축")
    print(f"  3. 표 데이터 기반 질의응답 최적화")

if __name__ == "__main__":
    main()
