#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 메타데이터 추출기 (PDF + DWG 지원)

모든 메타데이터 추출 기능을 하나의 파일에 통합:
- PDF 처리 (UnstructuredPDFLoader)
- DWG/DXF 처리 
- LLM 기반 메타데이터 추출
- Self-Query 형식 지원
- 프로젝트 폴더명 자동 설정

사용법:
    python extract_metadata_unified.py --project_name="부산장안지구"
    python extract_metadata_unified.py --project_name="부산장안지구" --file_types=pdf
    python extract_metadata_unified.py --project_name="부산장안지구" --file_types=dwg
    python extract_metadata_unified.py --project_name="부산장안지구" --file_types=pdf,dwg
"""

import json
import os
import re
import sys
import time
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# LangChain document loaders
from langchain_community.document_loaders import UnstructuredPDFLoader

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    print("langchain-ollama가 설치되지 않았습니다. pip install langchain-ollama로 설치해주세요.")
    ChatOllama = None
    HAS_OLLAMA = False

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
    from prompt_manager import get_prompt_manager
    # LangSmith 추적 import (optional)
    if HAS_LANGSMITH:
        from langsmith_integration import langsmith_tracker, trace_llm_call
        print(f"✅ LangSmith 추적 활성화: {langsmith_tracker.is_enabled()}")
    HAS_ENV_CONFIG = True
    print("✅ 환경 설정 및 프롬프트 매니저 로드 성공")
except ImportError as e:
    print(f"⚠️  환경 설정을 불러올 수 없습니다: {e}")
    HAS_ENV_CONFIG = False
    def trace_llm_call(name): 
        return lambda x: x

class UnifiedMetadataExtractor:
    """통합 메타데이터 추출기 (PDF + DWG + LLM)"""
    
    def __init__(self, uploads_root_dir="uploads"):
        self.uploads_root_dir = Path(uploads_root_dir)
        
        # 환경 설정 로드
        if HAS_ENV_CONFIG:
            self.env_config = get_env_config()
            self.prompt_manager = get_prompt_manager()
            self.model_name = self.env_config.model_config.model_name
        else:
            self.env_config = None
            self.prompt_manager = None
            self.model_name = "gemma3:12b-it-qat"
        
        # LLM 초기화
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """LLM 초기화 - Multi-LLM Wrapper 사용"""
        # Multi-LLM Wrapper 사용
        try:
            sys.path.append(str(Path(__file__).parent / "src"))
            from multi_llm_wrapper import get_llm
            
            self.llm = get_llm(
                temperature=0.1,  # 메타데이터 추출은 일관성이 중요
                num_predict=1024,  # 메타데이터 추출용으로 충분한 토큰
                timeout=60,  # 타임아웃 설정
            )
            
            print(f"✅ LLM 초기화 완료 - 제공자: {self.llm.get_provider()}, 모델: {self.llm.get_model_name()}")
            
        except Exception as e:
            print(f"❌ LLM 초기화 실패: {e}")
            self.llm = None

    def get_project_files(self, project_name: str) -> tuple[List[Path], List[Path]]:
        """프로젝트 폴더에서 PDF 및 DWG 파일 목록 반환"""
        # 프로젝트 폴더 찾기 (하위 폴더 포함)
        project_paths = []
        
        # 직접 매칭
        direct_path = self.uploads_root_dir / project_name
        if direct_path.exists():
            project_paths.append(direct_path)
        
        # 하위 폴더에서 프로젝트명 포함된 폴더 찾기
        for folder in self.uploads_root_dir.iterdir():
            if folder.is_dir() and project_name in folder.name:
                project_paths.append(folder)
        
        if not project_paths:
            print(f"❌ 프로젝트 폴더를 찾을 수 없습니다: {project_name}")
            return [], []
        
        # 첫 번째 매칭된 폴더 사용
        project_path = project_paths[0]
        print(f"📁 프로젝트 폴더: {project_path}")
        
        # PDF 파일 찾기
        pdf_files = list(project_path.glob("*.pdf")) + list(project_path.glob("*.PDF"))
        
        # DWG/DXF 파일 찾기
        dwg_files = (
            list(project_path.glob("*.dwg")) + 
            list(project_path.glob("*.DWG")) +
            list(project_path.glob("*.dxf")) + 
            list(project_path.glob("*.DXF"))
        )
        
        return pdf_files, dwg_files

    def extract_pdf_with_unstructured(self, pdf_path: Path) -> dict:
        """UnstructuredPDFLoader를 사용한 PDF 추출"""
        print(f"📄 PDF 추출 중: {pdf_path.name}")
        
        start_time = time.time()
        
        try:
            # UnstructuredPDFLoader로 추출
            loader = UnstructuredPDFLoader(
                str(pdf_path), 
                mode="elements",
                strategy="fast"
            )
            docs = loader.load()
            
            print(f"   📄 추출된 요소 수: {len(docs)}")
            
            # 텍스트 결합
            all_text_parts = []
            for doc in docs:
                if hasattr(doc, 'page_content') and doc.page_content.strip():
                    all_text_parts.append(doc.page_content.strip())
            
            combined_text = "\n".join(all_text_parts)
            
            return {
                "success": True,
                "data": {
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    "total_elements": len(docs),
                    "combined_text": combined_text,
                    "text_length": len(combined_text),
                    "extraction_time": time.time() - start_time
                }
            }
            
        except Exception as e:
            print(f"   ❌ PDF 추출 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_metadata_prompt(self, text_content: str, file_name: str) -> str:
        """메타데이터 추출을 위한 프롬프트 생성"""
        
        # 텍스트 길이 제한 (3000자)
        truncated_text = text_content[:3000]
        if len(text_content) > 3000:
            truncated_text += "..."
        
        if self.prompt_manager:
            return self.prompt_manager.format_prompt(
                "metadata_extraction",
                file_name=file_name,
                page_number=1,  # 기본값
                text_content=truncated_text,
                html_content="",  # 기본값 
                tables_data="",  # 기본값
                has_images=False  # 기본값
            )
        else:
            # 기본 프롬프트 (프롬프트 매니저가 없을 때)
            return f"""
건축 PDF 문서에서 메타데이터를 추출해주세요.

파일명: {file_name}
텍스트 내용:
{truncated_text}

다음 JSON 형식으로 응답해주세요:
{{
  "content": "문서 내용 요약",
  "metadata": {{
    "drawing_number": "도면번호 또는 null",
    "drawing_title": "도면 제목",
    "drawing_type": "도면 유형",
    "project_name": "프로젝트명",
    "building_area": "건축면적 (숫자만, 단위 제외)",
    "total_floor_area": "연면적 (숫자만, 단위 제외)",
    "floors_above": "지상층수 (숫자만)",
    "floors_below": "지하층수 (숫자만)",
    "building_height": "건물높이 (숫자만, 단위 제외)"
  }}
}}
"""

    @trace_llm_call(name="Extract PDF Metadata")
    def extract_pdf_metadata_with_llm(self, text_content: str, file_name: str, file_path: str) -> Dict[str, Any]:
        """LLM을 사용한 PDF 메타데이터 추출"""
        
        if not self.llm:
            return self._fallback_metadata(file_name, file_path)
        
        try:
            # 파일 경로에서 프로젝트 이름 추출
            project_name = self._extract_project_name_from_path(file_path)
            
            prompt = self._create_metadata_prompt(text_content, file_name)
            
            print(f"   🤖 LLM 메타데이터 추출 시작: {file_name}")
            print(f"   📋 텍스트 길이: {len(text_content)}자")
            
            # Multi-LLM Wrapper 호출
            response = self.llm.invoke(prompt)
            
            print(f"   🧹 LLM 응답 정리 중...")
            
            # JSON 응답 파싱
            try:
                cleaned_response = self._clean_json_response(response)
                metadata = json.loads(cleaned_response)
                
                # 프로젝트 이름을 폴더명으로 강제 설정
                if "metadata" in metadata:
                    metadata["metadata"]["project_name"] = project_name
                    metadata["metadata"]["file_name"] = file_name
                    metadata["metadata"]["file_path"] = file_path
                    metadata["metadata"]["extracted_at"] = datetime.now().isoformat()
                
                print(f"   ✅ LLM 메타데이터 추출 완료")
                return metadata
                
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON 파싱 실패: {e}")
                print(f"   📋 응답 내용: {response[:200]}...")
                return self._fallback_metadata(file_name, file_path, project_name)
                
        except Exception as e:
            print(f"   ❌ LLM 메타데이터 추출 실패: {e}")
            return self._fallback_metadata(file_name, file_path, project_name)

    def _extract_project_name_from_path(self, file_path: str) -> str:
        """파일 경로에서 프로젝트 이름 추출"""
        project_name = "Unknown"
        try:
            path_parts = Path(file_path).parts
            uploads_idx = -1
            for i, part in enumerate(path_parts):
                if part == "uploads":
                    uploads_idx = i
                    break
            
            if uploads_idx >= 0 and uploads_idx + 1 < len(path_parts):
                project_name = path_parts[uploads_idx + 1]
        except Exception as e:
            print(f"⚠️  프로젝트 이름 추출 실패 ({file_path}): {e}")
        
        return project_name

    def _clean_json_response(self, response: str) -> str:
        """LLM 응답에서 JSON 부분만 추출하여 정리"""
        if not response:
            return ""
            
        # 마크다운 코드 블록 제거
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # JSON 시작과 끝 찾기
        start_brace = response.find("{")
        end_brace = response.rfind("}")
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            response = response[start_brace:end_brace + 1]
        
        return response.strip()

    def _fallback_metadata(self, file_name: str, file_path: str, project_name: str = None) -> Dict[str, Any]:
        """LLM 실패 시 기본 메타데이터 생성"""
        if not project_name:
            project_name = self._extract_project_name_from_path(file_path)
        
        return {
            "content": f"{file_name}에서 추출된 내용",
            "metadata": {
                "drawing_number": None,
                "drawing_title": file_name.replace('.pdf', ''),
                "drawing_type": "기타",
                "project_name": project_name,
                "file_name": file_name,
                "file_path": file_path,
                "extracted_at": datetime.now().isoformat(),
                "building_area": None,
                "total_floor_area": None,
                "floors_above": None,
                "floors_below": None,
                "building_height": None
            }
        }

    def _create_fallback_dwg_metadata(self, file_name: str, file_path: str, project_name: str) -> Dict[str, Any]:
        """DWG 파일 처리 실패 시 Self-Query 호환 fallback 메타데이터 생성"""
        
        # 파일명에서 도면 정보 추출
        drawing_title = file_name.replace('.dwg', '').replace('.DWG', '')
        
        # 파일명 패턴 분석
        drawing_type = "기타"
        if "일람표" in drawing_title:
            drawing_type = "일람표"
        elif "평면도" in drawing_title:
            drawing_type = "평면도"
        elif "입면도" in drawing_title:
            drawing_type = "입면도"
        elif "단면도" in drawing_title:
            drawing_type = "단면도"
        elif "설계개요" in drawing_title:
            drawing_type = "설계개요"
        
        return {
            "content": f"{drawing_title} DWG 도면 파일",
            "metadata": {
                "drawing_number": "정보 없음",
                "drawing_title": drawing_title,
                "drawing_type": drawing_type,
                "drawing_category": "구조도면",
                "project_name": project_name,
                "project_address": "정보 없음",
                "file_name": file_name,
                "file_path": file_path,
                "page_number": 1,
                "has_tables": False,
                "has_images": False,
                "land_area": None,
                "building_area": None,
                "total_floor_area": None,
                "building_height": None,
                "floors_above": 0,
                "floors_below": 0,
                "parking_spaces": 0,
                "apartment_units": 0,
                "building_coverage_ratio": None,
                "floor_area_ratio": None,
                "structure_type": "정보 없음",
                "main_use": "정보 없음",
                "approval_date": None,
                "design_firm": "정보 없음",
                "construction_firm": "정보 없음",
                "room_list": [],
                "extracted_at": datetime.now().isoformat(),
                "extraction_method": "dwg_fallback"
            }
        }

    def extract_dwg_metadata(self, dwg_files: List[Path]) -> List[Dict[str, Any]]:
        """DWG 파일에서 메타데이터 추출"""
        print(f"\n🏗️ DWG 메타데이터 추출 시작 ({len(dwg_files)}개 파일)")
        
        results = []
        
        try:
            # DWG 메타데이터 추출기 import
            from dwg_metadata_extractor import DWGMetadataExtractor
            extractor = DWGMetadataExtractor()
            
            for dwg_file in dwg_files:
                print(f"   처리 중: {dwg_file.name}")
                try:
                    metadata = extractor.extract_from_dwg_file(str(dwg_file))
                    
                    # 프로젝트 이름을 폴더명으로 설정
                    project_name = self._extract_project_name_from_path(str(dwg_file))
                    
                    # metadata가 비어있거나 올바르지 않으면 fallback 생성
                    if not metadata or not isinstance(metadata, dict) or not metadata.get("content"):
                        metadata = self._create_fallback_dwg_metadata(dwg_file.name, str(dwg_file), project_name)
                    elif "metadata" in metadata and "basic_info" in metadata["metadata"]:
                        metadata["metadata"]["basic_info"]["project_name"] = project_name
                    
                    results.append({
                        "file_name": dwg_file.name,
                        "success": True,
                        "metadata": metadata,
                        "file_path": str(dwg_file)
                    })
                    
                    print(f"   ✅ {dwg_file.name} 처리 완료")
                    
                except Exception as e:
                    print(f"   ❌ {dwg_file.name} 처리 실패: {e}")
                    results.append({
                        "file_name": dwg_file.name,
                        "success": False,
                        "error": str(e),
                        "file_path": str(dwg_file)
                    })
            
            print(f"✅ DWG 메타데이터 추출 완료")
            return results
            
        except ImportError as e:
            print(f"❌ DWG 추출기 모듈을 불러올 수 없습니다: {e}")
            return []
        except Exception as e:
            print(f"❌ DWG 메타데이터 추출 실패: {e}")
            return []

    def save_metadata_json(self, metadata: Dict[str, Any], file_path: Path) -> Optional[str]:
        """메타데이터를 JSON 파일로 저장"""
        try:
            output_file = file_path.with_name(f"{file_path.stem}_metadata.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"   💾 메타데이터 저장: {output_file.name}")
            return str(output_file)
            
        except Exception as e:
            print(f"   ❌ 메타데이터 저장 실패: {e}")
            return None

    def process_project(self, project_name: str, file_types: set = {"pdf", "dwg"}) -> Dict[str, Any]:
        """프로젝트의 모든 파일 처리"""
        
        print(f"\n🚀 통합 메타데이터 추출 시작")
        print(f"📁 프로젝트: {project_name}")
        print(f"🔧 처리 형식: {', '.join(file_types)}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 프로젝트 파일 찾기
        pdf_files, dwg_files = self.get_project_files(project_name)
        
        print(f"📄 PDF 파일: {len(pdf_files)}개")
        print(f"🏗️ DWG 파일: {len(dwg_files)}개")
        
        results = {
            "project_name": project_name,
            "processing_time": 0,
            "pdf_results": [],
            "dwg_results": [],
            "summary": {
                "total_files": 0,
                "success_count": 0,
                "error_count": 0
            }
        }
        
        # PDF 처리
        if "pdf" in file_types and pdf_files:
            print(f"\n📄 PDF 파일 처리 시작...")
            
            for pdf_file in pdf_files:
                # PDF 텍스트 추출
                extraction_result = self.extract_pdf_with_unstructured(pdf_file)
                
                if extraction_result["success"]:
                    # LLM으로 메타데이터 추출
                    text_content = extraction_result["data"]["combined_text"]
                    metadata = self.extract_pdf_metadata_with_llm(
                        text_content, pdf_file.name, str(pdf_file)
                    )
                    
                    # 메타데이터 저장
                    json_file = self.save_metadata_json(metadata, pdf_file)
                    
                    results["pdf_results"].append({
                        "file_name": pdf_file.name,
                        "success": True,
                        "metadata": metadata,
                        "json_file": json_file,
                        "text_length": extraction_result["data"]["text_length"]
                    })
                    
                    results["summary"]["success_count"] += 1
                else:
                    results["pdf_results"].append({
                        "file_name": pdf_file.name,
                        "success": False,
                        "error": extraction_result["error"]
                    })
                    
                    results["summary"]["error_count"] += 1
                
                results["summary"]["total_files"] += 1
        
        # DWG 처리
        if "dwg" in file_types and dwg_files:
            dwg_results = self.extract_dwg_metadata(dwg_files)
            results["dwg_results"] = dwg_results
            
            for dwg_result in dwg_results:
                if dwg_result["success"]:
                    # DWG 메타데이터도 JSON으로 저장
                    dwg_file_path = Path(dwg_result["file_path"])
                    self.save_metadata_json(dwg_result["metadata"], dwg_file_path)
                    results["summary"]["success_count"] += 1
                else:
                    results["summary"]["error_count"] += 1
                
                results["summary"]["total_files"] += 1
        
        results["processing_time"] = time.time() - start_time
        
        # 결과 요약
        print(f"\n✅ 통합 메타데이터 추출 완료")
        print(f"⏱️  총 처리 시간: {results['processing_time']:.2f}초")
        print(f"📊 처리 결과: 성공 {results['summary']['success_count']}개, 실패 {results['summary']['error_count']}개")
        
        return results

    def _create_selfquery_conversion_prompt(self, metadata: Dict[str, Any], file_name: str) -> str:
        """기존 메타데이터를 Self-Query 형식으로 변환하는 프롬프트 생성"""
        
        metadata_str = json.dumps(metadata, ensure_ascii=False, indent=2)
        
        if self.prompt_manager:
            # 프롬프트 매니저에서 Self-Query 변환 프롬프트 사용
            try:
                return self.prompt_manager.format_prompt(
                    "convert_to_self_query",
                    file_name=file_name,
                    original_metadata=metadata_str
                )
            except Exception as e:
                print(f"   ⚠️ 프롬프트 매니저 사용 실패: {e}")
                # fallback to default prompt
                pass
        
        # 기본 Self-Query 변환 프롬프트
        return f"""
기존 메타데이터를 Self-Query Retriever 호환 형식으로 변환해주세요.

파일명: {file_name}
기존 메타데이터:
{metadata_str}

다음 Self-Query 형식으로 변환해주세요:
{{
  "page_content": "문서의 주요 내용 요약 (한 문단으로)",
  "metadata": {{
    "drawing_number": "도면번호 (string)",
    "drawing_title": "도면 제목 (string)",
    "drawing_type": "도면 유형 (string)",
    "project_name": "프로젝트명 (string)",
    "project_address": "프로젝트 주소 (string)",
    "file_name": "파일명 (string)",
    "page_number": 페이지 번호 (integer),
    "has_tables": 테이블 포함 여부 (boolean),
    "has_images": 이미지 포함 여부 (boolean),
    "land_area": 대지면적 (float, 숫자만),
    "building_area": 건축면적 (float, 숫자만),
    "total_floor_area": 연면적 (float, 숫자만),
    "building_height": 건물높이 (float, 숫자만),
    "floors_above": 지상층수 (integer),
    "floors_below": 지하층수 (integer),
    "parking_spaces": 주차대수 (integer),
    "apartment_units": 세대수 (integer),
    "building_coverage_ratio": 건폐율 (float, 소수점),
    "floor_area_ratio": 용적률 (float, 소수점),
    "structure_type": "구조형식 (string)",
    "main_use": "주용도 (string)",
    "approval_date": "승인일자 (string, YYYY-MM-DD 형식)",
    "design_firm": "설계사 (string)",
    "construction_firm": "시공사 (string)",
    "room_list": ["방 목록 (array of strings)"],
    "extracted_at": "추출일시 (string, ISO 8601 형식)"
  }}
}}

중요사항:
1. metadata의 모든 값은 검색/필터링 가능한 타입(string, integer, float, boolean)이어야 합니다.
2. 숫자 값은 단위를 제거하고 숫자만 포함해주세요.
3. 정보가 없는 경우 null을 사용해주세요.
4. JSON 형식만 출력하고 다른 설명은 포함하지 마세요.
"""

    @trace_llm_call(name="Convert to Self-Query Format")
    def convert_to_selfquery_format(self, metadata: Dict[str, Any], file_name: str) -> Dict[str, Any]:
        """기존 메타데이터를 Self-Query 형식으로 변환"""
        
        if not self.llm:
            return self._fallback_selfquery_conversion(metadata, file_name)
        
        try:
            prompt = self._create_selfquery_conversion_prompt(metadata, file_name)
            
            print(f"   🔄 Self-Query 형식으로 변환 중: {file_name}")
            
            # Multi-LLM Wrapper 호출
            response = self.llm.invoke(prompt)
            
            # JSON 응답 파싱
            try:
                cleaned_response = self._clean_json_response(response)
                selfquery_metadata = json.loads(cleaned_response)
                
                # 필수 필드 확인 및 추가
                if "metadata" not in selfquery_metadata:
                    selfquery_metadata["metadata"] = {}
                
                # 파일 정보 강제 설정
                selfquery_metadata["metadata"]["file_name"] = file_name
                selfquery_metadata["metadata"]["extracted_at"] = datetime.now().isoformat()
                
                print(f"   ✅ Self-Query 변환 완료")
                return selfquery_metadata
                
            except json.JSONDecodeError as e:
                print(f"   ⚠️ Self-Query 변환 JSON 파싱 실패: {e}")
                return self._fallback_selfquery_conversion(metadata, file_name)
                
        except Exception as e:
            print(f"   ❌ Self-Query 변환 실패: {e}")
            return self._fallback_selfquery_conversion(metadata, file_name)

    def _fallback_selfquery_conversion(self, metadata: Dict[str, Any], file_name: str) -> Dict[str, Any]:
        """Self-Query 변환 실패 시 기본 변환"""
        
        # 기존 메타데이터에서 정보 추출
        original_meta = metadata.get("metadata", {})
        content = metadata.get("content", f"{file_name}에서 추출된 내용")
        
        # Self-Query 형식으로 변환
        selfquery_format = {
            "page_content": content,
            "metadata": {
                "drawing_number": original_meta.get("drawing_number"),
                "drawing_title": original_meta.get("drawing_title", file_name.replace('.pdf', '')),
                "drawing_type": original_meta.get("drawing_type", "기타"),
                "project_name": original_meta.get("project_name", "Unknown"),
                "project_address": original_meta.get("project_address"),
                "file_name": file_name,
                "page_number": 1,
                "has_tables": original_meta.get("has_tables", False),
                "has_images": original_meta.get("has_images", False),
                "land_area": self._convert_to_float(original_meta.get("land_area")),
                "building_area": self._convert_to_float(original_meta.get("building_area")),
                "total_floor_area": self._convert_to_float(original_meta.get("total_floor_area")),
                "building_height": self._convert_to_float(original_meta.get("building_height")),
                "floors_above": self._convert_to_int(original_meta.get("floors_above")),
                "floors_below": self._convert_to_int(original_meta.get("floors_below")),
                "parking_spaces": self._convert_to_int(original_meta.get("parking_spaces")),
                "apartment_units": self._convert_to_int(original_meta.get("apartment_units")),
                "building_coverage_ratio": self._convert_to_float(original_meta.get("building_coverage_ratio")),
                "floor_area_ratio": self._convert_to_float(original_meta.get("floor_area_ratio")),
                "structure_type": original_meta.get("structure_type"),
                "main_use": original_meta.get("main_use"),
                "approval_date": original_meta.get("approval_date"),
                "design_firm": original_meta.get("design_firm"),
                "construction_firm": original_meta.get("construction_firm"),
                "room_list": original_meta.get("room_list", []),
                "extracted_at": datetime.now().isoformat()
            }
        }
        
        return selfquery_format

    def _convert_to_float(self, value) -> Optional[float]:
        """값을 float로 변환 (단위 제거)"""
        if value is None:
            return None
        
        try:
            # 문자열인 경우 숫자만 추출
            if isinstance(value, str):
                # 숫자와 소수점만 추출
                import re
                numbers = re.findall(r'\d+\.?\d*', value.replace(',', ''))
                if numbers:
                    return float(numbers[0])
                return None
            
            return float(value)
        except (ValueError, TypeError):
            return None

    def _convert_to_int(self, value) -> Optional[int]:
        """값을 int로 변환"""
        if value is None:
            return None
        
        try:
            if isinstance(value, str):
                # 숫자만 추출
                import re
                numbers = re.findall(r'\d+', value.replace(',', ''))
                if numbers:
                    return int(numbers[0])
                return None
            
            return int(float(value))  # float를 거쳐서 int로 변환
        except (ValueError, TypeError):
            return None

    def process_existing_metadata_files(self, project_name: str) -> Dict[str, Any]:
        """기존 메타데이터 JSON 파일들을 Self-Query 형식으로 변환"""
        
        print(f"\n🔄 기존 메타데이터 파일을 Self-Query 형식으로 변환 시작")
        print(f"📁 프로젝트: {project_name}")
        print("=" * 60)
        
        start_time = time.time()
        
        # 프로젝트 폴더 찾기
        project_paths = []
        direct_path = self.uploads_root_dir / project_name
        if direct_path.exists():
            project_paths.append(direct_path)
        
        for folder in self.uploads_root_dir.iterdir():
            if folder.is_dir() and project_name in folder.name:
                project_paths.append(folder)
        
        if not project_paths:
            print(f"❌ 프로젝트 폴더를 찾을 수 없습니다: {project_name}")
            return {"error": "프로젝트 폴더 없음"}
        
        project_path = project_paths[0]
        print(f"📁 프로젝트 폴더: {project_path}")
        
        # 메타데이터 폴더 확인
        metadata_folder = project_path / "metadata"
        if not metadata_folder.exists():
            print(f"❌ 메타데이터 폴더가 없습니다: {metadata_folder}")
            # 프로젝트 루트에서 메타데이터 파일 찾기
            metadata_files = list(project_path.glob("*_metadata.json"))
        else:
            metadata_files = list(metadata_folder.glob("*_metadata.json"))
        
        if not metadata_files:
            print(f"❌ 메타데이터 JSON 파일을 찾을 수 없습니다")
            return {"error": "메타데이터 파일 없음"}
        
        print(f"📄 발견된 메타데이터 파일: {len(metadata_files)}개")
        
        results = {
            "project_name": project_name,
            "processing_time": 0,
            "conversion_results": [],
            "summary": {
                "total_files": len(metadata_files),
                "success_count": 0,
                "error_count": 0
            }
        }
        
        # 각 메타데이터 파일 처리
        for metadata_file in metadata_files:
            print(f"🔄 변환 중: {metadata_file.name}")
            
            try:
                # 기존 메타데이터 로드
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    original_metadata = json.load(f)
                
                # Self-Query 형식으로 변환
                selfquery_metadata = self.convert_to_selfquery_format(
                    original_metadata, 
                    metadata_file.name.replace('_metadata.json', '')
                )
                
                # 변환된 메타데이터 저장 (기존 파일 덮어쓰기)
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(selfquery_metadata, f, ensure_ascii=False, indent=2)
                
                print(f"   💾 Self-Query 형식으로 저장 완료: {metadata_file.name}")
                
                results["conversion_results"].append({
                    "file_name": metadata_file.name,
                    "success": True,
                    "original_metadata": original_metadata,
                    "selfquery_metadata": selfquery_metadata
                })
                
                results["summary"]["success_count"] += 1
                
            except Exception as e:
                print(f"   ❌ 변환 실패: {e}")
                results["conversion_results"].append({
                    "file_name": metadata_file.name,
                    "success": False,
                    "error": str(e)
                })
                results["summary"]["error_count"] += 1
        
        results["processing_time"] = time.time() - start_time
        
        print(f"\n✅ Self-Query 형식 변환 완료")
        print(f"⏱️  총 처리 시간: {results['processing_time']:.2f}초")
        print(f"📊 변환 결과: 성공 {results['summary']['success_count']}개, 실패 {results['summary']['error_count']}개")
        
        return results

def main():
    parser = argparse.ArgumentParser(
        description="통합 메타데이터 추출기 (PDF + DWG 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 모든 파일 형식 처리 (기본값)
  python extract_metadata_unified.py --project_name="부산장안지구"
  
  # PDF만 처리
  python extract_metadata_unified.py --project_name="부산장안지구" --file_types=pdf
  
  # DWG만 처리  
  python extract_metadata_unified.py --project_name="부산장안지구" --file_types=dwg
  
  # PDF와 DWG 모두 처리
  python extract_metadata_unified.py --project_name="부산장안지구" --file_types=pdf,dwg
  
  # 기존 메타데이터를 Self-Query 형식으로 변환
  python extract_metadata_unified.py --project_name="부산장안지구" --convert_to_selfquery
        """
    )
    
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="처리할 프로젝트 이름"
    )
    
    parser.add_argument(
        "--file_types",
        type=str,
        default="pdf,dwg",
        help="처리할 파일 형식 (pdf, dwg, 또는 pdf,dwg). 기본값: pdf,dwg"
    )
    
    parser.add_argument(
        "--convert_to_selfquery",
        action="store_true",
        help="기존 메타데이터 JSON 파일들을 Self-Query 형식으로 변환"
    )
    
    args = parser.parse_args()
    
    # 통합 메타데이터 추출기 생성
    extractor = UnifiedMetadataExtractor()
    
    # Self-Query 변환 모드
    if args.convert_to_selfquery:
        results = extractor.process_existing_metadata_files(args.project_name)
        
        # 결과 저장
        output_file = f"selfquery_conversion_{args.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 변환 결과 저장: {output_file}")
        return
    
    # 일반 메타데이터 추출 모드
    # 파일 형식 파싱
    file_types = {ft.strip().lower() for ft in args.file_types.split(",")}
    valid_types = {"pdf", "dwg"}
    
    if not file_types.issubset(valid_types):
        print(f"❌ 잘못된 파일 형식: {file_types - valid_types}")
        print(f"   지원되는 형식: {', '.join(valid_types)}")
        sys.exit(1)
    
    # 통합 메타데이터 추출기 실행
    results = extractor.process_project(args.project_name, file_types)
    
    # 결과 저장
    output_file = f"unified_metadata_extraction_{args.project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 전체 결과 저장: {output_file}")

if __name__ == "__main__":
    main()
