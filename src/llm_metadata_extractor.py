#!/usr/bin/env python3
"""
LLM 기반 건축 도면 메타데이터 추출기
PDF 텍스트에서 건축 도면 메타데이터를 추출합니다.
.env 파일의 설정을 사용하고, Ollama와 LangChain을 연동합니다.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    print("langchain-ollama가 설치되지 않았습니다. pip install langchain-ollama로 설치해주세요.")
    ChatOllama = None
    HAS_OLLAMA = False

# MultiLLMWrapper import
try:
    from .multi_llm_wrapper import MultiLLMWrapper
except ImportError:
    try:
        from multi_llm_wrapper import MultiLLMWrapper
    except ImportError:
        MultiLLMWrapper = None

# 절대 import 또는 sys.path 기반 import
try:
    from prompt_manager import get_prompt_manager
except ImportError:
    import sys
    from pathlib import Path
    current_dir = str(Path(__file__).parent)
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from prompt_manager import get_prompt_manager

# 환경 설정 import
try:
    from env_config import EnvironmentConfig, get_env_str
except ImportError:
    import sys
    from pathlib import Path
    current_dir = str(Path(__file__).parent)
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from env_config import EnvironmentConfig, get_env_str

# LangSmith 추적 import
try:
    from langsmith_integration import trace_llm_call, LangSmithTracker
except ImportError:
    print("⚠️  LangSmith 모듈을 불러올 수 없습니다. 기본 기능으로 대체합니다.")
    def trace_llm_call(name): 
        return lambda x: x
    class LangSmithTracker: 
        def __init__(self):
            pass

logger = logging.getLogger(__name__)

class LLMMetadataExtractor:
    """LLM을 사용한 건축 도면 메타데이터 추출기 (Ollama 연동)"""
    
    def __init__(self, llm_wrapper=None, model_name: str = None, base_url: str = None):
        """
        Args:
            llm_wrapper: MultiLLMWrapper 인스턴스 (권장)
            model_name: 사용할 모델명 (None이면 .env에서 로드) - 호환성용
            base_url: 서버 주소 (Ollama의 경우) - 호환성용
        """
        self.env_config = EnvironmentConfig()
        self.prompt_manager = get_prompt_manager()  # 프롬프트 매니저 추가
        
        # Multi-LLM Wrapper 직접 사용
        if llm_wrapper is not None and hasattr(llm_wrapper, 'invoke'):
            self.llm = llm_wrapper
        else:
            # 기존 호환성을 위한 fallback
            from multi_llm_wrapper import get_llm
            self.llm = get_llm(
                model_name=model_name,
                base_url=base_url,
                temperature=0.1,  # 메타데이터 추출은 일관성이 중요
                num_predict=1024,  # 메타데이터 추출용으로 충분한 토큰
            timeout=60,  # 타임아웃 설정
        )
        
        logger.info(f"LLM 초기화 완료 - 제공자: {self.llm.get_provider()}, 모델: {self.llm.get_model_name()}")
        
    def _initialize_llm(self):
        """레거시 메서드 - Multi-LLM Wrapper로 대체됨"""
        pass

    def _create_metadata_extraction_prompt(self, text_content: str, file_name: str, page_number: int) -> str:
        """메타데이터 추출을 위한 프롬프트 생성 - 중앙 관리 프롬프트 사용"""
        
        # 텍스트 길이 제한 (4000자)
        truncated_text = text_content[:4000]
        if len(text_content) > 4000:
            truncated_text += "..."
        
        return self.prompt_manager.format_prompt(
            "pdf_metadata_extraction",
            file_name=file_name,
            page_number=page_number,
            text_content=truncated_text,
            html_content="",
            tables_data="",
            has_images=False
        )

    def _create_enhanced_metadata_extraction_prompt(self, text_content: str, file_name: str, 
                                                  page_number: int, html_content: str = "", 
                                                  tables_data: List[Dict] = None, 
                                                  has_images: bool = False) -> str:
        """향상된 메타데이터 추출을 위한 프롬프트 생성 - HTML과 표 데이터 포함"""
        
        # 텍스트 길이 제한 (3000자)
        truncated_text = text_content[:3000]
        if len(text_content) > 3000:
            truncated_text += "..."
        
        # HTML 내용 길이 제한 (1000자)
        truncated_html = html_content[:1000] if html_content else ""
        if html_content and len(html_content) > 1000:
            truncated_html += "..."
        
        # 표 데이터 정리 및 제한
        tables_str = ""
        if tables_data:
            tables_list = []
            for i, table in enumerate(tables_data[:5]):  # 최대 5개 표만
                table_content = table.get('content', '')
                if len(table_content) > 500:  # 각 표 내용 500자 제한
                    table_content = table_content[:500] + "..."
                bbox = table.get('bbox', [])
                tables_list.append(f"표 {i+1}: {table_content} (위치: {bbox})")
            tables_str = "\n".join(tables_list)
        
        return self.prompt_manager.format_prompt(
            "pdf_metadata_extraction",
            file_name=file_name,
            page_number=page_number,
            text_content=truncated_text,
            html_content=truncated_html if truncated_html else "HTML 데이터 없음",
            tables_data=tables_str if tables_str else "표 데이터 없음",
            has_images=has_images
        )

    @trace_llm_call(name="Extract Metadata from Text")
    def extract_metadata_from_text(self, text_content: str, file_name: str, page_number: int, 
                                 html_content: str = "", tables_data: List[Dict] = None, 
                                 has_images: bool = False) -> Dict[str, Any]:
        """텍스트, HTML, 표 데이터에서 LLM을 사용하여 메타데이터 추출"""
        
        if not self.llm:
            return self._fallback_regex_extraction(text_content, file_name, page_number)
        
        try:
            prompt = self._create_enhanced_metadata_extraction_prompt(
                text_content, file_name, page_number, html_content, tables_data, has_images
            )
            
            # Multi-LLM Wrapper 호출
            response = self.llm.invoke(prompt)
            
            # JSON 응답 파싱 (마크다운 코드 블록 제거)
            try:
                # 마크다운 코드 블록 제거
                cleaned_response = self._clean_json_response(response)
                metadata = json.loads(cleaned_response)
                
                # 기본 정보 추가
                metadata["file_name"] = file_name
                metadata["page_number"] = page_number
                metadata["raw_text_snippet"] = text_content[:500]  # 처음 500자 저장
                metadata["processing_info"] = {
                    "has_html": bool(html_content),
                    "has_tables": bool(tables_data),
                    "table_count": len(tables_data) if tables_data else 0,
                    "has_images": has_images
                }
                
                # 유효성 검사 및 기본값 설정
                metadata = self._validate_and_clean_metadata(metadata)
                
                return metadata
                
            except json.JSONDecodeError:
                logger.warning(f"LLM 응답을 JSON으로 파싱 실패: {response[:200]}...")
                return self._fallback_regex_extraction(text_content, file_name, page_number)
                
        except Exception as e:
            logger.error(f"LLM 메타데이터 추출 중 오류: {e}")
            return self._fallback_regex_extraction(text_content, file_name, page_number)

    def extract_metadata_from_json_self_query(self, raw_json: Dict[str, Any], file_name: str, page_number: int) -> Dict[str, Any]:
        """
        이미 JSON으로 추출된 원시 데이터를 LLM을 사용해 Self-Query 형식으로 파싱합니다.
        
        Args:
            raw_json: JSON 형식의 원시 메타데이터 데이터
            file_name: 원본 파일 이름
            page_number: 처리된 페이지 번호
        Returns:
            파싱된 메타데이터 사전
        """
        # raw_json을 문자열로 직렬화
        json_str = json.dumps(raw_json, ensure_ascii=False)
        # Self-Query 파싱 프롬프트 생성 (prompt_manager에 'self_query_parsing' 프롬프트 템플릿이 있다고 가정)
        prompt = self.prompt_manager.format_prompt(
            "self_query_parsing",
            json_data=json_str,
            file_name=file_name,
            page_number=page_number
        )
        try:
            # LLM 호출
            response = self.llm.invoke(prompt)
            # JSON만 남기도록 정리
            cleaned = self._clean_json_response(response)
            parsed = json.loads(cleaned)
            # 기본 필드 추가
            parsed["file_name"] = file_name
            parsed["page_number"] = page_number
            # 유효성 검사 및 정리
            return self._validate_and_clean_metadata(parsed)
        except Exception as e:
            logger.warning(f"Self-Query 형식 파싱 실패: {e}")
            # 폴백으로 기존 텍스트 기반 추출 사용
            return self._fallback_regex_extraction(json_str, file_name, page_number)

    def _validate_and_clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 유효성 검사 및 정리"""
        
        # 필수 필드 기본값 설정
        defaults = {
            "drawing_number": "근거 부족",
            "drawing_title": "정보 없음",
            "drawing_type": "기타",
            "scale": "정보 없음",
            "level_info": [],
            "room_list": [],
            "area_info": {},
            "materials": [],
            "dimensions": [],
            "symbols_annotations": []
        }
        
        for key, default_value in defaults.items():
            if key not in metadata or metadata[key] is None:
                metadata[key] = default_value
        
        # 리스트 타입 검증
        list_fields = ["level_info", "room_list", "materials", "dimensions", "symbols_annotations"]
        for field in list_fields:
            if not isinstance(metadata[field], list):
                metadata[field] = []
        
        # 문자열 타입 검증
        string_fields = ["drawing_number", "drawing_title", "drawing_type", "scale"]
        for field in string_fields:
            if not isinstance(metadata[field], str):
                metadata[field] = str(metadata[field]) if metadata[field] else defaults[field]
        
        return metadata

    def _fallback_regex_extraction(self, text_content: str, file_name: str, page_number: int) -> Dict[str, Any]:
        """LLM 실패 시 정규표현식 기반 폴백 추출"""
        
        metadata = {
            "file_name": file_name,
            "page_number": page_number,
            "drawing_number": "근거 부족",
            "drawing_title": "정보 없음",
            "drawing_type": "기타",
            "scale": "정보 없음",
            "level_info": [],
            "room_list": [],
            "area_info": {},
            "materials": [],
            "dimensions": [],
            "symbols_annotations": [],
            "raw_text_snippet": text_content[:500]
        }
        
        # 간단한 정규표현식 기반 추출
        
        # 도면번호 추출
        drawing_number_patterns = [
            r"[A-Z]+\d*[-_]\d+",  # A01-001, B1_002 등
            r"도면\s*번호\s*[:\s]*([A-Z0-9\-_]+)",
            r"DWG\s*NO\s*[:\s]*([A-Z0-9\-_]+)"
        ]
        
        for pattern in drawing_number_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                if len(pattern.split('(')) > 1:  # 그룹이 있는 패턴
                    metadata["drawing_number"] = match.group(1).strip()
                else:
                    metadata["drawing_number"] = match.group(0).strip()
                break
        
        # 축척 추출
        scale_patterns = [
            r"SCALE\s*[:\s]*([0-9/:]+)",
            r"축척\s*[:\s]*([0-9/:]+)",
            r"S\s*=\s*([0-9/:]+)"
        ]
        
        for pattern in scale_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                metadata["scale"] = match.group(1).strip()
                break
        
        return metadata

    def _clean_json_response(self, response: str) -> str:
        """LLM 응답에서 JSON 부분만 추출하여 정리"""
        if not response:
            return ""
            
        # 마크다운 코드 블록 제거
        if "```json" in response:
            # ```json과 ``` 사이의 내용 추출
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            # 일반 코드 블록 제거
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

    def batch_extract_metadata(self, text_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """여러 텍스트에서 배치로 메타데이터 추출"""
        
        results = []
        total = len(text_data_list)
        
        print(f"🤖 LLM으로 {total}개 도면의 메타데이터를 추출합니다...")
        
        for i, text_data in enumerate(text_data_list):
            text_content = text_data.get("text_content", "")
            file_name = text_data.get("file_name", "unknown")
            page_number = text_data.get("page_number", 0)
            
            metadata = self.extract_metadata_from_text(text_content, file_name, page_number)
            results.append(metadata)
            
            if (i + 1) % 50 == 0:
                print(f"   진행률: {i + 1}/{total} ({(i + 1)/total*100:.1f}%)")
        
        print(f"✅ LLM 기반 메타데이터 추출 완료: {len(results)}개 처리")
        return results

def test_llm_metadata_extraction():
    """LLM 메타데이터 추출 테스트"""
    
    # 테스트용 텍스트
    test_text = """
    도면번호: A01-001
    도면제목: 1층 평면도
    축척: 1/100
    
    거실 45.2㎡
    주방 12.5㎡ 
    침실1 15.8㎡
    욕실 4.2㎡
    
    전용면적: 85.2㎡
    공급면적: 102.5㎡
    """
    
    try:
        extractor = LLMMetadataExtractor()
        result = extractor.extract_metadata_from_text(test_text, "test.pdf", 1)
        
        print("🧪 LLM 메타데이터 추출 테스트 결과:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    test_llm_metadata_extraction()
