#!/usr/bin/env python3
"""
LLM 기반 건축 도면 메타데이터 추출기
Qwen2.5-7B-Instruct를 사용하여 PDF 텍스트에서 건축 도면 메타데이터를 추출합니다.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM이 설치되지 않았습니다. pip install vllm으로 설치해주세요.")
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)

class LLMMetadataExtractor:
    """LLM을 사용한 건축 도면 메타데이터 추출기"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Args:
            model_name: 사용할 LLM 모델명
        """
        self.model_name = model_name
        self.llm = None
        self.sampling_params = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """LLM 초기화"""
        if LLM is None:
            logger.error("vLLM이 설치되지 않았습니다.")
            return
            
        try:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.7,
                max_model_len=32768,
                dtype="bfloat16",
                trust_remote_code=True
            )
            
            self.sampling_params = SamplingParams(
                temperature=0.1,  # 메타데이터 추출은 일관성이 중요하므로 낮은 온도
                top_p=0.9,
                max_tokens=1024,
                repetition_penalty=1.02,
                stop=["<|endoftext|>", "<|im_end|>"]
            )
            
            logger.info(f"LLM 모델 '{self.model_name}' 초기화 완료")
            
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            self.llm = None

    def _create_metadata_extraction_prompt(self, text_content: str, file_name: str, page_number: int) -> str:
        """메타데이터 추출을 위한 프롬프트 생성"""
        
        prompt = f"""<|im_start|>system
당신은 건축 도면 메타데이터 추출 전문가입니다. 주어진 PDF 텍스트에서 건축 도면의 메타데이터를 정확하게 추출해주세요.

다음 JSON 형식으로 정확히 답변해주세요:
{{
    "drawing_number": "도면번호 (예: A01-001, 없으면 '근거 부족')",
    "drawing_title": "도면제목 (예: 1층 평면도, 없으면 '정보 없음')",
    "drawing_type": "도면유형 (평면도/입면도/단면도/상세도/배치도/기타 중 하나)",
    "scale": "축척 (예: 1/100, 1:200, 없으면 '정보 없음')",
    "level_info": ["층 정보 배열 (예: ['1F', '2F'], 없으면 [])"],
    "room_list": ["공간 목록 배열 (예: ['거실', '주방'], 없으면 [])"],
    "area_info": {{
        "exclusive_area": {{"value": "면적값", "unit": "단위"}},
        "공급면적": {{"value": "면적값", "unit": "단위"}}
    }},
    "materials": ["재료 목록 (없으면 [])"],
    "dimensions": ["치수 정보 (없으면 [])"],
    "symbols_annotations": ["기호/주석 (없으면 [])"]
}}

JSON만 출력하고 다른 설명은 하지 마세요.<|im_end|>

<|im_start|>user
파일명: {file_name}
페이지: {page_number}

PDF 텍스트 내용:
{text_content[:4000]}...

이 텍스트에서 건축 도면 메타데이터를 추출해주세요.<|im_end|>

<|im_start|>assistant
"""
        return prompt

    def extract_metadata_from_text(self, text_content: str, file_name: str, page_number: int) -> Dict[str, Any]:
        """텍스트에서 LLM을 사용하여 메타데이터 추출"""
        
        if not self.llm:
            return self._fallback_regex_extraction(text_content, file_name, page_number)
        
        try:
            prompt = self._create_metadata_extraction_prompt(text_content, file_name, page_number)
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # JSON 응답 파싱
            try:
                metadata = json.loads(response)
                
                # 기본 정보 추가
                metadata["file_name"] = file_name
                metadata["page_number"] = page_number
                metadata["raw_text_snippet"] = text_content[:500]  # 처음 500자 저장
                
                # 유효성 검사 및 기본값 설정
                metadata = self._validate_and_clean_metadata(metadata)
                
                return metadata
                
            except json.JSONDecodeError:
                logger.warning(f"LLM 응답을 JSON으로 파싱 실패: {response[:200]}...")
                return self._fallback_regex_extraction(text_content, file_name, page_number)
                
        except Exception as e:
            logger.error(f"LLM 메타데이터 추출 중 오류: {e}")
            return self._fallback_regex_extraction(text_content, file_name, page_number)

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
