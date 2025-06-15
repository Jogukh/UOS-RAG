#!/usr/bin/env python3
"""
LLM 기반 도면 관계 추론기
Qwen2.5-7B-Instruct를 사용하여 건축 도면 간의 의미적 관계를 추론합니다.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import re

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM이 설치되지 않았습니다. pip install vllm으로 설치해주세요.")
    LLM = None
    SamplingParams = None

logger = logging.getLogger(__name__)

class LLMDrawingRelationshipInferencer:
    """LLM을 사용한 건축 도면 관계 추론기"""
    
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
                temperature=0.3,
                top_p=0.8,
                max_tokens=2048,
                repetition_penalty=1.02,
                stop=["<|endoftext|>", "<|im_end|>"]
            )
            
            logger.info(f"LLM 모델 '{self.model_name}' 초기화 완료")
            
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            self.llm = None

    def _create_relationship_prompt(self, drawing1: Dict[str, Any], drawing2: Dict[str, Any]) -> str:
        """두 도면 간의 관계 분석을 위한 프롬프트 생성"""
        
        prompt = f"""<|im_start|>system
당신은 건축 도면 전문가입니다. 두 도면 간의 관계를 분석하여 다음 형식으로 답변해주세요:

관계유형: [참조관계/계층관계/공간관계/시퀀스관계/독립관계 중 하나]
관계강도: [강함/보통/약함 중 하나]  
관계설명: [관계에 대한 간단한 설명]

관계유형 정의:
- 참조관계: 한 도면이 다른 도면을 직접 참조
- 계층관계: 상하층 또는 부모-자식 관계  
- 공간관계: 같은 공간이나 인접 공간
- 시퀀스관계: 설계 단계나 시공 순서상 연관
- 독립관계: 직접적인 관계 없음<|im_end|>

<|im_start|>user
도면1 정보:
- 파일명: {drawing1.get('file_name', 'N/A')}
- 도면번호: {drawing1.get('drawing_number', 'N/A')}  
- 도면제목: {drawing1.get('drawing_title', 'N/A')}
- 도면유형: {drawing1.get('drawing_type', 'N/A')}
- 층 정보: {', '.join(drawing1.get('level_info', []))}
- 공간정보: {', '.join(drawing1.get('room_list', []))}

도면2 정보:
- 파일명: {drawing2.get('file_name', 'N/A')}
- 도면번호: {drawing2.get('drawing_number', 'N/A')}
- 도면제목: {drawing2.get('drawing_title', 'N/A')}  
- 도면유형: {drawing2.get('drawing_type', 'N/A')}
- 층 정보: {', '.join(drawing2.get('level_info', []))}
- 공간정보: {', '.join(drawing2.get('room_list', []))}

이 두 도면 간의 관계를 분석해주세요.<|im_end|>

<|im_start|>assistant
"""
        return prompt

    def _create_text_analysis_prompt(self, drawing_text: str, other_drawings: List[Dict[str, Any]]) -> str:
        """도면의 텍스트 내용을 분석하여 다른 도면과의 참조 관계를 찾는 프롬프트"""
        
        other_drawings_info = []
        for drawing in other_drawings[:10]:  # 최대 10개까지만
            other_drawings_info.append(f"- {drawing.get('drawing_number', 'N/A')}: {drawing.get('drawing_title', 'N/A')}")
        
        prompt = f"""<|im_start|>system
당신은 건축 도면 텍스트 분석 전문가입니다. 주어진 도면의 텍스트 내용에서 다른 도면을 참조하는 부분을 찾아주세요.

답변 형식:
참조도면: [참조되는 도면번호들을 쉼표로 구분]
참조내용: [참조 내용 요약]

참조가 없으면 "참조도면: 없음"으로 답변하세요.<|im_end|>

<|im_start|>user
분석할 도면의 텍스트 내용:
{drawing_text[:2000]}...

프로젝트 내 다른 도면들:
{chr(10).join(other_drawings_info)}

이 텍스트에서 다른 도면을 참조하는 부분을 찾아주세요.<|im_end|>

<|im_start|>assistant
"""
        return prompt

    def analyze_drawing_relationship(self, drawing1: Dict[str, Any], drawing2: Dict[str, Any]) -> Dict[str, Any]:
        """두 도면 간의 관계를 LLM으로 분석"""
        
        if not self.llm:
            return {
                "relationship_type": "unknown",
                "relationship_strength": "unknown", 
                "description": "LLM 초기화 실패"
            }
        
        try:
            prompt = self._create_relationship_prompt(drawing1, drawing2)
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # 응답 파싱
            relationship_type = "독립관계"
            relationship_strength = "약함"
            description = response
            
            # 정규표현식으로 구조화된 응답 파싱
            type_match = re.search(r'관계유형:\s*([^\n]+)', response)
            if type_match:
                relationship_type = type_match.group(1).strip()
                
            strength_match = re.search(r'관계강도:\s*([^\n]+)', response)  
            if strength_match:
                relationship_strength = strength_match.group(1).strip()
                
            desc_match = re.search(r'관계설명:\s*([^\n]+)', response)
            if desc_match:
                description = desc_match.group(1).strip()
            
            return {
                "relationship_type": relationship_type,
                "relationship_strength": relationship_strength,
                "description": description,
                "raw_response": response
            }
            
        except Exception as e:
            logger.error(f"관계 분석 중 오류: {e}")
            return {
                "relationship_type": "unknown",
                "relationship_strength": "unknown",
                "description": f"분석 오류: {str(e)}"
            }

    def find_text_references(self, drawing: Dict[str, Any], other_drawings: List[Dict[str, Any]]) -> List[str]:
        """도면 텍스트에서 다른 도면에 대한 참조를 찾기"""
        
        if not self.llm:
            return []
            
        drawing_text = drawing.get('raw_text_snippet', '')
        if not drawing_text or len(drawing_text.strip()) < 50:
            return []
            
        try:
            prompt = self._create_text_analysis_prompt(drawing_text, other_drawings)
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            
            # 참조도면 파싱
            ref_match = re.search(r'참조도면:\s*([^\n]+)', response)
            if ref_match:
                ref_text = ref_match.group(1).strip()
                if ref_text.lower() != '없음' and ref_text.lower() != 'none':
                    # 쉼표로 구분된 도면번호들 추출
                    referenced_drawings = [d.strip() for d in ref_text.split(',') if d.strip()]
                    return referenced_drawings
                    
            return []
            
        except Exception as e:
            logger.error(f"텍스트 참조 분석 중 오류: {e}")
            return []

    def batch_analyze_relationships(self, drawings: List[Dict[str, Any]], use_text_analysis: bool = True) -> List[Dict[str, Any]]:
        """다수 도면 간의 관계를 배치로 분석"""
        
        relationships = []
        total_pairs = len(drawings) * (len(drawings) - 1) // 2
        processed = 0
        
        print(f"🔍 LLM으로 {total_pairs}개 도면 쌍의 관계를 분석합니다...")
        
        # 1. 도면 쌍별 관계 분석
        for i in range(len(drawings)):
            for j in range(i + 1, len(drawings)):
                drawing1 = drawings[i]
                drawing2 = drawings[j]
                
                # 기본 관계 분석
                relationship = self.analyze_drawing_relationship(drawing1, drawing2)
                
                if relationship["relationship_type"] != "독립관계":
                    relationships.append({
                        "drawing1": drawing1.get("drawing_number", f"drawing_{i}"),
                        "drawing2": drawing2.get("drawing_number", f"drawing_{j}"), 
                        "type": relationship["relationship_type"],
                        "strength": relationship["relationship_strength"],
                        "description": relationship["description"],
                        "method": "llm_semantic"
                    })
                    
                processed += 1
                if processed % 10 == 0:
                    print(f"   진행률: {processed}/{total_pairs} ({processed/total_pairs*100:.1f}%)")
        
        # 2. 텍스트 기반 참조 관계 분석 (선택적)
        if use_text_analysis:
            print("📝 텍스트 기반 참조 관계를 분석합니다...")
            for i, drawing in enumerate(drawings):
                other_drawings = drawings[:i] + drawings[i+1:]
                referenced = self.find_text_references(drawing, other_drawings)
                
                for ref_drawing_num in referenced:
                    # 참조된 도면 찾기
                    for other_drawing in other_drawings:
                        if other_drawing.get("drawing_number") == ref_drawing_num:
                            relationships.append({
                                "drawing1": drawing.get("drawing_number", f"drawing_{i}"),
                                "drawing2": ref_drawing_num,
                                "type": "참조관계", 
                                "strength": "강함",
                                "description": f"텍스트에서 직접 참조",
                                "method": "llm_text_reference"
                            })
                            break
        
        print(f"✅ LLM 기반 관계 분석 완료: {len(relationships)}개 관계 발견")
        return relationships

def test_llm_relationship_inference():
    """LLM 관계 추론 테스트"""
    
    # 테스트용 도면 데이터
    drawing1 = {
        "file_name": "A01-001.pdf",
        "drawing_number": "A01-001", 
        "drawing_title": "1층 평면도",
        "drawing_type": "평면도",
        "level_info": ["1F"],
        "room_list": ["거실", "주방", "침실"]
    }
    
    drawing2 = {
        "file_name": "A01-002.pdf", 
        "drawing_number": "A01-002",
        "drawing_title": "2층 평면도", 
        "drawing_type": "평면도",
        "level_info": ["2F"],
        "room_list": ["침실", "욕실", "발코니"]
    }
    
    try:
        inferencer = LLMDrawingRelationshipInferencer()
        result = inferencer.analyze_drawing_relationship(drawing1, drawing2)
        
        print("🧪 LLM 관계 추론 테스트 결과:")
        print(f"   관계유형: {result['relationship_type']}")
        print(f"   관계강도: {result['relationship_strength']}")  
        print(f"   설명: {result['description']}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

if __name__ == "__main__":
    test_llm_relationship_inference()
