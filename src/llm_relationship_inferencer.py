#!/usr/bin/env python3
"""
LLM 기반 도면 관계 추론기
.env 파일의 설정을 사용하여 건축 도면 간의 의미적 관계를 추론합니다.
Ollama와 LangChain을 연동합니다.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import re

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    print("langchain-ollama가 설치되지 않았습니다. pip install langchain-ollama로 설치해주세요.")
    ChatOllama = None
    HAS_OLLAMA = False

from prompt_manager import get_prompt_manager

from .env_config import EnvironmentConfig, get_env_str
from .langsmith_integration import trace_llm_call, LangSmithTracker

logger = logging.getLogger(__name__)

class LLMDrawingRelationshipInferencer:
    """LLM을 사용한 건축 도면 관계 추론기 (Ollama 연동)"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        """
        Args:
            model_name: 사용할 Ollama 모델명 (None이면 .env에서 로드)
            base_url: Ollama 서버 주소 (기본값: http://localhost:11434)
        """
        self.env_config = EnvironmentConfig()
        self.model_name = model_name or self.env_config.model_config.model_name
        self.prompt_manager = get_prompt_manager()  # 프롬프트 매니저 추가
        
        # Ollama 서버 설정
        self.base_url = base_url or get_env_str('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        self.llm = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """LLM 초기화 - Ollama 서버와 연결"""
        if not HAS_OLLAMA:
            logger.error("langchain-ollama가 설치되지 않았습니다.")
            return
            
        try:
            # LangSmith 추적 설정
            self.langsmith_tracker = LangSmithTracker()
            
            # Ollama ChatOllama 연결
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0.3,  # 관계 추론은 약간의 창의성 필요
                num_predict=2048,  # 관계 설명을 위한 충분한 토큰
                timeout=60,  # 타임아웃 설정
            )
            
            logger.info(f"LLM 모델 '{self.model_name}' Ollama로 초기화 완료 (서버: {self.base_url})")
            
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            self.llm = None

    def _create_relationship_prompt(self, drawing1: Dict[str, Any], drawing2: Dict[str, Any]) -> str:
        """두 도면 간의 관계 분석을 위한 프롬프트 생성 - 중앙 관리 프롬프트 사용"""
        
        return self.prompt_manager.format_prompt(
            "relationship_inference",
            drawing1_file_name=drawing1.get('file_name', 'N/A'),
            drawing1_number=drawing1.get('drawing_number', 'N/A'),
            drawing1_title=drawing1.get('drawing_title', 'N/A'),
            drawing1_type=drawing1.get('drawing_type', 'N/A'),
            drawing1_levels=', '.join(drawing1.get('level_info', [])),
            drawing1_rooms=', '.join(drawing1.get('room_list', [])),
            drawing2_file_name=drawing2.get('file_name', 'N/A'),
            drawing2_number=drawing2.get('drawing_number', 'N/A'),
            drawing2_title=drawing2.get('drawing_title', 'N/A'),
            drawing2_type=drawing2.get('drawing_type', 'N/A'),
            drawing2_levels=', '.join(drawing2.get('level_info', [])),
            drawing2_rooms=', '.join(drawing2.get('room_list', []))
        )

    def _create_text_analysis_prompt(self, drawing_text: str, other_drawings: List[Dict[str, Any]]) -> str:
        """도면의 텍스트 내용을 분석하여 다른 도면과의 참조 관계를 찾는 프롬프트 - 중앙 관리 프롬프트 사용"""
        
        # 다른 도면 정보 포맷팅
        other_drawings_info = []
        for drawing in other_drawings[:10]:  # 최대 10개까지만
            other_drawings_info.append(f"- {drawing.get('drawing_number', 'N/A')}: {drawing.get('drawing_title', 'N/A')}")
        
        # 텍스트 길이 제한 (2000자)
        truncated_text = drawing_text[:2000]
        if len(drawing_text) > 2000:
            truncated_text += "..."
        
        return self.prompt_manager.format_prompt(
            "text_analysis",
            drawing_text=truncated_text,
            other_drawings_info='\n'.join(other_drawings_info)
        )

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
            response = self.llm.invoke(prompt).content
            
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
            response = self.llm.invoke(prompt).content
            
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
