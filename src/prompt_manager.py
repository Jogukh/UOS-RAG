"""
VLM 건축 도면 분석 시스템 - 프롬프트 중앙 관리 모듈

이 모듈은 시스템 전체에서 사용되는 모든 LLM 프롬프트를 중앙에서 관리합니다.
프롬프트는 별도의 YAML 파일에서 로드되어 수정이 용이합니다.
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging


class PromptType(Enum):
    """프롬프트 유형 분류"""
    METADATA_EXTRACTION = "metadata_extraction"
    RELATIONSHIP_INFERENCE = "relationship_inference"
    TEXT_ANALYSIS = "text_analysis"
    RAG_QUERY = "rag_query"
    SYSTEM_PROMPT = "system_prompt"


@dataclass
class PromptTemplate:
    """프롬프트 템플릿 데이터 클래스"""
    name: str
    type: PromptType
    description: str
    used_by: List[str]  # 사용하는 모듈 목록
    template: str
    input_params: List[str]  # 필요한 입력 파라미터
    output_format: str  # 예상 출력 형식
    version: str = "1.0"


class PromptManager:
    """프롬프트 중앙 관리 클래스 - YAML 파일에서 프롬프트 로드"""
    
    def __init__(self, prompts_dir: str = None):
        self.logger = logging.getLogger(__name__)
        
        # 프롬프트 디렉토리 설정
        if prompts_dir is None:
            # 현재 파일 기준으로 ../prompts 디렉토리 사용
            current_file = Path(__file__)
            self.prompts_dir = current_file.parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        
        self.config_file = self.prompts_dir / "config.yaml"
        self.prompts = {}
        self.config = {}
        
        self._load_config()
        self._load_prompts()
    
    def _load_config(self):
        """프롬프트 설정 파일 로드"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"프롬프트 설정 로드됨: {self.config_file}")
            else:
                self.logger.warning(f"설정 파일을 찾을 수 없습니다: {self.config_file}")
                self._create_default_config()
        except Exception as e:
            self.logger.error(f"설정 파일 로드 실패: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """기본 설정 생성"""
        self.config = {
            "prompts_directory": "./prompts",
            "default_encoding": "utf-8",
            "prompt_files": {
                "metadata_extraction": "metadata_extraction.yaml",
                "relationship_inference": "relationship_inference.yaml", 
                "text_analysis": "text_analysis.yaml",
                "rag_query": "rag_query.yaml",
                "system_test": "system_test.yaml",
                "gemma_chat_wrapper": "gemma_chat_wrapper.yaml"
            },
            "cache_prompts": True,
            "validate_params": True
        }
    
    def _load_prompts(self):
        """모든 프롬프트 파일 로드"""
        if not self.prompts_dir.exists():
            self.logger.error(f"프롬프트 디렉토리를 찾을 수 없습니다: {self.prompts_dir}")
            return
        
        prompt_files = self.config.get("prompt_files", {})
        encoding = self.config.get("default_encoding", "utf-8")
        
        for prompt_id, filename in prompt_files.items():
            prompt_file = self.prompts_dir / filename
            
            try:
                if prompt_file.exists():
                    with open(prompt_file, 'r', encoding=encoding) as f:
                        prompt_data = yaml.safe_load(f)
                    
                    # PromptTemplate 객체 생성
                    prompt_template = PromptTemplate(
                        name=prompt_data.get("name", prompt_id),
                        type=PromptType(prompt_data.get("type", "system_prompt")),
                        description=prompt_data.get("description", ""),
                        used_by=prompt_data.get("used_by", []),
                        template=prompt_data.get("template", ""),
                        input_params=prompt_data.get("input_params", []),
                        output_format=prompt_data.get("output_format", ""),
                        version=prompt_data.get("version", "1.0")
                    )
                    
                    self.prompts[prompt_id] = prompt_template
                    self.logger.debug(f"프롬프트 로드됨: {prompt_id} from {filename}")
                else:
                    self.logger.warning(f"프롬프트 파일을 찾을 수 없습니다: {prompt_file}")
                    
            except Exception as e:
                self.logger.error(f"프롬프트 파일 로드 실패 {prompt_file}: {e}")
        
        self.logger.info(f"총 {len(self.prompts)}개의 프롬프트가 로드되었습니다.")
    
    def reload_prompts(self):
        """프롬프트 파일들을 다시 로드"""
        self.prompts.clear()
        self._load_config()
        self._load_prompts()
        self.logger.info("프롬프트가 다시 로드되었습니다.")
    
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """레거시 호환성을 위한 메서드 (더 이상 사용하지 않음)"""
        self.logger.warning("_initialize_prompts는 더 이상 사용되지 않습니다. YAML 파일에서 로드됩니다.")
        return self.prompts
    
    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        """프롬프트 ID로 프롬프트 템플릿 가져오기"""
        return self.prompts.get(prompt_id)
    
    def format_prompt(self, prompt_id: str, **kwargs) -> str:
        """프롬프트 ID와 파라미터로 포맷된 프롬프트 생성"""
        prompt_template = self.get_prompt(prompt_id)
        if not prompt_template:
            raise ValueError(f"프롬프트 ID '{prompt_id}'를 찾을 수 없습니다.")
        
        try:
            return prompt_template.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"프롬프트 '{prompt_id}'에 필요한 파라미터가 누락되었습니다: {e}")
    
    def list_prompts(self) -> List[Dict[str, str]]:
        """모든 프롬프트 목록 반환"""
        return [
            {
                "id": prompt_id,
                "name": template.name,
                "type": template.type.value,
                "description": template.description,
                "used_by": ", ".join(template.used_by),
                "version": template.version
            }
            for prompt_id, template in self.prompts.items()
        ]
    
    def get_prompts_by_type(self, prompt_type: PromptType) -> List[PromptTemplate]:
        """유형별 프롬프트 목록 반환"""
        return [
            template for template in self.prompts.values()
            if template.type == prompt_type
        ]
    
    def get_prompts_by_module(self, module_name: str) -> List[PromptTemplate]:
        """모듈별 프롬프트 목록 반환"""
        return [
            template for template in self.prompts.values()
            if module_name in template.used_by
        ]
    
    def validate_prompt_params(self, prompt_id: str, **kwargs) -> tuple[bool, List[str]]:
        """프롬프트 파라미터 유효성 검사"""
        prompt_template = self.get_prompt(prompt_id)
        if not prompt_template:
            return False, [f"프롬프트 ID '{prompt_id}'를 찾을 수 없습니다."]
        
        missing_params = []
        for param in prompt_template.input_params:
            if param not in kwargs:
                missing_params.append(param)
        
        return len(missing_params) == 0, missing_params
    
    def get_prompt_info(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """프롬프트 상세 정보 반환"""
        prompt_template = self.get_prompt(prompt_id)
        if not prompt_template:
            return None
        
        return {
            "id": prompt_id,
            "name": prompt_template.name,
            "type": prompt_template.type.value,
            "description": prompt_template.description,
            "used_by": prompt_template.used_by,
            "input_params": prompt_template.input_params,
            "output_format": prompt_template.output_format,
            "version": prompt_template.version,
            "template_preview": prompt_template.template[:200] + "..." if len(prompt_template.template) > 200 else prompt_template.template
        }


# 전역 프롬프트 매니저 인스턴스
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """전역 프롬프트 매니저 인스턴스 반환 (싱글톤 패턴)"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


# 편의 함수들
def get_prompt(prompt_id: str) -> Optional[PromptTemplate]:
    """프롬프트 가져오기 편의 함수"""
    return get_prompt_manager().get_prompt(prompt_id)

def format_prompt(prompt_id: str, **kwargs) -> str:
    """프롬프트 포맷팅 편의 함수"""
    return get_prompt_manager().format_prompt(prompt_id, **kwargs)

def list_prompts() -> List[Dict[str, str]]:
    """프롬프트 목록 편의 함수"""
    return get_prompt_manager().list_prompts()


if __name__ == "__main__":
    # 프롬프트 매니저 테스트
    pm = PromptManager()
    
    print("=== VLM 프롬프트 매니저 테스트 ===\n")
    
    # 1. 모든 프롬프트 목록
    print("📋 등록된 프롬프트 목록:")
    for prompt_info in pm.list_prompts():
        print(f"  - {prompt_info['id']}: {prompt_info['name']}")
        print(f"    타입: {prompt_info['type']}")
        print(f"    사용 모듈: {prompt_info['used_by']}")
        print(f"    설명: {prompt_info['description']}")
        print()
    
    # 2. 특정 프롬프트 테스트
    print("🧪 메타데이터 추출 프롬프트 테스트:")
    try:
        formatted = pm.format_prompt(
            "metadata_extraction",
            file_name="test.pdf",
            page_number=1,
            text_content="1층 평면도 A01-001 거실 주방 1/100"
        )
        print("✅ 프롬프트 포맷팅 성공")
        print(f"프롬프트 길이: {len(formatted)} 문자")
    except Exception as e:
        print(f"❌ 프롬프트 포맷팅 실패: {e}")
    
    # 3. 파라미터 유효성 검사
    print("\n🔍 파라미터 유효성 검사:")
    valid, missing = pm.validate_prompt_params("metadata_extraction", file_name="test.pdf")
    if valid:
        print("✅ 모든 필수 파라미터가 제공됨")
    else:
        print(f"❌ 누락된 파라미터: {missing}")
