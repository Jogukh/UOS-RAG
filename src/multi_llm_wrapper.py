#!/usr/bin/env python3
"""
Multi-LLM Wrapper: Ollama와 OpenAI API를 모두 지원하는 통합 LLM 래퍼
환경 설정에 따라 적절한 LLM 제공자를 선택하여 사용
"""

import os
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    ChatOpenAI = None

try:
    from .env_config import get_env_config
except ImportError:
    try:
        from env_config import get_env_config
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from env_config import get_env_config
from langsmith_integration import trace_llm_call

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """기본 LLM 인터페이스"""
    
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """프롬프트를 받아 응답을 생성"""
        pass


class OllamaLLM(BaseLLM):
    """Ollama LLM 래퍼"""
    
    def __init__(self, model_name: str = None, base_url: str = None, **kwargs):
        if not HAS_OLLAMA:
            raise ImportError("langchain-ollama가 설치되지 않았습니다. pip install langchain-ollama로 설치해주세요.")
        
        config = get_env_config().llm_provider_config
        self.model_name = model_name or config.ollama_model
        self.base_url = base_url or config.ollama_base_url
        
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=kwargs.get('temperature', 0.1),
            num_predict=kwargs.get('num_predict', 2048),
            timeout=kwargs.get('timeout', 60),
        )
        logger.info(f"Ollama LLM 초기화 완료: {self.model_name}")
    
    @trace_llm_call(name="Ollama Generate")
    def invoke(self, prompt: str, **kwargs) -> str:
        """Ollama API로 텍스트 생성"""
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Ollama API 호출 실패: {e}")
            raise


class OpenAILLM(BaseLLM):
    """OpenAI LLM 래퍼"""
    
    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None, **kwargs):
        if not HAS_OPENAI:
            raise ImportError("langchain-openai가 설치되지 않았습니다. pip install langchain-openai로 설치해주세요.")
        
        config = get_env_config().llm_provider_config
        self.model_name = model_name or config.gpt4_nano_model
        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url
        
        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일의 OPENAI_API_KEY를 확인해주세요.")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=kwargs.get('temperature', config.gpt4_nano_temperature),
            max_tokens=kwargs.get('max_tokens', config.gpt4_nano_max_tokens),
            timeout=kwargs.get('timeout', 60),
        )
        logger.info(f"OpenAI LLM 초기화 완료: {self.model_name}")
    
    @trace_llm_call(name="OpenAI Generate")
    def invoke(self, prompt: str, **kwargs) -> str:
        """OpenAI API로 텍스트 생성"""
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            raise


class MultiLLMWrapper:
    """환경 설정에 따라 적절한 LLM을 선택하는 통합 래퍼"""
    
    def __init__(self, provider: str = None, **kwargs):
        self.config = get_env_config()
        self.provider = provider or self.config.llm_provider_config.provider
        self.llm = None
        
        # provider는 kwargs에서 제거하여 LLM 클래스에 전달되지 않도록 함
        kwargs.pop('provider', None)
        self._initialize_llm(**kwargs)
    
    def _initialize_llm(self, **kwargs):
        """환경 설정에 따라 LLM 초기화"""
        try:
            if self.provider == "ollama":
                if not HAS_OLLAMA:
                    logger.warning("Ollama가 설치되지 않아 OpenAI로 대체합니다.")
                    self.provider = "openai"
                else:
                    self.llm = OllamaLLM(**kwargs)
                    return
            
            if self.provider == "openai":
                if not HAS_OPENAI:
                    logger.warning("OpenAI가 설치되지 않아 Ollama로 대체합니다.")
                    self.provider = "ollama"
                    if HAS_OLLAMA:
                        self.llm = OllamaLLM(**kwargs)
                    else:
                        raise ImportError("Ollama와 OpenAI 모두 사용할 수 없습니다.")
                else:
                    self.llm = OpenAILLM(**kwargs)
                    return
            
            if not self.llm:
                raise ValueError(f"지원하지 않는 LLM 제공자: {self.provider}")
                
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            raise
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """통합 LLM 호출 인터페이스"""
        if not self.llm:
            raise RuntimeError("LLM이 초기화되지 않았습니다.")
        
        return self.llm.invoke(prompt, **kwargs)
    
    def get_provider(self) -> str:
        """현재 사용 중인 LLM 제공자 반환"""
        return self.provider
    
    def get_model_name(self) -> str:
        """현재 사용 중인 모델명 반환"""
        if isinstance(self.llm, OllamaLLM):
            return self.llm.model_name
        elif isinstance(self.llm, OpenAILLM):
            return self.llm.model_name
        else:
            return "unknown"


# 전역 LLM 인스턴스
_llm_instance = None


def get_llm(**kwargs) -> MultiLLMWrapper:
    """전역 LLM 인스턴스 반환 (싱글톤 패턴)"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = MultiLLMWrapper(**kwargs)
    return _llm_instance


def reset_llm():
    """전역 LLM 인스턴스 리셋"""
    global _llm_instance
    _llm_instance = None


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 Multi-LLM Wrapper 테스트")
    print("=" * 50)
    
    try:
        llm = get_llm()
        print(f"✅ LLM 초기화 성공")
        print(f"📋 제공자: {llm.get_provider()}")
        print(f"🤖 모델: {llm.get_model_name()}")
        
        # 간단한 테스트 프롬프트
        test_prompt = "안녕하세요! 간단한 인사말로 응답해주세요."
        print(f"\n📝 테스트 프롬프트: {test_prompt}")
        
        response = llm.invoke(test_prompt)
        print(f"🤖 응답: {response}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
