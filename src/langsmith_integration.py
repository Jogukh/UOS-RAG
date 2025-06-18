#!/usr/bin/env python3
"""
LangSmith 추적 및 관찰성 통합 모듈
VLM 프로젝트의 모든 워크플로우와 LLM 호출을 LangSmith에서 추적할 수 있도록 지원
"""

import os
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime

try:
    from langsmith import traceable, Client
    from langsmith.wrappers import wrap_openai
    HAS_LANGSMITH = True
except ImportError:
    print("⚠️  LangSmith가 설치되지 않았습니다. pip install langsmith로 설치해주세요.")
    HAS_LANGSMITH = False
    
    # Mock decorators for when LangSmith is not available
    def traceable(func=None, *, run_type: str = "chain", name: str = None):
        def decorator(f):
            return f
        return decorator(func) if func else decorator

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

from src.env_config import EnvironmentConfig

logger = logging.getLogger(__name__)


class LangSmithTracker:
    """LangSmith 추적 관리 클래스"""
    
    def __init__(self):
        self.env_config = EnvironmentConfig()
        self.langsmith_config = self.env_config.langsmith_config
        self.client = None
        self.enabled = False
        
        if HAS_LANGSMITH and self.langsmith_config['tracing']:
            self._setup_langsmith()
    
    def _setup_langsmith(self):
        """LangSmith 환경 설정"""
        try:
            # 환경변수 설정
            self.env_config.setup_langsmith()
            
            # 클라이언트 초기화
            if self.langsmith_config['api_key']:
                self.client = Client(
                    api_key=self.langsmith_config['api_key'],
                    api_url=self.langsmith_config['endpoint']
                )
                self.enabled = True
                logger.info("✅ LangSmith 클라이언트 초기화 성공")
            else:
                logger.warning("⚠️  LANGSMITH_API_KEY가 설정되지 않았습니다.")
                
        except Exception as e:
            logger.error(f"❌ LangSmith 설정 실패: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """LangSmith 추적이 활성화되어 있는지 확인"""
        return self.enabled and HAS_LANGSMITH
    
    def create_session(self, session_name: str, metadata: Dict[str, Any] = None) -> str:
        """새로운 추적 세션 생성"""
        if not self.is_enabled():
            return "disabled"
            
        try:
            session_id = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # LangSmith에서는 세션을 프로젝트 레벨에서 관리
            return session_id
        except Exception as e:
            logger.error(f"❌ LangSmith 세션 생성 실패: {e}")
            return "error"
    
    def start_session(self, session_name: str, metadata: Dict[str, Any] = None) -> str:
        """새로운 추적 세션 시작 (호환성을 위한 별칭)"""
        return self.create_session(session_name, metadata)
    
    def end_session(self, session_id: str) -> bool:
        """추적 세션 종료"""
        if not self.is_enabled():
            return False
            
        try:
            logger.info(f"🔚 LangSmith 세션 종료: {session_id}")
            return True
        except Exception as e:
            logger.error(f"❌ LangSmith 세션 종료 실패: {e}")
            return False


# 전역 추적기 인스턴스
langsmith_tracker = LangSmithTracker()


def trace_llm_call(name: str = None, run_type: str = "llm"):
    """LLM 호출을 추적하는 데코레이터"""
    def decorator(func: Callable):
        if not langsmith_tracker.is_enabled():
            return func
            
        @traceable(run_type=run_type, name=name or func.__name__)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def trace_workflow_step(name: str = None, run_type: str = "chain"):
    """워크플로우 단계를 추적하는 데코레이터"""
    def decorator(func: Callable):
        if not langsmith_tracker.is_enabled():
            return func
            
        @traceable(run_type=run_type, name=name or func.__name__)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def trace_tool_call(name: str = None):
    """도구 호출을 추적하는 데코레이터"""
    def decorator(func: Callable):
        if not langsmith_tracker.is_enabled():
            return func
            
        @traceable(run_type="tool", name=name or func.__name__)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def trace_retrieval(name: str = None):
    """검색/RAG 호출을 추적하는 데코레이터"""
    def decorator(func: Callable):
        if not langsmith_tracker.is_enabled():
            return func
            
        @traceable(run_type="retriever", name=name or func.__name__)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class TrackedVLLM:
    """LangSmith 추적이 통합된 vLLM 래퍼"""
    
    def __init__(self, model_name: str = None, **vllm_kwargs):
        self.env_config = EnvironmentConfig()
        self.model_name = model_name or self.env_config.model_config.model_name
        
        if HAS_VLLM:
            # vLLM 설정 가져오기
            vllm_config = self.env_config.get_vllm_config()
            vllm_config.update(vllm_kwargs)
            
            self.llm = LLM(**vllm_config)
            self.sampling_params = SamplingParams(**self.env_config.get_sampling_params())
        else:
            self.llm = None
            self.sampling_params = None
            logger.warning("⚠️  vLLM이 설치되지 않았습니다.")
    
    @trace_llm_call(name="vLLM Generate")
    def generate(self, prompts, sampling_params=None, **kwargs):
        """추적이 포함된 텍스트 생성"""
        if not self.llm:
            raise RuntimeError("vLLM이 초기화되지 않았습니다.")
            
        params = sampling_params or self.sampling_params
        return self.llm.generate(prompts, params, **kwargs)
    
    @trace_llm_call(name="vLLM Chat")
    def chat(self, messages, sampling_params=None, **kwargs):
        """추적이 포함된 채팅 완성"""
        if not self.llm:
            raise RuntimeError("vLLM이 초기화되지 않았습니다.")
            
        params = sampling_params or self.sampling_params
        # vLLM의 chat 인터페이스 사용 (모델이 지원하는 경우)
        return self.llm.chat(messages, params, **kwargs)


def setup_langsmith_for_project():
    """프로젝트 전체에 LangSmith 설정 적용"""
    if langsmith_tracker.is_enabled():
        logger.info("🚀 LangSmith 추적 활성화됨")
        return True
    else:
        logger.info("ℹ️  LangSmith 추적이 비활성화되어 있습니다.")
        return False


def get_langsmith_project_url() -> Optional[str]:
    """현재 프로젝트의 LangSmith URL 반환"""
    if not langsmith_tracker.is_enabled():
        return None
        
    project_name = langsmith_tracker.langsmith_config['project']
    return f"https://smith.langchain.com/projects/p/{project_name}"


if __name__ == "__main__":
    # 테스트 실행
    print("🧪 LangSmith 통합 테스트")
    print("=" * 50)
    
    tracker = LangSmithTracker()
    print(f"LangSmith 활성화: {tracker.is_enabled()}")
    
    if tracker.is_enabled():
        print(f"프로젝트: {tracker.langsmith_config['project']}")
        print(f"엔드포인트: {tracker.langsmith_config['endpoint']}")
        print(f"프로젝트 URL: {get_langsmith_project_url()}")
        
        # 테스트 세션 생성
        session_id = tracker.create_session("test_session", {"test": True})
        print(f"테스트 세션 ID: {session_id}")
    
    print("✅ LangSmith 통합 모듈 준비 완료")
