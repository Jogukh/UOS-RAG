"""
공통 LLM 유틸리티: Ollama와 OpenAI(GPT-4.1-nano) 모두 지원
"""
import os
import requests
from typing import Optional, Dict, Any
from env_config import get_env_config

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def get_llm_provider():
    """환경설정 기반 LLM 제공자 반환 (ollama, openai)"""
    config = get_env_config()
    return config.llm_provider_config.provider


def call_ollama_llm(prompt: str, model: Optional[str] = None, base_url: Optional[str] = None, **kwargs) -> str:
    """Ollama API로 LLM 호출"""
    config = get_env_config().llm_provider_config
    model = model or config.ollama_model
    base_url = base_url or config.ollama_base_url
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": kwargs or {}
    }
    resp = requests.post(f"{base_url}/api/generate", json=data, timeout=60)
    resp.raise_for_status()
    return resp.json().get("response", "")


def call_openai_llm(prompt: str, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> str:
    """OpenAI API(GPT-4.1-nano 등)로 LLM 호출"""
    if not HAS_OPENAI:
        raise ImportError("openai 패키지가 필요합니다. pip install openai")
    config = get_env_config().llm_provider_config
    model = model or config.gpt4_nano_model
    api_key = api_key or config.openai_api_key
    base_url = base_url or config.openai_base_url
    max_tokens = max_tokens or config.gpt4_nano_max_tokens
    temperature = temperature if temperature is not None else config.gpt4_nano_temperature
    openai.api_key = api_key
    openai.base_url = base_url
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )
    return response.choices[0].message["content"]


def call_llm(prompt: str, **kwargs) -> str:
    """환경설정에 따라 Ollama 또는 OpenAI LLM 호출"""
    provider = get_llm_provider()
    if provider == "ollama":
        return call_ollama_llm(prompt, **kwargs)
    elif provider == "openai":
        return call_openai_llm(prompt, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 LLM 제공자: {provider}")
