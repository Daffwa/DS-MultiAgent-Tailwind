"""Universal LLM Factory for Hybrid Multi-Agent Orchestration.
Mendukung:
1. Google Gemini (Cloud API via ChatGoogleGenerativeAI)
2. Local LLMs (Ollama, FreeToken, LM Studio, vLLM via ChatOpenAI / OpenAI-compatible API)
"""

import os
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel


def get_llm(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_output_tokens: int = 3000
) -> BaseChatModel:
    """Factory terpadu untuk membuat instance LLM (Cloud Gemini atau Local LLM)."""
    provider_lower = (provider or "gemini").lower().strip()

    if provider_lower in ["local", "ollama", "freetoken", "lmstudio", "openai_compatible"]:
        from langchain_openai import ChatOpenAI

        target_base_url = (base_url or os.getenv("LOCAL_LLM_BASE_URL") or "http://localhost:11434/v1").strip()
        target_model = (model_name or os.getenv("LOCAL_LLM_MODEL") or "qwen2.5-coder:7b").strip()
        target_key = (api_key or os.getenv("LOCAL_LLM_API_KEY") or "ollama").strip()

        # Pastikan URL memiliki /v1 jika belum ada
        if not target_base_url.endswith("/v1") and not target_base_url.endswith("/v1/"):
            target_base_url = target_base_url.rstrip("/") + "/v1"

        return ChatOpenAI(
            model=target_model,
            api_key=target_key,
            base_url=target_base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
            timeout=120
        )
    else:
        # Default: Google Gemini Cloud
        from langchain_google_genai import ChatGoogleGenerativeAI

        target_key = (api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        target_model = (model_name or "gemma-4-31b-it").strip()

        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=target_key,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens
        )
