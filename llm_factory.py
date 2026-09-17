"""
llm_factory.py - Multi-provider factory and automatic rate-limit fallback wrapper.

Supports:
1. Groq (ChatGroq)
2. Google Gemini (ChatGoogleGenerativeAI)
3. OpenRouter (ChatOpenAI with openrouter base_url)
"""

import os
from typing import List, Dict, Any, Optional, Callable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage


PROVIDER_MODELS = {
    "Groq": [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "Google Gemini": [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "OpenRouter": [
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2.5-72b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
    ],
}

PROVIDER_DEFAULT_MODELS = {
    "Groq": "openai/gpt-oss-20b",
    "Google Gemini": "gemini-2.0-flash",
    "OpenRouter": "meta-llama/llama-3.1-8b-instruct:free",
}

PROVIDER_ENV_KEYS = {
    "Groq": ["GROQ_API_KEY"],
    "Google Gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "OpenRouter": ["OPENROUTER_API_KEY"],
}


def get_default_key_for_provider(provider: str) -> str:
    """Finds API key from environment for given provider."""
    env_keys = PROVIDER_ENV_KEYS.get(provider, [])
    for key_name in env_keys:
        val = os.getenv(key_name, "")
        if val:
            return val
    return ""


def create_llm(
    provider: str,
    api_key: str,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> BaseChatModel:
    """
    Factory function to instantiate a Chat model for the specified provider.
    """
    if not api_key:
        raise ValueError(f"API key is required for provider '{provider}'.")

    model = model_name or PROVIDER_DEFAULT_MODELS.get(provider)

    if provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=temperature,
        )

    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Set os env as fallback for google genai
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
        )

    elif provider == "OpenRouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=temperature,
            default_headers={
                "HTTP-Referer": "https://github.com/ai-devils-advocate",
                "X-Title": "AI Devil's Advocate",
            },
        )

    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Choose from {list(PROVIDER_MODELS.keys())}")


class FallbackLLMWrapper:
    """
    Wrapper around multiple LLM providers.
    If the primary provider fails due to a rate limit error (429 or quota exceeded),
    it automatically falls back to the next available provider in the list.
    Also tracks session call count and notifies listeners.
    """

    def __init__(
        self,
        providers_configs: List[Dict[str, Any]],
        on_fallback: Optional[Callable[[str, str, str], None]] = None,
        on_call_completed: Optional[Callable[[], None]] = None,
    ):
        """
        providers_configs: List of dicts, each with keys:
            - "provider": str ("Groq", "Google Gemini", "OpenRouter")
            - "api_key": str
            - "model_name": str
            - "temperature": float (optional)
        """
        self.providers_configs = [cfg for cfg in providers_configs if cfg.get("api_key")]
        self.on_fallback = on_fallback
        self.on_call_completed = on_call_completed
        self.active_index = 0
        self._llm_instances: Dict[int, BaseChatModel] = {}

    def _get_or_create_llm(self, index: int) -> BaseChatModel:
        if index not in self._llm_instances:
            cfg = self.providers_configs[index]
            self._llm_instances[index] = create_llm(
                provider=cfg["provider"],
                api_key=cfg["api_key"],
                model_name=cfg.get("model_name"),
                temperature=cfg.get("temperature", 0.7),
            )
        return self._llm_instances[index]

    def _is_rate_limit_error(self, err: Exception) -> bool:
        err_str = str(err).lower()
        return (
            "429" in err_str
            or "rate_limit" in err_str
            or "rate limit" in err_str
            or "quota" in err_str
            or "resource_exhausted" in err_str
            or "too many requests" in err_str
        )

    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Invokes the LLM with automatic fallback upon 429 / rate limits.
        """
        if not self.providers_configs:
            raise ValueError("No providers with valid API keys configured.")

        attempts = 0
        total_providers = len(self.providers_configs)

        while attempts < total_providers:
            current_cfg = self.providers_configs[self.active_index]
            current_provider = current_cfg["provider"]

            try:
                llm = self._get_or_create_llm(self.active_index)
                result = llm.invoke(messages, **kwargs)

                if self.on_call_completed:
                    self.on_call_completed()

                return result

            except Exception as e:
                if self._is_rate_limit_error(e) and total_providers > 1 and attempts < total_providers - 1:
                    prev_provider = current_provider
                    # Fallback to next provider
                    self.active_index = (self.active_index + 1) % total_providers
                    next_cfg = self.providers_configs[self.active_index]
                    next_provider = next_cfg["provider"]

                    if self.on_fallback:
                        self.on_fallback(prev_provider, next_provider, str(e))

                    attempts += 1
                    continue
                else:
                    # Non-recoverable error or out of fallbacks
                    raise e
