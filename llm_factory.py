"""
llm_factory.py - Multi-provider factory and automatic rate-limit fallback wrapper.

Supports:
1. Groq (ChatGroq)
2. Google Gemini (ChatGoogleGenerativeAI)
3. OpenRouter (ChatOpenAI with openrouter base_url)
"""

import os
import re
from typing import List, Dict, Any, Optional, Callable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage


PROVIDER_MODELS = {
    "Groq": [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3.8-27b",
    ],
    "Google Gemini": [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-1.5-flash",
    ],
    "OpenRouter": [
        "qwen/qwen3-coder:free",
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "meta-llama/llama-3.3-70b:free",
    ],
}

PROVIDER_DEFAULT_MODELS = {
    "Groq": "openai/gpt-oss-120b",
    "Google Gemini": "gemini-3-flash-preview",
    "OpenRouter": "qwen/qwen3-coder:free",
}

# 100% verified rock-solid fallback default models to recover gracefully if a model fails or is not found
VERIFIED_FALLBACK_MODELS = {
    "Groq": "llama-3.3-70b-versatile",
    "Google Gemini": "gemini-1.5-flash",
    "OpenRouter": "meta-llama/llama-3.3-70b:free",
}

PROVIDER_ENV_KEYS = {
    "Groq": ["GROQ_API_KEY"],
    "Google Gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "OpenRouter": ["OPENROUTER_API_KEY"],
}


def extract_text_content(response: Any) -> str:
    """
    Extracts plain text content cleanly from LLM responses:
    - Handles standard string responses and LangChain AIMessage objects.
    - Handles structured content lists like [{'type': 'text', 'text': '...'}, ...] commonly returned by ChatGoogleGenerativeAI or message chunks.
    - Strips internal tool call metadata, dictionary wrappers, and extra JSON/literal formatting.
    """
    if response is None:
        return ""

    # If object has .content attribute (e.g., AIMessage / BaseMessage)
    content = getattr(response, "content", response)

    if isinstance(content, str):
        raw_str = content.strip()
        # Check if raw_str is a stringified list/dict (e.g. "[{'type': 'text', 'text': '...'}]")
        if (raw_str.startswith("[") and raw_str.endswith("]")) or (raw_str.startswith("{") and raw_str.endswith("}")):
            try:
                import ast
                parsed = ast.literal_eval(raw_str)
                return extract_text_content(parsed)
            except Exception:
                try:
                    import json
                    parsed = json.loads(raw_str)
                    return extract_text_content(parsed)
                except Exception:
                    pass
        return raw_str

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item.strip())
            elif isinstance(item, dict):
                # Check for standard text keys in dictionary blocks
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(str(item["text"]).strip())
                elif "text" in item and isinstance(item["text"], str):
                    text_parts.append(item["text"].strip())
                elif "content" in item and isinstance(item["content"], str):
                    text_parts.append(item["content"].strip())
                elif "value" in item and isinstance(item["value"], str):
                    text_parts.append(item["value"].strip())
            elif hasattr(item, "text"):
                text_parts.append(str(item.text).strip())
            elif hasattr(item, "content"):
                text_parts.append(extract_text_content(getattr(item, "content")))
        return "\n\n".join(p for p in text_parts if p).strip()

    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"]).strip()
        if "text" in content:
            return str(content["text"]).strip()
        if "content" in content:
            return extract_text_content(content["content"])
        if "parts" in content:
            return extract_text_content(content["parts"])

    return str(content).strip()


def sanitize_model_name(model_name: Optional[str], provider: str) -> str:
    """
    Sanitizes user-typed custom or standard model identifiers:
    - Strips leading and trailing whitespace and surrounding single/double quotes.
    - Passes the identifier directly to the model constructor without strict validation blocking.
    """
    if not model_name or not str(model_name).strip():
        return PROVIDER_DEFAULT_MODELS.get(provider, "")

    return str(model_name).strip().strip("'\"")


def get_default_key_for_provider(provider: str) -> str:
    """Finds API key from environment for given provider."""
    env_keys = PROVIDER_ENV_KEYS.get(provider, [])
    for key_name in env_keys:
        val = os.getenv(key_name, "")
        if val:
            return val.strip().strip("'\"")
    return ""


def create_llm(
    provider: str,
    api_key: str,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> BaseChatModel:
    """
    Factory function to instantiate a Chat model for the specified provider.
    Supports standard and custom model identifiers dynamically with robust sanitization.
    """
    if not api_key:
        raise ValueError(f"API key is required for provider '{provider}'.")

    # Sanitize dynamic API key: strip leading/trailing whitespace and quotes
    clean_key = str(api_key).strip().strip("'\"")
    if not clean_key:
        raise ValueError(f"A non-empty API key is required for provider '{provider}'.")

    # Sanitize user-typed model name (handles spaces, quotes, prefixes)
    model = sanitize_model_name(model_name, provider)

    if provider == "Groq":
        from langchain_groq import ChatGroq
        os.environ["GROQ_API_KEY"] = clean_key
        return ChatGroq(
            groq_api_key=clean_key,
            model_name=model,
            temperature=temperature,
        )

    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Set both environment variables to ensure underlying SDK and vertex/genai clients prioritize the dynamic key
        os.environ["GOOGLE_API_KEY"] = clean_key
        os.environ["GEMINI_API_KEY"] = clean_key
        return ChatGoogleGenerativeAI(
            google_api_key=clean_key,
            model=model,
            temperature=temperature,
        )

    elif provider == "OpenRouter":
        from langchain_openai import ChatOpenAI
        os.environ["OPENROUTER_API_KEY"] = clean_key
        os.environ["OPENAI_API_KEY"] = clean_key
        return ChatOpenAI(
            api_key=clean_key,
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

    def _is_model_error(self, err: Exception) -> bool:
        err_str = str(err).lower()
        return (
            "model_not_found" in err_str
            or "not_found" in err_str
            or "not found" in err_str
            or "404" in err_str
            or "does not exist" in err_str
            or "unknown model" in err_str
            or "decommissioned" in err_str
            or "unsupported model" in err_str
            or "not supported" in err_str
            or "resourcenotfound" in err_str
            or "notfounderror" in err_str
            or ("invalid_argument" in err_str and "model" in err_str)
        )

    def invoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Invokes the LLM with automatic fallback upon 429 / rate limits or model errors.
        """
        if not self.providers_configs:
            raise ValueError("No providers with valid API keys configured.")

        attempts = 0
        total_providers = len(self.providers_configs)

        while attempts < total_providers + 2:
            current_cfg = self.providers_configs[self.active_index]
            current_provider = current_cfg["provider"]

            try:
                llm = self._get_or_create_llm(self.active_index)
                result = llm.invoke(messages, **kwargs)

                if self.on_call_completed:
                    self.on_call_completed()

                return result

            except Exception as e:
                # 1. Rate Limit fallback: rotate to next available provider
                if self._is_rate_limit_error(e) and total_providers > 1 and attempts < total_providers - 1:
                    prev_provider = current_provider
                    self.active_index = (self.active_index + 1) % total_providers
                    next_cfg = self.providers_configs[self.active_index]
                    next_provider = next_cfg["provider"]

                    if self.on_fallback:
                        self.on_fallback(prev_provider, next_provider, str(e))

                    attempts += 1
                    continue

                # 2. Model Failure fallback: if custom/failed model, retry with verified default model first
                elif self._is_model_error(e):
                    verified_model = VERIFIED_FALLBACK_MODELS.get(current_provider) or PROVIDER_DEFAULT_MODELS.get(current_provider)
                    current_model = current_cfg.get("model_name")
                    if verified_model and current_model != verified_model:
                        # Update config to verified default model and clear cached instance to recreate
                        current_cfg["model_name"] = verified_model
                        if self.active_index in self._llm_instances:
                            del self._llm_instances[self.active_index]
                        if self.on_fallback:
                            self.on_fallback(
                                current_provider,
                                current_provider,
                                f"Model '{current_model}' is unavailable on {current_provider}. Retrying with verified working default '{verified_model}'."
                            )
                        attempts += 1
                        continue
                    elif total_providers > 1 and attempts < total_providers - 1:
                        prev_provider = current_provider
                        self.active_index = (self.active_index + 1) % total_providers
                        next_cfg = self.providers_configs[self.active_index]
                        next_provider = next_cfg["provider"]
                        if self.on_fallback:
                            self.on_fallback(
                                prev_provider,
                                next_provider,
                                f"Model '{current_model}' failed on {prev_provider}. Switched to {next_provider}."
                            )
                        attempts += 1
                        continue
                    else:
                        raise e
                else:
                    # Non-recoverable error or out of fallbacks
                    raise e
