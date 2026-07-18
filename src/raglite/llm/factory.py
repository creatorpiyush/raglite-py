import os
from typing import Optional, Any, Union

from ..types import LLMProviderConfig, LLMProviderName
from .models import DEFAULT_LLM_MODELS
from ..errors import LLMError


class ResolvedLLM:
    def __init__(
        self,
        provider: LLMProviderName,
        model: str,
        client: Any,
        temperature: float,
        max_tokens: Optional[int] = None,
    ):
        self.provider = provider
        self.model = model
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens


def create_llm(config: Union[LLMProviderConfig, dict]) -> ResolvedLLM:
    """Create a resolved LLM client instance from configuration."""
    if isinstance(config, dict):
        config = LLMProviderConfig.model_validate(config)
    model = config.model or DEFAULT_LLM_MODELS[config.provider]
    temperature = config.temperature if config.temperature is not None else 0.0
    max_tokens = config.maxTokens

    client = build_language_client(config.provider, config)

    return ResolvedLLM(
        provider=config.provider,
        model=model,
        client=client,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_language_client(provider: LLMProviderName, config: LLMProviderConfig) -> Any:
    apiKey = config.apiKey
    baseURL = config.baseURL

    if provider == "openai":
        from openai import OpenAI

        return OpenAI(
            api_key=apiKey or os.environ.get("OPENAI_API_KEY"), base_url=baseURL
        )

    elif provider == "anthropic":
        from anthropic import Anthropic

        return Anthropic(
            api_key=apiKey or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=baseURL,
        )

    elif provider == "google":
        import google.generativeai as genai

        key = (
            apiKey
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        genai.configure(api_key=key)
        return genai

    elif provider == "mistral":
        from mistralai import Mistral

        key = apiKey or os.environ.get("MISTRAL_API_KEY")
        return Mistral(api_key=key, server_url=baseURL)

    elif provider == "cohere":
        import cohere

        key = apiKey or os.environ.get("COHERE_API_KEY")
        return cohere.Client(api_key=key, base_url=baseURL)

    elif provider == "groq":
        from openai import OpenAI

        base = (
            baseURL
            or os.environ.get("GROQ_BASE_URL")
            or "https://api.groq.com/openai/v1"
        )
        key = apiKey or os.environ.get("GROQ_API_KEY")
        return OpenAI(api_key=key, base_url=base)

    elif provider == "xai":
        from openai import OpenAI

        base = (
            baseURL or os.environ.get("XAI_BASE_URL") or "https://api.x.ai/v1"
        )
        key = apiKey or os.environ.get("XAI_API_KEY")
        return OpenAI(api_key=key, base_url=base)

    elif provider == "ollama":
        from openai import OpenAI

        raw = (baseURL or "http://localhost:11434").rstrip("/")
        base = raw if raw.endswith("/v1") else f"{raw}/v1"
        key = apiKey or "ollama"
        return OpenAI(api_key=key, base_url=base)

    else:
        raise LLMError(f"Unsupported LLM provider: {provider}")


# Alias for TS parity
createLLM = create_llm
