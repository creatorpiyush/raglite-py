import os
from typing import Any, Optional, Union

from ..errors import LLMError, import_optional
from ..types import LLMProviderConfig, LLMProviderName
from .models import DEFAULT_LLM_MODELS


class ResolvedLLM:
    def __init__(
        self,
        provider: LLMProviderName,
        model: str,
        client: Any,
        temperature: Optional[float],
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
    # Claude 5 models reject non-default sampling values, so Anthropic gets no
    # temperature unless one is configured.
    temperature = config.temperature
    if temperature is None and config.provider != "anthropic":
        temperature = 0.0
    max_tokens = config.maxTokens

    client = build_language_client(config.provider, config)

    return ResolvedLLM(
        provider=config.provider,
        model=model,
        client=client,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def mistral_client_class(purpose: str) -> Any:
    """`Mistral` lives at the top level in mistralai 1.x and in `mistralai.client` in 2.x and later."""
    module = import_optional("mistralai", "mistral", purpose)
    if hasattr(module, "Mistral"):
        return module.Mistral
    return import_optional("mistralai.client", "mistral", purpose).Mistral


def build_language_client(provider: LLMProviderName, config: LLMProviderConfig) -> Any:
    apiKey = config.apiKey
    baseURL = config.baseURL
    purpose = f'The "{provider}" LLM provider'

    if provider == "openai":
        OpenAI = import_optional("openai", "openai", purpose).OpenAI

        return OpenAI(
            api_key=apiKey or os.environ.get("OPENAI_API_KEY"), base_url=baseURL
        )

    elif provider == "anthropic":
        Anthropic = import_optional("anthropic", "anthropic", purpose).Anthropic

        return Anthropic(
            api_key=apiKey or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=baseURL,
        )

    elif provider == "google":
        genai = import_optional("google.genai", "google", purpose)

        key = (
            apiKey
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        http_options = {"base_url": baseURL} if baseURL else None
        return genai.Client(api_key=key, http_options=http_options)

    elif provider == "mistral":
        Mistral = mistral_client_class(purpose)

        key = apiKey or os.environ.get("MISTRAL_API_KEY")
        return Mistral(api_key=key, server_url=baseURL)

    elif provider == "cohere":
        cohere = import_optional("cohere", "cohere", purpose)

        key = apiKey or os.environ.get("COHERE_API_KEY")
        return cohere.ClientV2(api_key=key, base_url=baseURL)

    elif provider == "groq":
        OpenAI = import_optional("openai", "groq", purpose).OpenAI

        base = (
            baseURL
            or os.environ.get("GROQ_BASE_URL")
            or "https://api.groq.com/openai/v1"
        )
        key = apiKey or os.environ.get("GROQ_API_KEY")
        return OpenAI(api_key=key, base_url=base)

    elif provider == "xai":
        OpenAI = import_optional("openai", "xai", purpose).OpenAI

        base = (
            baseURL or os.environ.get("XAI_BASE_URL") or "https://api.x.ai/v1"
        )
        key = apiKey or os.environ.get("XAI_API_KEY")
        return OpenAI(api_key=key, base_url=base)

    elif provider == "ollama":
        OpenAI = import_optional("openai", "ollama", purpose).OpenAI

        raw = (baseURL or "http://localhost:11434").rstrip("/")
        base = raw if raw.endswith("/v1") else f"{raw}/v1"
        key = apiKey or "ollama"
        return OpenAI(api_key=key, base_url=base)

    else:
        raise LLMError(f"Unsupported LLM provider: {provider}")


# Alias for TS parity
createLLM = create_llm
