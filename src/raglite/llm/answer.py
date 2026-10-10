from typing import Any, Dict, Generator, List, Optional

from ..errors import LLMError
from ..types import AnswerResult, LLMProviderConfig, SearchResult
from .factory import ResolvedLLM, create_llm
from .prompt import build_system_prompt, build_user_prompt, extract_citations


def generate_answer(
    options: Dict[str, Any],
    *,
    llm_config: Optional[LLMProviderConfig] = None,
    question: Optional[str] = None,
    context: Optional[List[SearchResult]] = None,
) -> AnswerResult:
    """Generate answer from LLM with exact option structure or keyword args."""
    # Handle dict options or keyword args
    opts = options or {}
    llm_conf = llm_config or opts.get("llm")
    q = question or opts.get("question")
    ctx = context or opts.get("context")

    if not llm_conf:
        raise LLMError("No LLM provider configuration specified")
    if q is None:
        raise LLMError("No question specified")
    if ctx is None:
        raise LLMError("No context specified")

    llm = create_llm(llm_conf)
    system_prompt = build_system_prompt(opts)
    user_prompt = build_user_prompt(q, ctx, opts)

    try:
        res_dict = _generate(llm, system_prompt, user_prompt)
        result = AnswerResult.model_validate(res_dict)
        result.citations = extract_citations(result.text or "", ctx)
        return result
    except Exception as cause:
        raise LLMError(
            f"Failed to generate answer via {llm.provider} ({llm.model})",
            cause=cause,
        )


def stream_answer(
    options: Dict[str, Any],
    *,
    llm_config: Optional[LLMProviderConfig] = None,
    question: Optional[str] = None,
    context: Optional[List[SearchResult]] = None,
) -> Generator[str, None, None]:
    """Stream answer from LLM as an iterator of text deltas."""
    opts = options or {}
    llm_conf = llm_config or opts.get("llm")
    q = question or opts.get("question")
    ctx = context or opts.get("context")

    if not llm_conf:
        raise LLMError("No LLM provider configuration specified")
    if q is None:
        raise LLMError("No question specified")
    if ctx is None:
        raise LLMError("No context specified")

    llm = create_llm(llm_conf)
    system_prompt = build_system_prompt(opts)
    user_prompt = build_user_prompt(q, ctx, opts)

    try:
        yield from _stream(llm, system_prompt, user_prompt)
    except Exception as cause:
        raise LLMError(
            f"Failed to stream answer via {llm.provider} ({llm.model})",
            cause=cause,
        )


def _generate(llm: ResolvedLLM, system_prompt: str, user_prompt: str) -> dict:
    p = llm.provider

    if p in ("openai", "groq", "xai", "ollama"):
        params = {
            "model": llm.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": llm.temperature,
        }
        if llm.max_tokens is not None:
            params["max_tokens"] = llm.max_tokens
        resp = llm.client.chat.completions.create(**params)
        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "promptTokens": resp.usage.prompt_tokens,
                "completionTokens": resp.usage.completion_tokens,
                "totalTokens": resp.usage.total_tokens,
            }
        return {
            "text": choice.message.content,
            "provider": llm.provider,
            "model": llm.model,
            "usage": usage,
            "finishReason": choice.finish_reason,
        }

    elif p == "anthropic":
        params = {
            "model": llm.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        _anthropic_options(llm, params)
        resp = llm.client.messages.create(**params)
        usage = {}
        if resp.usage:
            usage = {
                "promptTokens": resp.usage.input_tokens,
                "completionTokens": resp.usage.output_tokens,
                "totalTokens": resp.usage.input_tokens + resp.usage.output_tokens,
            }
        return {
            # Thinking blocks may come before the text.
            "text": "".join(b.text for b in resp.content if b.type == "text"),
            "provider": llm.provider,
            "model": llm.model,
            "usage": usage,
            "finishReason": resp.stop_reason,
        }

    elif p == "google":
        resp = llm.client.models.generate_content(
            model=llm.model,
            contents=user_prompt,
            config=_google_config(llm, system_prompt),
        )
        usage = {}
        if resp.usage_metadata:
            usage = {
                "promptTokens": resp.usage_metadata.prompt_token_count,
                "completionTokens": resp.usage_metadata.candidates_token_count,
                "totalTokens": resp.usage_metadata.total_token_count,
            }
        finish_reason = None
        if resp.candidates:
            finish_reason = str(resp.candidates[0].finish_reason)
        return {
            "text": resp.text,
            "provider": llm.provider,
            "model": llm.model,
            "usage": usage,
            "finishReason": finish_reason,
        }

    elif p == "cohere":
        resp = llm.client.chat(**_cohere_params(llm, system_prompt, user_prompt))
        usage = {}
        tokens = resp.usage.tokens if resp.usage else None
        if tokens and tokens.input_tokens is not None and tokens.output_tokens is not None:
            usage = {
                "promptTokens": int(tokens.input_tokens),
                "completionTokens": int(tokens.output_tokens),
                "totalTokens": int(tokens.input_tokens + tokens.output_tokens),
            }
        content = resp.message.content if resp.message else None
        return {
            "text": "".join(c.text for c in (content or []) if c.type == "text"),
            "provider": llm.provider,
            "model": llm.model,
            "usage": usage,
            "finishReason": resp.finish_reason,
        }

    elif p == "mistral":
        params = {
            "model": llm.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": llm.temperature,
        }
        if llm.max_tokens is not None:
            params["max_tokens"] = llm.max_tokens
        resp = llm.client.chat.complete(**params)
        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "promptTokens": resp.usage.prompt_tokens,
                "completionTokens": resp.usage.completion_tokens,
                "totalTokens": resp.usage.total_tokens,
            }
        return {
            "text": choice.message.content,
            "provider": llm.provider,
            "model": llm.model,
            "usage": usage,
            "finishReason": choice.finish_reason,
        }

    else:
        raise LLMError(f"Unsupported LLM provider: {llm.provider}")


def _stream(llm: ResolvedLLM, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
    p = llm.provider

    if p in ("openai", "groq", "xai", "ollama"):
        params = {
            "model": llm.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": llm.temperature,
            "stream": True,
        }
        if llm.max_tokens is not None:
            params["max_tokens"] = llm.max_tokens
        stream = llm.client.chat.completions.create(**params)
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    elif p == "anthropic":
        params = {
            "model": llm.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        _anthropic_options(llm, params)
        with llm.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text

    elif p == "google":
        resp = llm.client.models.generate_content_stream(
            model=llm.model,
            contents=user_prompt,
            config=_google_config(llm, system_prompt),
        )
        for chunk in resp:
            if chunk.text:
                yield chunk.text

    elif p == "cohere":
        for event in llm.client.chat_stream(**_cohere_params(llm, system_prompt, user_prompt)):
            if event.type != "content-delta":
                continue
            message = event.delta.message if event.delta else None
            text = message.content.text if message and message.content else None
            if text:
                yield text

    elif p == "mistral":
        params = {
            "model": llm.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": llm.temperature,
        }
        if llm.max_tokens is not None:
            params["max_tokens"] = llm.max_tokens
        resp = llm.client.chat.stream(**params)
        for chunk in resp:
            delta = chunk.data.choices[0].delta.content if chunk.data.choices else None
            if delta:
                yield delta

    else:
        raise LLMError(f"Unsupported LLM provider: {llm.provider}")


# Aliases for TS parity
generateAnswer = generate_answer
streamAnswer = stream_answer


def _google_config(llm: ResolvedLLM, system_prompt: str) -> dict:
    """GenerateContentConfig fields for the google-genai client."""
    config = {"system_instruction": system_prompt, "temperature": llm.temperature}
    if llm.max_tokens is not None:
        config["max_output_tokens"] = llm.max_tokens
    return config


def _anthropic_options(llm: ResolvedLLM, params: dict) -> None:
    """max_tokens (thinking counts against it) and an explicitly configured temperature.

    anthropic>=1.0 dropped the `temperature` argument, so it goes in extra_body.
    """
    params["max_tokens"] = llm.max_tokens if llm.max_tokens is not None else 16000
    if llm.temperature is not None:
        params["extra_body"] = {"temperature": llm.temperature}


def _cohere_params(llm: ResolvedLLM, system_prompt: str, user_prompt: str) -> dict:
    """Chat arguments for the Cohere v2 client."""
    params = {
        "model": llm.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": llm.temperature,
    }
    if llm.max_tokens is not None:
        params["max_tokens"] = llm.max_tokens
    return params
