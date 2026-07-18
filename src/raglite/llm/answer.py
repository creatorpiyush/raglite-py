from typing import List, AsyncGenerator, Generator, Optional, Any, Dict

from ..types import LLMProviderConfig, SearchResult, AnswerResult
from .factory import create_llm, ResolvedLLM
from .prompt import build_system_prompt, build_user_prompt
from ..errors import LLMError


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
        return AnswerResult.model_validate(res_dict)
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
            "temperature": llm.temperature,
        }
        if llm.max_tokens is not None:
            params["max_tokens"] = llm.max_tokens
        else:
            params["max_tokens"] = 4096
        resp = llm.client.messages.create(**params)
        usage = {}
        if resp.usage:
            usage = {
                "promptTokens": resp.usage.input_tokens,
                "completionTokens": resp.usage.output_tokens,
                "totalTokens": resp.usage.input_tokens + resp.usage.output_tokens,
            }
        return {
            "text": resp.content[0].text,
            "provider": llm.provider,
            "model": llm.model,
            "usage": usage,
            "finishReason": resp.stop_reason,
        }

    elif p == "google":
        model = llm.client.GenerativeModel(
            model_name=llm.model,
            system_instruction=system_prompt,
            generation_config={
                "temperature": llm.temperature,
                "max_output_tokens": llm.max_tokens,
            },
        )
        resp = model.generate_content(user_prompt)
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
        params = {
            "model": llm.model,
            "preamble": system_prompt,
            "message": user_prompt,
            "temperature": llm.temperature,
        }
        if llm.max_tokens is not None:
            params["max_tokens"] = llm.max_tokens
        resp = llm.client.chat(**params)
        usage = {}
        if resp.meta and resp.meta.tokens:
            usage = {
                "promptTokens": resp.meta.tokens.input_tokens,
                "completionTokens": resp.meta.tokens.output_tokens,
                "totalTokens": resp.meta.tokens.input_tokens
                + resp.meta.tokens.output_tokens,
            }
        finish_reason = None
        if hasattr(resp, "finish_reason"):
            finish_reason = resp.finish_reason
        return {
            "text": resp.text,
            "provider": llm.provider,
            "model": llm.model,
            "usage": usage,
            "finishReason": finish_reason,
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
            "temperature": llm.temperature,
        }
        if llm.max_tokens is not None:
            params["max_tokens"] = llm.max_tokens
        else:
            params["max_tokens"] = 4096
        with llm.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text

    elif p == "google":
        model = llm.client.GenerativeModel(
            model_name=llm.model,
            system_instruction=system_prompt,
            generation_config={
                "temperature": llm.temperature,
                "max_output_tokens": llm.max_tokens,
            },
        )
        resp = model.generate_content(user_prompt, stream=True)
        for chunk in resp:
            if chunk.text:
                yield chunk.text

    elif p == "cohere":
        params = {
            "model": llm.model,
            "preamble": system_prompt,
            "message": user_prompt,
            "temperature": llm.temperature,
        }
        if llm.max_tokens is not None:
            params["max_tokens"] = llm.max_tokens
        resp = llm.client.chat_stream(**params)
        for event in resp:
            if event.event_type == "text-generation":
                yield event.text

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
