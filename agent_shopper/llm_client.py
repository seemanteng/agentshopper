"""Provider-agnostic LLM client bootstrap.

Adapted from the sibling shopify-ucp-explorer project's llm_client.py: a
thin, lazily-imported wrapper so `agent_shopper` has no hard dependency on
`openai`/`anthropic` unless a key is actually configured. OpenAI is tried
first (matching the sibling's existing convention and .env), Anthropic is a
fallback for teams that only have that key.
"""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from pydantic import BaseModel

from agent_shopper.config import LLM_MODEL_ENV, LLM_MODEL_FALLBACK

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when no provider is configured, or the call itself fails.
    Callers (reranker.LLMReranker) must always catch this and fall back --
    never let an LLM failure crash a turn."""


def resolve_model(env_var: str = LLM_MODEL_ENV, fallback: str = LLM_MODEL_FALLBACK) -> str:
    return os.environ.get(env_var) or fallback


def active_provider() -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def call_structured(system_prompt: str, user_payload: dict, schema: Type[T]) -> T:
    """One structured-output call. Raises LLMUnavailable on any failure
    (missing key, network error, malformed response) -- always catch this."""
    provider = active_provider()
    if provider == "openai":
        return _call_openai(system_prompt, user_payload, schema)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_payload, schema)
    raise LLMUnavailable("no OPENAI_API_KEY or ANTHROPIC_API_KEY configured")


def _call_openai(system_prompt: str, user_payload: dict, schema: Type[T]) -> T:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailable("openai package not installed") from exc
    try:
        client = OpenAI()
        response = client.responses.parse(
            model=resolve_model(),
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise LLMUnavailable("openai response had no parsed output")
        return parsed
    except LLMUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 -- any provider/network failure degrades, never crashes
        raise LLMUnavailable(f"openai call failed: {exc}") from exc


def _call_anthropic(system_prompt: str, user_payload: dict, schema: Type[T]) -> T:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailable("anthropic package not installed") from exc
    try:
        client = anthropic.Anthropic()
        model = resolve_model("AGENT_SHOPPER_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        schema_hint = json.dumps(schema.model_json_schema())
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            system=f"{system_prompt}\n\nRespond with ONLY JSON matching this schema:\n{schema_hint}",
            messages=[{"role": "user", "content": json.dumps(user_payload)}],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        return schema.model_validate_json(text)
    except LLMUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMUnavailable(f"anthropic call failed: {exc}") from exc
