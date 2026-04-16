from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any

from genai_pyo3 import (
    ChatMessage,
    ChatOptions,
    ChatRequest,
    ChatResponse,
    Client,
    JsonSpec,
    Tool,
)
from pydantic import BaseModel, RootModel

logger = logging.getLogger(__name__)


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Synchronous wrapper called from a running event loop")


def response_text(response: ChatResponse) -> str:
    """Coalesce a :class:`ChatResponse`'s text segments into a single string.

    Prefers ``first_text()`` when it is non-empty; falls back to joining
    every non-empty segment in ``texts()``. Returns ``""`` when the
    response carries no text at all (e.g. a pure tool-call response).
    """
    first = response.first_text()
    if first:
        return first
    return "\n".join(segment for segment in response.texts() if segment)


def _is_root_model_type(schema_model: type[BaseModel]) -> bool:
    return issubclass(schema_model, RootModel)


def _validate_schema_response(schema_model: type[BaseModel], text: str) -> BaseModel:
    try:
        return schema_model.model_validate_json(text)
    except Exception:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, list) and "results" in schema_model.model_fields:
            return schema_model.model_validate({"results": parsed_json})
        raise


@dataclass(slots=True)
class NativeToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    handler: Any

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(**arguments)
        return await asyncio.to_thread(self.handler, **arguments)

    def invoke(self, arguments: dict[str, Any] | None = None) -> Any:
        return _run_coro_sync(self.ainvoke(arguments or {}))

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.schema


class AsyncLLMClient:
    """Native async wrapper around genai-pyo3 for sourcehunt/runtime use.

    This intentionally bypasses LangChain's message/result model and exposes
    only the pieces Clearwing actually needs: text, tool calls, usage, and
    bounded concurrency.
    """

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str,
        api_key: str,
        base_url: str | None = None,
        max_concurrency: int = 4,
        default_system: str = "You are a helpful assistant.",
        rate_limit_max_retries: int = 6,
        rate_limit_initial_backoff_seconds: float = 1.0,
        rate_limit_max_backoff_seconds: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.provider_name = provider_name
        self.api_key = api_key
        self.base_url = base_url
        self.default_system = default_system
        self.rate_limit_max_retries = max(0, rate_limit_max_retries)
        self.rate_limit_initial_backoff_seconds = max(0.1, rate_limit_initial_backoff_seconds)
        self.rate_limit_max_backoff_seconds = max(
            self.rate_limit_initial_backoff_seconds,
            rate_limit_max_backoff_seconds,
        )
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def achat(
        self,
        *,
        messages: list[ChatMessage],
        system: str | None = None,
        tools: list[NativeToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: type[BaseModel] | None = None,
        response_schema_name: str | None = None,
        response_schema_description: str | None = None,
    ) -> ChatResponse:
        request_tools = None
        if tools:
            request_tools = [
                Tool(
                    tool.name,
                    tool.description,
                    json.dumps(tool.schema),
                )
                for tool in tools
            ]

        request = ChatRequest(
            messages=list(messages),
            system=system or self.default_system,
            tools=request_tools,
        )
        options = ChatOptions(
            temperature=temperature,
            max_tokens=max_tokens,
            capture_content=True,
            capture_usage=True,
            capture_tool_calls=True,
            response_json_spec=(
                _json_spec_from_model(
                    response_schema,
                    name=response_schema_name,
                    description=response_schema_description,
                )
                if response_schema is not None
                else None
            ),
        )

        async with self._semaphore:
            client = self._build_client(Client)
            response = await self._with_rate_limit_retries(
                lambda: self._achat_with_provider_policy(client, request, options)
            )
        return response

    def chat(self, **kwargs: Any) -> ChatResponse:
        return _run_coro_sync(self.achat(**kwargs))

    async def aask_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: type[BaseModel] | None = None,
        response_schema_name: str | None = None,
        response_schema_description: str | None = None,
    ) -> ChatResponse:
        return await self.achat(
            messages=[ChatMessage("user", user)],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
            response_schema_name=response_schema_name,
            response_schema_description=response_schema_description,
        )

    async def aask_json(
        self,
        *,
        system: str,
        user: str,
        expect: str = "object",
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema_model: type[BaseModel] | None = None,
        schema_name: str | None = None,
        schema_description: str | None = None,
    ) -> tuple[Any, ChatResponse]:
        response = await self.aask_text(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=schema_model,
            response_schema_name=schema_name,
            response_schema_description=schema_description,
        )
        text = response_text(response)
        if schema_model is not None:
            parsed_model = _validate_schema_response(schema_model, text)
            if _is_root_model_type(schema_model):
                return parsed_model.root, response
            return parsed_model.model_dump(), response
        if expect == "array":
            return extract_json_array(text), response
        return extract_json_object(text), response

    def _build_client(self, client_cls):
        base_url = self.base_url
        if base_url:
            base_url = base_url if base_url.endswith("/") else f"{base_url}/"
            if self.api_key:
                return client_cls.with_api_key_and_base_url(
                    self.provider_name,
                    self.api_key,
                    base_url,
                )
            return client_cls.with_base_url(self.provider_name, base_url)
        if self.api_key:
            return client_cls.with_api_key(self.provider_name, self.api_key)
        return client_cls()

    async def _achat_with_provider_policy(
        self,
        client: Client,
        request: ChatRequest,
        options: ChatOptions,
    ) -> ChatResponse:
        # openai_resp backends that require `stream=true` (e.g. our local
        # gateway) reject `exec_chat`. genai-pyo3's `achat_via_stream`
        # streams internally and hands back a fully-collected ChatResponse,
        # so callers never see chunk events.
        if self.provider_name == "openai_resp":
            return await client.achat_via_stream(self.model_name, request, options)
        return await client.achat(self.model_name, request, options)

    async def _with_rate_limit_retries(self, op) -> ChatResponse:
        attempt = 0
        while True:
            try:
                return await op()
            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt >= self.rate_limit_max_retries:
                    raise

                delay = self._retry_delay_seconds(exc, attempt)
                attempt += 1
                logger.warning(
                    "LLM call rate-limited for model=%s provider=%s; retrying in %.2fs (attempt %d/%d): %s",
                    self.model_name,
                    self.provider_name,
                    delay,
                    attempt,
                    self.rate_limit_max_retries,
                    exc,
                )
                await asyncio.sleep(delay)

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            " 429" in text
            or text.startswith("429")
            or "status code 429" in text
            or "too many requests" in text
            or "rate limit" in text
            or "ratelimit" in text
        )

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        retry_after = self._parse_retry_after_seconds(str(exc))
        if retry_after is not None:
            base_delay = retry_after
        else:
            base_delay = min(
                self.rate_limit_initial_backoff_seconds * (2**attempt),
                self.rate_limit_max_backoff_seconds,
            )

        jitter = min(1.0, base_delay * 0.2) * random.random()
        return min(base_delay + jitter, self.rate_limit_max_backoff_seconds)

    def _parse_retry_after_seconds(self, text: str) -> float | None:
        patterns = [
            r"retry[- ]after[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"try again in\s*([0-9]+(?:\.[0-9]+)?)s",
            r"wait\s*([0-9]+(?:\.[0-9]+)?)s",
        ]
        lowered = text.lower()
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return None
        return None


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("response did not contain a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("response JSON was not an object")
    return parsed


def extract_json_array(text: str) -> list[Any]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("response did not contain a JSON array")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, list):
        raise ValueError("response JSON was not an array")
    return parsed


def _json_spec_from_model(
    schema_model: type[BaseModel],
    *,
    name: str | None = None,
    description: str | None = None,
) -> JsonSpec:
    schema = schema_model.model_json_schema()
    return JsonSpec(
        name=name or _schema_name_for_model(schema_model),
        schema_json=json.dumps(schema),
        description=description,
    )


def _schema_name_for_model(schema_model: type[BaseModel]) -> str:
    raw_name = getattr(schema_model, "__name__", "response_schema")
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name).strip("_")
    return normalized or "response_schema"
