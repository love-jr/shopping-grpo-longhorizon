"""OpenAI-compatible JSON client for frozen Rubric and Judge prompts."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RETRYABLE_HTTP_STATUSES = frozenset(
    {403, 408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
)


class ModelResponseError(ValueError):
    """Raised when a provider response cannot satisfy the JSON contract."""


def _retry_after_seconds(
    error: HTTPError,
    *,
    now: datetime | None = None,
) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (retry_at - current).total_seconds())


class OpenAIJSONClient:
    """Call one chat-completions model without serializing credentials."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 4096,
        timeout: float = 120,
        retries: int = 2,
        retry_delay_seconds: float = 2,
        response_format_json: bool = False,
        thinking: bool = False,
        reasoning_effort: str = "high",
        transport: Callable | None = None,
    ):
        if not str(model).strip():
            raise ValueError("model is required")
        if not str(base_url).strip():
            raise ValueError("base_url is required")
        if not str(api_key):
            raise ValueError("api_key is required")
        if int(max_tokens) < 1:
            raise ValueError("max_tokens must be positive")
        if int(retries) < 0:
            raise ValueError("retries cannot be negative")
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.response_format_json = bool(response_format_json)
        self.thinking = bool(thinking)
        self.reasoning_effort = str(reasoning_effort)
        self.transport = transport

    def _request_payload(self, messages: list[Mapping]) -> dict:
        payload = {
            "model": self.model,
            "messages": deepcopy(messages),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
        }
        if self.model.casefold().startswith("deepseek-v4"):
            if self.thinking:
                payload["thinking"] = {"type": "enabled"}
                payload["reasoning_effort"] = self.reasoning_effort
                payload.pop("temperature", None)
                payload.pop("top_p", None)
            else:
                payload["thinking"] = {"type": "disabled"}
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete_json(self, messages: list[Mapping]) -> dict:
        """Return parsed JSON plus non-secret request metadata."""

        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        payload = self._request_payload(messages)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "shopping-grpo-longhorizon-evaluator/0.1",
        }
        url = f"{self.base_url}/chat/completions"
        started = time.monotonic()
        response = None
        attempts = 0
        retry_http_statuses = []
        retry_wait_seconds = 0.0
        for attempt in range(self.retries + 1):
            attempts = attempt + 1
            try:
                if self.transport is not None:
                    response = self.transport(
                        url,
                        payload,
                        headers,
                        self.timeout,
                    )
                else:
                    request = Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlopen(request, timeout=self.timeout) as raw:
                        response = json.loads(raw.read().decode("utf-8"))
                break
            except HTTPError as exc:
                status = int(exc.code)
                if (
                    status not in RETRYABLE_HTTP_STATUSES
                    or attempt >= self.retries
                ):
                    raise
                retry_http_statuses.append(status)
                exponential = self.retry_delay_seconds * (2**attempt)
                retry_after = _retry_after_seconds(exc)
                delay = max(exponential, retry_after or 0.0)
                retry_wait_seconds += delay
                if delay > 0:
                    time.sleep(delay)
            except (RemoteDisconnected, TimeoutError, URLError):
                if attempt >= self.retries:
                    raise
                delay = self.retry_delay_seconds * (2**attempt)
                retry_wait_seconds += delay
                if delay > 0:
                    time.sleep(delay)
        latency = time.monotonic() - started
        if not isinstance(response, Mapping):
            raise ModelResponseError("provider response must be an object")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelResponseError("provider response is missing choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise ModelResponseError(
                "provider response message.content must be non-empty JSON text"
            )
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(
                "provider response content is not strict JSON"
            ) from exc
        if not isinstance(result, dict):
            raise ModelResponseError(
                "provider response JSON root must be an object"
            )
        usage = response.get("usage")
        usage = deepcopy(dict(usage)) if isinstance(usage, Mapping) else {}
        return {
            "result": result,
            "metadata": {
                "provider_request_id": response.get("id"),
                "provider_model": response.get("model") or self.model,
                "requested_model": self.model,
                "requested_thinking": self.thinking,
                "requested_reasoning_effort": (
                    self.reasoning_effort if self.thinking else None
                ),
                "attempts": attempts,
                "retry_http_statuses": retry_http_statuses,
                "retry_wait_seconds": retry_wait_seconds,
                "latency_seconds": latency,
                "usage": usage,
            },
        }


def client_from_environment(
    *,
    model: str,
    max_tokens: int,
    timeout: float = 120,
    retries: int = 2,
    response_format_json: bool = False,
    thinking: bool = False,
    reasoning_effort: str = "high",
) -> OpenAIJSONClient:
    """Use the same environment-variable convention as Teacher collection."""

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is required")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    return OpenAIJSONClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        response_format_json=response_format_json,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
