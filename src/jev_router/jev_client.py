"""Thin HTTP client for the TypeSafe Jev (System One) API."""

import random
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
RETRYABLE = {429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 30.0
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class JevError(RuntimeError):
    pass


@dataclass(frozen=True)
class JevResult:
    model: str
    answers: dict[str, Any]
    usage: dict[str, int]
    latency_seconds: float


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS:
        return
    raise ValueError("base_url must use https (http is allowed only for localhost)")


class JevClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_URL,
        model: str = "jev-latest",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        _validate_url(base_url)
        self._url = base_url
        self._model = model
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._http = httpx.Client(
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def __repr__(self) -> str:
        return f"JevClient(url={self._url!r}, model={self._model!r})"

    @property
    def model(self) -> str:
        return self._model

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        delay = self._backoff * (2**attempt)
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            delay = max(delay, float(retry_after))
        delay = min(delay, MAX_BACKOFF_SECONDS)
        time.sleep(delay + random.uniform(0, self._backoff))

    def analyze(self, state: str, questions: dict[str, Any]) -> JevResult:
        payload = {"model": self._model, "state": state, "questions": questions}
        last: str | int = "none"
        for attempt in range(self._max_retries):
            retry_after = None
            started = time.perf_counter()
            try:
                response = self._http.post(self._url, json=payload)
            except httpx.TransportError as exc:
                last = type(exc).__name__
            else:
                if response.status_code == 200:
                    try:
                        body = response.json()
                        result = JevResult(
                            model=body.get("model", ""),
                            answers=body["answers"],
                            usage=body.get("usage", {}),
                            latency_seconds=time.perf_counter() - started,
                        )
                    except (ValueError, KeyError, TypeError, AttributeError):
                        last = "malformed_body"
                    else:
                        return result
                else:
                    last = response.status_code
                    if response.status_code not in RETRYABLE:
                        raise JevError(f"Jev request failed with HTTP {last}") from None
                    retry_after = response.headers.get("retry-after")
            if attempt < self._max_retries - 1:
                self._sleep(attempt, retry_after)
        raise JevError(f"Jev request failed after retries (last: {last})") from None

    def close(self) -> None:
        self._http.close()
