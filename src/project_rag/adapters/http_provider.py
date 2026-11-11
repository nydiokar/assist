"""Shared HTTP plumbing for model-provider adapters (S3): one httpx client, explicit timeouts,
bounded retries (tenacity) and translation of transport failures into port errors (R6).

Transient failures (timeout, refused/reset connection, HTTP 5xx/429) are retried; contract
problems (other 4xx, malformed JSON) are not.
"""

from __future__ import annotations

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..domain.config import RetryPolicy
from ..domain.errors import ProviderContractViolation, ProviderTimeout, ProviderUnavailable


class ProviderClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float,
        connect_timeout_s: float,
        retry: RetryPolicy,
        api_key: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            headers=headers,
            transport=transport,
        )
        self._retry = retry
        self.base_url = base_url

    def post_json(self, path: str, payload: dict[str, object]) -> object:
        """POST and return the decoded JSON body, retrying transient failures."""
        retrying = Retrying(
            stop=stop_after_attempt(max(1, self._retry.attempts)),
            wait=wait_exponential(multiplier=self._retry.backoff_s, max=self._retry.max_backoff_s),
            retry=retry_if_exception_type((ProviderTimeout, ProviderUnavailable)),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self._post_once(path, payload)
        raise ProviderUnavailable(f"{self.base_url}{path}: no attempt made")  # pragma: no cover

    def _post_once(self, path: str, payload: dict[str, object]) -> object:
        url = f"{self.base_url}{path}"
        try:
            resp = self._client.post(path, json=payload)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(f"{url}: {type(e).__name__}") from e
        except httpx.TransportError as e:  # refused, reset, DNS, protocol errors
            raise ProviderUnavailable(f"{url}: {type(e).__name__}: {e}") from e
        if resp.status_code >= 500 or resp.status_code == 429:
            raise ProviderUnavailable(f"{url}: HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise ProviderContractViolation(f"{url}: HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderContractViolation(f"{url}: response is not JSON") from e

    def get_ok(self, path: str) -> bool:
        """Liveness probe (no retries): True when the server answers 2xx."""
        try:
            return self._client.get(path).is_success
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()
