from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from support_agent.runtime.errors import SupportAgentError, TransientDependencyError


T = TypeVar("T")


def run_with_retry(fn: Callable[[], T], *, attempts: int, backoff_seconds: float) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - simple retry wrapper
            last_error = exc
            retryable = isinstance(exc, (TransientDependencyError,)) or (
                isinstance(exc, SupportAgentError) and getattr(exc, "retryable", False)
            )
            if not retryable or attempt >= attempts:
                raise
            time.sleep(backoff_seconds * attempt)
    assert last_error is not None
    raise last_error

