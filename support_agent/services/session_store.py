from __future__ import annotations

import json
import threading
from typing import Protocol

from support_agent.config import Settings
from support_agent.runtime.errors import PermanentDependencyError, TransientDependencyError


class SessionStore(Protocol):
    def save(self, session_id: str, payload: dict) -> None: ...
    def load(self, session_id: str) -> dict | None: ...
    def append_event(self, session_id: str, payload: dict) -> None: ...
    def load_events(self, session_id: str, start_index: int = 0) -> list[dict]: ...
    def clear_events(self, session_id: str) -> None: ...


class RedisSessionStore:
    def __init__(self, settings: Settings) -> None:
        self.redis_url = settings.redis_url
        self.ttl_seconds = settings.support_ai_session_ttl_seconds
        self._client = None

    def _client_or_raise(self):
        if self._client is not None:
            return self._client
        try:
            import redis
        except ModuleNotFoundError as exc:
            raise PermanentDependencyError("Redis dependency is not installed.") from exc
        if not self.redis_url:
            raise PermanentDependencyError("Redis is not configured. Set REDIS_URL.")
        self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def save(self, session_id: str, payload: dict) -> None:
        client = self._client_or_raise()
        try:
            client.setex(self._key(session_id), self.ttl_seconds, json.dumps(payload, default=str))
        except Exception as exc:
            raise TransientDependencyError(f"Redis session save failed: {exc}") from exc

    def load(self, session_id: str) -> dict | None:
        client = self._client_or_raise()
        try:
            raw = client.get(self._key(session_id))
        except Exception as exc:
            raise TransientDependencyError(f"Redis session load failed: {exc}") from exc
        if not raw:
            return None
        return json.loads(raw)

    def append_event(self, session_id: str, payload: dict) -> None:
        client = self._client_or_raise()
        try:
            client.rpush(self._events_key(session_id), json.dumps(payload, default=str))
            client.expire(self._events_key(session_id), self.ttl_seconds)
        except Exception as exc:
            raise TransientDependencyError(f"Redis session event append failed: {exc}") from exc

    def load_events(self, session_id: str, start_index: int = 0) -> list[dict]:
        client = self._client_or_raise()
        try:
            raw_items = client.lrange(self._events_key(session_id), start_index, -1)
        except Exception as exc:
            raise TransientDependencyError(f"Redis session event load failed: {exc}") from exc
        return [json.loads(item) for item in raw_items]

    def clear_events(self, session_id: str) -> None:
        client = self._client_or_raise()
        try:
            client.delete(self._events_key(session_id))
        except Exception as exc:
            raise TransientDependencyError(f"Redis session event clear failed: {exc}") from exc

    @staticmethod
    def _key(session_id: str) -> str:
        return f"support_ai_session:{session_id}"

    @staticmethod
    def _events_key(session_id: str) -> str:
        return f"support_ai_events:{session_id}"


class InMemorySessionStore:
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        self._events: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def save(self, session_id: str, payload: dict) -> None:
        with self._lock:
            self._store[session_id] = json.loads(json.dumps(payload, default=str))

    def load(self, session_id: str) -> dict | None:
        with self._lock:
            payload = self._store.get(session_id)
        if payload is None:
            return None
        return json.loads(json.dumps(payload, default=str))

    def append_event(self, session_id: str, payload: dict) -> None:
        with self._lock:
            self._events.setdefault(session_id, []).append(json.loads(json.dumps(payload, default=str)))

    def load_events(self, session_id: str, start_index: int = 0) -> list[dict]:
        with self._lock:
            payload = self._events.get(session_id, [])
            return json.loads(json.dumps(payload[start_index:], default=str))

    def clear_events(self, session_id: str) -> None:
        with self._lock:
            self._events[session_id] = []
