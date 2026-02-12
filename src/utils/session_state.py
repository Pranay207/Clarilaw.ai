from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple


class SessionState:
    """
    Simple in-memory session state with optional per-key TTL.
    This is process-local and not shared across workers.
    """

    def __init__(self, default_ttl_seconds: Optional[float] = None, max_items: Optional[int] = None):
        self._default_ttl = default_ttl_seconds
        self._max_items = max_items
        self._lock = threading.Lock()
        self._store: Dict[str, Tuple[Any, Optional[float]]] = {}

    def _now(self) -> float:
        return time.monotonic()

    def _cleanup_expired(self) -> None:
        if not self._store:
            return
        now = self._now()
        expired_keys = [k for k, (_, exp) in self._store.items() if exp is not None and exp <= now]
        for key in expired_keys:
            self._store.pop(key, None)

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        expires_at: Optional[float] = None
        ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        if ttl is not None:
            expires_at = self._now() + float(ttl)

        with self._lock:
            self._cleanup_expired()
            if self._max_items is not None and len(self._store) >= self._max_items:
                # Remove an arbitrary item to respect max size.
                self._store.pop(next(iter(self._store)), None)
            self._store[key] = (value, expires_at)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            self._cleanup_expired()
            if key not in self._store:
                return default
            value, _ = self._store[key]
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            self._cleanup_expired()
            return key in self._store

    def __len__(self) -> int:
        with self._lock:
            self._cleanup_expired()
            return len(self._store)


_DEFAULT_STATE = SessionState()


def get_state() -> SessionState:
    return _DEFAULT_STATE


def set_value(key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
    _DEFAULT_STATE.set(key, value, ttl_seconds=ttl_seconds)


def get_value(key: str, default: Any = None) -> Any:
    return _DEFAULT_STATE.get(key, default=default)


def delete_value(key: str) -> None:
    _DEFAULT_STATE.delete(key)


def clear_state() -> None:
    _DEFAULT_STATE.clear()
