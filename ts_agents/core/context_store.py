"""
core/context_store.py
─────────────────────
Thread-safe shared state store accessible by all agents.
The Orchestrator uses this as the single source of truth.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class ContextStore:
    """
    Lightweight in-process key-value store.
    Keys are agent names or logical data labels.
    Values are typically AgentResult objects or DataFrames.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._lock = threading.RLock()

    # ── write ─────────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def update(self, mapping: Dict[str, Any]) -> None:
        with self._lock:
            self._store.update(mapping)

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def require(self, key: str) -> Any:
        """Like get() but raises KeyError if missing."""
        with self._lock:
            if key not in self._store:
                raise KeyError(
                    f"ContextStore: required key '{key}' not found. "
                    f"Available: {list(self._store.keys())}"
                )
            return self._store[key]

    def keys(self):
        with self._lock:
            return list(self._store.keys())

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of the entire store."""
        with self._lock:
            return dict(self._store)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    def __repr__(self) -> str:
        with self._lock:
            return f"ContextStore(keys={list(self._store.keys())})"
