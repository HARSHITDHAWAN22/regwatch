"""
Rate limiter (Adapter pattern) - one abstract interface, swappable backend.

Same philosophy as app/reasoning/llm_client.py's Factory-wrapped LLM client
and app/vectorstore/faiss_store.py's Repository-wrapped index: callers never
know or care which backend is enforcing the limit, only that
`get_rate_limiter().check(key, max_calls, period_seconds)` tells them
whether to proceed.

Two use sites share this one adapter:
  - app/api/auth_routes.py  -> keyed by client IP, protects /auth/login
  - app/reasoning/llm_client.py -> keyed by a fixed "gemini" key, throttles
    outgoing LLM calls before they're made

Backend today: in-memory sliding window (InMemoryRateLimitBackend) - correct
for a single worker process, same tradeoff as the in-memory fallback cache
in app/cache.py. Swapping to Redis later (so the limit is shared across
multiple Uvicorn workers) means writing one new class here and changing
get_rate_limiter() - zero changes to either call site.
"""
import time
import threading
from abc import ABC, abstractmethod
from collections import deque


class RateLimitBackend(ABC):
    @abstractmethod
    def check(self, key: str, max_calls: int, period_seconds: float) -> bool:
        """Returns True if the call is allowed and consumes one unit of
        the key's quota. Returns False if the key is currently over limit
        (caller decides what to do - reject, or block-and-retry)."""
        ...

    def wait_and_check(self, key: str, max_calls: int, period_seconds: float):
        """Blocks the calling thread until the key has capacity, then
        consumes it. Use for background/internal call sites (e.g. LLM
        calls) where waiting is acceptable. Do NOT use this for
        request-serving endpoints - use check() and return 429 instead,
        or a request would hang the connection open."""
        while not self.check(key, max_calls, period_seconds):
            time.sleep(0.1)


class InMemoryRateLimitBackend(RateLimitBackend):
    """Sliding-window counter per key, kept in a plain dict. Thread-safe
    via a single lock - fine at this call volume; a per-key lock would be
    overkill for the number of distinct keys this app actually sees
    (client IPs on login, one fixed key on the LLM path)."""

    def __init__(self):
        self._windows: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, max_calls: int, period_seconds: float) -> bool:
        with self._lock:
            now = time.time()
            window = self._windows.setdefault(key, deque())
            while window and now - window[0] >= period_seconds:
                window.popleft()

            if len(window) >= max_calls:
                return False

            window.append(now)
            return True


_backend: RateLimitBackend | None = None


def get_rate_limiter() -> RateLimitBackend:
    global _backend
    if _backend is None:
        _backend = InMemoryRateLimitBackend()
    return _backend
