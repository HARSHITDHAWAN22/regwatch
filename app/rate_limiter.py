"""
Rate limiter (Adapter pattern) - one abstract interface, swappable backend.

Same philosophy as app/reasoning/llm_client.py's Factory-wrapped LLM client
and app/vectorstore/faiss_store.py's Repository-wrapped index: callers never
know or care which backend is enforcing the limit, only that
`get_rate_limiter().check(key, max_calls, period_seconds)` tells them
whether to proceed.

Use sites sharing this one adapter:
  - app/api/auth_routes.py -> login rate limit, keyed by client IP
    (check(): every attempt, success or failure, consumes quota)
  - app/api/auth_routes.py -> account lockout, keyed by email
    (peek()/record_failure()/clear(): only FAILED attempts consume quota,
    so real users logging in successfully never get locked out)
  - app/reasoning/llm_client.py -> throttles outgoing LLM calls, one
    fixed key (wait_and_check(): blocks rather than rejecting, since
    this runs in a background job, not a live HTTP request)

Backend today: in-memory sliding window (InMemoryRateLimitBackend) - correct
for a single worker process, same tradeoff as the in-memory fallback cache
in app/cache.py. Swapping to Redis later (so limits are shared across
multiple Uvicorn workers) means writing one new class here and changing
get_rate_limiter() - zero changes to any call site.
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
        (caller decides what to do - reject, or block-and-retry).

        Use this when EVERY call should count toward the limit (e.g. rate
        limiting login attempts by IP - every request, success or failure,
        consumes quota)."""
        ...

    @abstractmethod
    def peek(self, key: str, max_calls: int, period_seconds: float) -> bool:
        """Returns True if the key is currently under its limit, WITHOUT
        consuming any quota. Takes period_seconds (not just max_calls) so
        it can correctly prune entries that have aged out of the window -
        without pruning, a key that once hit its limit would appear
        permanently over-limit, since nothing else would ever clean up
        its stale entries once callers stop reaching the point where
        check()/record_failure() run. Use this to check lockout status
        before deciding whether to even attempt an operation, when
        consumption should only happen conditionally afterward (see
        record_failure)."""
        ...

    @abstractmethod
    def record_failure(self, key: str, period_seconds: float):
        """Consumes one unit of the key's quota. Use this when only FAILED
        attempts should count (e.g. account lockout - a correct password
        should never move the counter closer to a lockout)."""
        ...

    @abstractmethod
    def clear(self, key: str):
        """Resets a key's quota entirely. Use this on success, so past
        failures don't linger and eventually cause an unrelated future
        lockout (e.g. clear the lockout counter on a successful login)."""
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
    (client IPs on login, one fixed key on the LLM path, emails on
    lockout)."""

    def __init__(self):
        self._windows: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, period_seconds: float) -> deque:
        now = time.time()
        window = self._windows.setdefault(key, deque())
        while window and now - window[0] >= period_seconds:
            window.popleft()
        return window

    def check(self, key: str, max_calls: int, period_seconds: float) -> bool:
        with self._lock:
            window = self._prune(key, period_seconds)
            if len(window) >= max_calls:
                return False
            window.append(time.time())
            return True

    def peek(self, key: str, max_calls: int, period_seconds: float) -> bool:
        with self._lock:
            window = self._prune(key, period_seconds)
            return len(window) < max_calls

    def record_failure(self, key: str, period_seconds: float):
        with self._lock:
            window = self._prune(key, period_seconds)
            window.append(time.time())

    def clear(self, key: str):
        with self._lock:
            self._windows.pop(key, None)


_backend: RateLimitBackend | None = None


def get_rate_limiter() -> RateLimitBackend:
    global _backend
    if _backend is None:
        _backend = InMemoryRateLimitBackend()
    return _backend
