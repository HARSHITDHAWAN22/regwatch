"""
Caches LLM impact assessments keyed on (clause_text, policy_id, prompt_version,
cache_key_extra) so re-processing the same circular, or re-checking a policy
that was already assessed, never wastes an LLM call.

Backed by Redis so the cache is shared across multiple worker processes -
a plain in-memory dict only lives inside one process, which breaks the
moment you run more than one Uvicorn worker (each worker would have its
own separate cache, silently missing hits it should have gotten). If Redis
is unreachable, falls back automatically to the original in-memory LRU
dict - same graceful-degradation instinct as the LLM circuit breaker in
app/reasoning/llm_client.py. The app never hard-depends on Redis being up.

Implemented as a decorator so it wraps the reasoning function transparently -
callers don't need to know caching (or which backend) exists.
"""
import hashlib
import json
import logging
import threading
from collections import OrderedDict
from functools import wraps

import redis
from app.redis_client import get_redis_client

logger = logging.getLogger("regwatch.cache")

_CACHE_MAX_SIZE = 2000
_CACHE_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days - bounds Redis memory growth
_REDIS_KEY_PREFIX = "regwatch:cache:"

# In-memory fallback - only used if Redis is unreachable
_fallback_cache: "OrderedDict[str, dict]" = OrderedDict()
_fallback_lock = threading.Lock()
_redis_warned = False


def _make_key(clause_text: str, policy_id: str, prompt_version: str, extra: str = "") -> str:
    raw = f"{clause_text.strip()}:{policy_id}:{prompt_version}:{extra}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _warn_redis_unavailable_once():
    global _redis_warned
    if not _redis_warned:
        logger.warning("Redis unavailable - cache falling back to in-memory (not shared across workers)")
        _redis_warned = True


def _redis_get(key: str) -> dict | None:
    try:
        client = get_redis_client()
        raw = client.get(_REDIS_KEY_PREFIX + key)
        return json.loads(raw) if raw else None
    except (redis.ConnectionError, redis.TimeoutError):
        _warn_redis_unavailable_once()
        return None


def _redis_set(key: str, value: dict) -> bool:
    """Returns True if the write actually reached Redis, False if it fell through."""
    try:
        client = get_redis_client()
        client.setex(_REDIS_KEY_PREFIX + key, _CACHE_TTL_SECONDS, json.dumps(value))
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        _warn_redis_unavailable_once()
        return False


def _fallback_get(key: str) -> dict | None:
    with _fallback_lock:
        if key in _fallback_cache:
            _fallback_cache.move_to_end(key)
            return dict(_fallback_cache[key])
    return None


def _fallback_set(key: str, value: dict):
    with _fallback_lock:
        _fallback_cache[key] = value
        _fallback_cache.move_to_end(key)
        if len(_fallback_cache) > _CACHE_MAX_SIZE:
            _fallback_cache.popitem(last=False)  # evict least-recently-used


def cached_assessment(func):
    """Decorator: wraps an LLM impact-reasoning call with Redis-backed caching
    (falls back to in-memory LRU if Redis is unreachable).

    Cache key is a hash of the actual clause TEXT (not chunk_id) + policy_id
    + prompt_version + an optional `cache_key_extra` (used to fold in a
    signature of any few-shot feedback examples currently being injected -
    without this, the cache would keep serving a pre-feedback answer even
    after a reviewer's correction should have changed the outcome).

    `cache_key_extra` is keyword-only and stripped before calling the
    wrapped function - it exists purely for cache-key purposes."""

    @wraps(func)
    def wrapper(clause_text: str, policy_id: str, prompt_version: str, *args, cache_key_extra: str = "", **kwargs):
        key = _make_key(clause_text, policy_id, prompt_version, cache_key_extra)

        cached = _redis_get(key)
        if cached is None:
            cached = _fallback_get(key)
        if cached is not None:
            result = dict(cached)
            result["was_cache_hit"] = True
            return result

        # Deliberately call the wrapped function without holding any lock.
        # This is an LLM call that can take seconds; blocking other callers
        # through it would serialize every concurrent assessment across the
        # whole app, which is a far worse bug than the narrow race it would
        # "fix" (two callers computing the same cold entry once each - wastes
        # one redundant LLM call in the rare case both miss simultaneously,
        # but never violates correctness).
        result = func(clause_text, policy_id, prompt_version, *args, **kwargs)
        result["was_cache_hit"] = False

        stored_in_redis = _redis_set(key, result)
        if not stored_in_redis:
            _fallback_set(key, result)

        return result

    return wrapper


def cache_stats() -> dict:
    try:
        client = get_redis_client()
        client.ping()
        size = len(list(client.scan_iter(match=_REDIS_KEY_PREFIX + "*", count=1000)))
        return {"backend": "redis", "size": size, "max_size": "unbounded (30-day TTL)"}
    except (redis.ConnectionError, redis.TimeoutError):
        with _fallback_lock:
            return {"backend": "in-memory-fallback", "size": len(_fallback_cache), "max_size": _CACHE_MAX_SIZE}


def clear_cache():
    """Useful for test isolation, and operationally to force re-assessment
    of everything after a deliberate prompt/pipeline change. Clears both
    backends unconditionally, regardless of which one is currently active."""
    try:
        client = get_redis_client()
        keys = list(client.scan_iter(match=_REDIS_KEY_PREFIX + "*", count=1000))
        if keys:
            client.delete(*keys)
    except (redis.ConnectionError, redis.TimeoutError):
        pass
    with _fallback_lock:
        _fallback_cache.clear()
