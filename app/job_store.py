"""
Tracks background job status (upload -> processing -> completed/failed).

Backed by Redis so job status is visible to every worker process, not just
the one that happened to pick up the background task. Falls back to an
in-memory dict if Redis is unreachable - the app still works with a single
worker in that case, it just loses the multi-worker guarantee, exactly the
same tradeoff as the cache module.
"""
import json
import threading
import redis
from app.redis_client import get_redis_client

_REDIS_KEY_PREFIX = "regwatch:job:"
_JOB_TTL_SECONDS = 60 * 60 * 24  # 24 hours - bounds growth, jobs are short-lived by nature

_fallback_jobs: dict[str, dict] = {}
_fallback_lock = threading.Lock()


def set_job(job_id: str, data: dict):
    try:
        client = get_redis_client()
        client.setex(_REDIS_KEY_PREFIX + job_id, _JOB_TTL_SECONDS, json.dumps(data))
        return
    except (redis.ConnectionError, redis.TimeoutError):
        pass
    with _fallback_lock:
        _fallback_jobs[job_id] = data


def update_job(job_id: str, **fields):
    """Merge fields into an existing job record (read-modify-write)."""
    current = get_job(job_id) or {}
    current.update(fields)
    set_job(job_id, current)


def get_job(job_id: str) -> dict | None:
    try:
        client = get_redis_client()
        raw = client.get(_REDIS_KEY_PREFIX + job_id)
        if raw is not None:
            return json.loads(raw)
        # Deliberately fall through to in-memory too: if Redis is up but this
        # particular job was created before Redis became available (or vice
        # versa), we still want a chance to find it rather than returning
        # a false "not found".
    except (redis.ConnectionError, redis.TimeoutError):
        pass
    with _fallback_lock:
        return _fallback_jobs.get(job_id)
