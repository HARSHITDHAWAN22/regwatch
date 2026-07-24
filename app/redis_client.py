"""
Factory for the Redis connection (same Factory pattern used for the LLM
client - one place to configure/swap the backing store).

Redis replaces two things that were previously plain in-memory Python
dicts: background job status, and the LLM-response cache. That mattered
because in-memory state only lives inside a single worker process - run
this app with more than one Uvicorn worker (which you'd want in any real
deployment) and a client polling GET /jobs/{id} could hit a *different*
worker than the one processing the job, and get a false 404 even though
the job is running fine. Redis makes that state shared across all workers.
"""
import redis
from app.config import get_settings

settings = get_settings()

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


def is_redis_available() -> bool:
    """Cheap health check used by callers to decide whether to fall back
    to in-memory behavior instead of raising on every operation."""
    try:
        get_redis_client().ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        return False
