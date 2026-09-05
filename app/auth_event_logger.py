"""
Thin helper around AuthEvent - keeps call sites in auth_routes.py a
one-liner, and keeps the "what gets logged" decision in one place rather
than repeated at every call site.

Failures here are swallowed, not raised - a logging failure (e.g. a
transient DB issue) should never block an actual login/logout from
completing. This mirrors the fail-open philosophy already used for
Redis-backed features elsewhere in the app (cache, job store), applied
here to writes rather than reads.
"""
import logging
from sqlalchemy.orm import Session
from app.models.auth_event import AuthEvent, AuthEventType

logger = logging.getLogger("regwatch.auth_events")


def log_auth_event(
    db: Session,
    event_type: AuthEventType,
    email: str | None = None,
    ip_address: str | None = None,
    detail: str | None = None,
):
    try:
        event = AuthEvent(event_type=event_type, email=email, ip_address=ip_address, detail=detail)
        db.add(event)
        db.commit()
    except Exception as e:
        # Never let audit logging itself break the auth flow it's observing.
        db.rollback()
        logger.error(f"Failed to write auth event {event_type}: {e}")
