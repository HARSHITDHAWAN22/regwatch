import uuid
import enum
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum
from app.db import Base


class AuthEventType(str, enum.Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    ACCOUNT_LOCKED = "account_locked"
    TOKEN_REVOKED = "token_revoked"


class AuthEvent(Base):
    """
    Append-only audit log for authentication events - same immutability
    philosophy as ImpactAssessment (never overwritten, only appended to).

    Deliberately stores `email` rather than a foreign key to User.id:
    a failed login for a non-existent or since-deleted email still needs
    to be traceable (e.g. "was someone probing for valid accounts?"), and
    a hard FK would make that impossible to log at all.

    Never stores: passwords, raw tokens. `detail` may reference a token's
    jti (safe - it's an opaque ID, not a credential) but never the token
    string itself.
    """
    __tablename__ = "auth_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type = Column(Enum(AuthEventType), nullable=False)
    email = Column(String, nullable=True, index=True)
    ip_address = Column(String, nullable=True)
    detail = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
