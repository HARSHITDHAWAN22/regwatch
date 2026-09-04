"""
Minimal JWT auth with role-based access control (RBAC guard pattern).
Three roles: admin (manage policy registry), reviewer (confirm/correct
impact assessments), viewer (read-only). Every mutating audit-log action
records which authenticated user performed it.
"""
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.config import get_settings
from app.db import get_db
from app.models.user import User, Role
import uuid
import redis
from app.redis_client import get_redis_client



settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
_REVOKED_KEY_PREFIX = "regwatch:revoked:"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": user_id, "role": role, "exp": expire, "jti": str(uuid.uuid4())}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def revoke_token(jti: str, expires_at: datetime):
    ttl_seconds = max(int((expires_at - datetime.utcnow()).total_seconds()), 1)
    try:
        client = get_redis_client()
        client.setex(_REVOKED_KEY_PREFIX + jti, ttl_seconds, "1")
    except (redis.ConnectionError, redis.TimeoutError):
        pass


def is_token_revoked(jti: str) -> bool:
    try:
        client = get_redis_client()
        return client.exists(_REVOKED_KEY_PREFIX + jti) == 1
    except (redis.ConnectionError, redis.TimeoutError):
        return False


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        jti = payload.get("jti")
        if user_id is None:
            raise credentials_exception
        if jti and is_token_revoked(jti):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user


def require_role(*allowed_roles: Role):
    """Dependency factory - use as Depends(require_role(Role.ADMIN))"""

    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions for this action")
        return user

    return checker
