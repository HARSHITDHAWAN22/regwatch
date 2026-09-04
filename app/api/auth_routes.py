from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.user import User, Role
from app.schemas import UserCreate, UserLogin, Token
from app.auth import (hash_password, verify_password, create_access_token, require_role, revoke_token, oauth2_scheme,)
from app.rate_limiter import get_rate_limiter
from datetime import datetime
from jose import jwt, JWTError
from app.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/auth", tags=["auth"])

LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 15 * 60

LOCKOUT_MAX_FAILURES = 5
LOCKOUT_WINDOW_SECONDS = 15 * 60


@router.post("/register", response_model=Token)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=payload.email, hashed_password=hash_password(payload.password), role=Role.VIEWER)
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, user.role.value)
    return Token(access_token=token)


@router.post("/login", response_model=Token)
def login(request: Request, payload: UserLogin, db: Session = Depends(get_db)):
    client_ip = request.client.host if request.client else "unknown"
    if not get_rate_limiter().check(f"login:{client_ip}", LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECONDS):
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again shortly.")

    lockout_key = f"lockout:{payload.email.lower()}"
    if not get_rate_limiter().peek(lockout_key, LOCKOUT_MAX_FAILURES, LOCKOUT_WINDOW_SECONDS):
        raise HTTPException(
            status_code=403,
            detail=f"Account temporarily locked after {LOCKOUT_MAX_FAILURES} failed attempts. Try again later."
        )

    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        get_rate_limiter().record_failure(lockout_key, LOCKOUT_WINDOW_SECONDS)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    get_rate_limiter().clear(lockout_key)
    token = create_access_token(user.id, user.role.value)
    return Token(access_token=token)


@router.post("/logout")
def logout(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    jti = payload.get("jti")
    exp = payload.get("exp")
    if jti and exp:
        revoke_token(jti, datetime.utcfromtimestamp(exp))

    return {"status": "logged out"}
