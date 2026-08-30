from fastapi import APIRouter, Depends, HTTPException,Request
from sqlalchemy.orm import Session
from app.db import get_db
from app.models.user import User, Role
from app.schemas import UserCreate, UserLogin, Token
from app.auth import hash_password, verify_password, create_access_token, require_role
from app.rate_limiter import get_rate_limiter

router = APIRouter(prefix="/auth", tags=["auth"])

LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 60


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
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.id, user.role.value)
    return Token(access_token=token)
