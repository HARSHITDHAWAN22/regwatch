from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db import init_db, SessionLocal
from app.api.routes import router as regwatch_router
from app.api.auth_routes import router as auth_router
from app.models.user import User, Role
from app.auth import hash_password

app = FastAPI(
    title="RegWatch",
    description="Regulatory circular impact analyzer with auditable, RAG-based impact assessment.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(regwatch_router)


@app.on_event("startup")
def on_startup():
    init_db()
    seed_admin_if_empty()


def seed_admin_if_empty():
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.email == "admin@regwatch.demo").first():
            admin = User(
                email="admin@regwatch.demo",
                hashed_password=hash_password("admin123"),
                role=Role.ADMIN,
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()


@app.get("/")
def root():
    return {"service": "RegWatch", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
