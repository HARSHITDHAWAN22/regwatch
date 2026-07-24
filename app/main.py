from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db import init_db
from app.api.routes import router as regwatch_router
from app.api.auth_routes import router as auth_router

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


@app.get("/")
def root():
    return {"service": "RegWatch", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
