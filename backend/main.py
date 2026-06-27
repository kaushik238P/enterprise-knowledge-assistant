# backend/main.py

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.logging import setup_logging

setup_logging()

from backend.api.chat import router as chat_router
from backend.api.search import router as search_router
from backend.api.ingest import router as ingest_router
from backend.api.documents import router as documents_router

logger = logging.getLogger(__name__)

_ALLOWED_ORIGINS: list[str] = [
    "http://localhost:3000",
    "http://localhost:8501",
    "http://localhost:8080",
    "http://localhost:5173",
]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Enterprise Knowledge Assistant started.")
    yield
    logger.info("Enterprise Knowledge Assistant stopped.")

APP_NAME = "Enterprise Knowledge Assistant"
APP_VERSION = "1.0.0"

app = FastAPI(
    title=APP_NAME,
    description="Enterprise-grade Retrieval-Augmented Generation API.",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(chat_router)
app.include_router(search_router)
app.include_router(ingest_router)


@app.get("/", summary="Root", tags=["Health"])
def root() -> dict[str, str]:
    logger.info("Incoming request: GET /")
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "healthy",
    }


@app.get("/health", summary="Health check", tags=["Health"])
def health() -> dict[str, str]:
    logger.info("Incoming request: GET /health")
    return {"status": "healthy"}

app.include_router(documents_router)