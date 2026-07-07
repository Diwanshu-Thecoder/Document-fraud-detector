"""
Automated Document Fraud Detection - Backend
Phase 1: Foundation - upload, store, and serve documents.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routers import documents

app = FastAPI(
    title="Document Fraud Detection API",
    description="ML-driven middleware that flags tampered or fraudulent documents.",
    version="0.1.0",
)

# Allow the frontend (running on a different port during dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files back so the frontend can preview them
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])


@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud Detection API is running"}
