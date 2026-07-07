"""
Endpoints for uploading and inspecting documents.

Phase 1: just accept a file, save it, return basic info.
(Phase 2 will add the actual forgery-detection logic here.)
"""
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

UPLOAD_DIR = "app/uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

# In-memory "database" for now — Phase 5 will swap this for real SQL storage.
DOCUMENTS_DB = {}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    doc_id = str(uuid.uuid4())
    saved_filename = f"{doc_id}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    contents = await file.read()
    with open(saved_path, "wb") as f:
        f.write(contents)

    record = {
        "id": doc_id,
        "original_filename": file.filename,
        "stored_filename": saved_filename,
        "uploaded_at": datetime.utcnow().isoformat(),
        "status": "Pending",
        "size_bytes": len(contents),
    }
    DOCUMENTS_DB[doc_id] = record

    return record


@router.get("/{doc_id}")
def get_document(doc_id: str):
    record = DOCUMENTS_DB.get(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return record


@router.get("/")
def list_documents():
    return list(DOCUMENTS_DB.values())
