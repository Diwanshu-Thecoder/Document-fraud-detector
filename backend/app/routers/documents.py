"""
Endpoints for uploading and inspecting documents.

Phase 2: uploads now run through ELA + EXIF forgery analysis before
being stored, and the response includes a fraud_score and reasons.
"""
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.forgery_detection import analyze_document

router = APIRouter()

UPLOAD_DIR = "app/uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
# PDF forgery analysis (page rasterization) lands in Phase 3 alongside OCR.
ANALYZABLE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Thresholds that decide the document's review status based on fraud_score.
FLAG_THRESHOLD = 50    # >= this -> Flagged for Review
APPROVE_THRESHOLD = 20  # < this -> Approved automatically

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
        "fraud_score": None,
        "reasons": [],
        "ela_image": None,
    }

    if ext in ANALYZABLE_EXTENSIONS:
        report = analyze_document(saved_path, UPLOAD_DIR, doc_id)

        if report.fraud_score >= FLAG_THRESHOLD:
            status = "Flagged for Review"
        elif report.fraud_score < APPROVE_THRESHOLD:
            status = "Approved"
        else:
            status = "Pending"

        record.update({
            "status": status,
            "fraud_score": report.fraud_score,
            "reasons": report.reasons,
            "ela_image": report.ela_image_path,
        })
    else:
        record["reasons"] = ["PDF analysis not yet supported — arrives in Phase 3."]

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
