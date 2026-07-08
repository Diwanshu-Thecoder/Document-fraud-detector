"""
Endpoints for uploading and inspecting documents.

Phase 3: adds OCR + NLP consistency checks (content_analysis) alongside the
Phase 2 image forgery checks (forgery_detection), and combines both into a
single fraud_score. PDFs now get analyzed too (via OCR), though image-level
forgery checks (ELA/EXIF) remain image-only since they need pixel data,
not PDF structure.
"""
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.forgery_detection import analyze_document
from app.services.content_analysis import analyze_content

router = APIRouter()

UPLOAD_DIR = "app/uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

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
        "extracted_text": None,
    }

    reasons: list[str] = []
    combined_score = 0
    is_pdf = ext == ".pdf"

    # --- Image-level forgery checks (ELA + EXIF) — images only ---
    if ext in IMAGE_EXTENSIONS:
        forgery_report = analyze_document(saved_path, UPLOAD_DIR, doc_id)
        reasons.extend(forgery_report.reasons)
        combined_score += forgery_report.fraud_score
        record["ela_image"] = forgery_report.ela_image_path

    # --- Content checks (OCR + NLP) — images and PDFs ---
    try:
        content_report = analyze_content(saved_path, is_pdf=is_pdf)
        reasons.extend(content_report.reasons)
        combined_score += content_report.content_score
        record["extracted_text"] = content_report.extracted_text
    except Exception as e:
        # OCR failing shouldn't take down the whole upload — surface it as
        # a finding instead so the reviewer knows analysis was incomplete.
        reasons.append(f"Content analysis could not run: {e}")

    combined_score = min(combined_score, 100)

    if combined_score >= FLAG_THRESHOLD:
        status = "Flagged for Review"
    elif combined_score < APPROVE_THRESHOLD:
        status = "Approved"
    else:
        status = "Pending"

    record.update({
        "status": status,
        "fraud_score": combined_score,
        "reasons": reasons,
    })

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
