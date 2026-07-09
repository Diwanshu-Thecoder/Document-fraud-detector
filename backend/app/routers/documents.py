"""
Endpoints for uploading, inspecting, and reviewing documents.

Phase 5: swaps the in-memory dict for real SQL persistence (SQLite via
SQLAlchemy), and adds a manual review-override endpoint so a human
reviewer can approve or reject a document regardless of what the
automated fraud_score said — the model is a triage aid, not a judge.
"""
import os
import json
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, engine, Base
from app.models import Document
from app.services.forgery_detection import analyze_document
from app.services.content_analysis import analyze_content
from app.services.cnn_inference import classify_tampering

router = APIRouter()

UPLOAD_DIR = "app/uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

FLAG_THRESHOLD = 50
APPROVE_THRESHOLD = 20

VALID_STATUSES = {"Approved", "Pending", "Flagged for Review", "Rejected"}

Base.metadata.create_all(bind=engine)


def _doc_to_dict(doc: Document) -> dict:
    return {
        "id": doc.id,
        "original_filename": doc.original_filename,
        "stored_filename": doc.stored_filename,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
        "status": doc.status,
        "size_bytes": doc.size_bytes,
        "fraud_score": doc.fraud_score,
        "reasons": json.loads(doc.reasons_json or "[]"),
        "ela_image": doc.ela_image,
        "extracted_text": doc.extracted_text,
        "cnn_tamper_probability": doc.cnn_tamper_probability,
        "reviewed_by_human": doc.reviewed_by_human,
    }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
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

    reasons: list[str] = []
    combined_score = 0
    is_pdf = ext == ".pdf"
    ela_image = None
    extracted_text = None
    cnn_prob = None

    if ext in IMAGE_EXTENSIONS:
        forgery_report = analyze_document(saved_path, UPLOAD_DIR, doc_id)
        reasons.extend(forgery_report.reasons)
        combined_score += forgery_report.fraud_score
        ela_image = forgery_report.ela_image_path

        try:
            cnn_result = classify_tampering(saved_path)
            cnn_prob = cnn_result["tampered_probability"]
            reasons.append(
                f"[Experimental] CNN classifier estimates "
                f"{cnn_prob*100:.0f}% tampering probability. This model was "
                f"trained/validated only on synthetic data and has NOT been "
                f"shown to generalize to real documents — informational "
                f"only, not included in the fraud score."
            )
        except Exception as e:
            reasons.append(f"CNN classifier could not run: {e}")

    try:
        content_report = analyze_content(saved_path, is_pdf=is_pdf)
        reasons.extend(content_report.reasons)
        combined_score += content_report.content_score
        extracted_text = content_report.extracted_text
    except Exception as e:
        reasons.append(f"Content analysis could not run: {e}")

    combined_score = min(combined_score, 100)

    if combined_score >= FLAG_THRESHOLD:
        status = "Flagged for Review"
    elif combined_score < APPROVE_THRESHOLD:
        status = "Approved"
    else:
        status = "Pending"

    doc = Document(
        id=doc_id,
        original_filename=file.filename,
        stored_filename=saved_filename,
        status=status,
        size_bytes=len(contents),
        fraud_score=combined_score,
        reasons_json=json.dumps(reasons),
        ela_image=ela_image,
        extracted_text=extracted_text,
        cnn_tamper_probability=cnn_prob,
        reviewed_by_human=False,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return _doc_to_dict(doc)


@router.get("/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_to_dict(doc)


@router.get("/")
def list_documents(status: str | None = None, db: Session = Depends(get_db)):
    """Optionally filter by status, e.g. /api/documents/?status=Flagged for Review"""
    query = db.query(Document).order_by(Document.uploaded_at.desc())
    if status:
        query = query.filter(Document.status == status)
    return [_doc_to_dict(d) for d in query.all()]


class StatusUpdate(BaseModel):
    status: str


@router.patch("/{doc_id}/status")
def update_status(doc_id: str, update: StatusUpdate, db: Session = Depends(get_db)):
    """
    Manual reviewer override — a human can approve, reject, or re-flag a
    document regardless of the automated fraud_score. The score and
    findings are preserved either way, since they're the evidence the
    decision was based on, not a verdict to be erased.
    """
    if update.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(VALID_STATUSES)}",
        )

    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.status = update.status
    doc.reviewed_by_human = True
    db.commit()
    db.refresh(doc)

    return _doc_to_dict(doc)
