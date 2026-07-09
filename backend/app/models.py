"""
Phase 5: ORM model for a reviewed document.

reasons is stored as JSON text (SQLite has no native array type) and
converted to/from a Python list at the API boundary — see routers/documents.py.
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.sql import func

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="Pending", index=True)
    size_bytes = Column(Integer)
    fraud_score = Column(Integer, nullable=True)
    reasons_json = Column(Text, default="[]")
    ela_image = Column(String, nullable=True)
    extracted_text = Column(Text, nullable=True)
    cnn_tamper_probability = Column(Float, nullable=True)
    # Set when a human reviewer overrides the automated status.
    reviewed_by_human = Column(Boolean, default=False)
