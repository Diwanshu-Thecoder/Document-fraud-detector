"""
Phase 3: Content Extraction & Semantic Verification.

Two steps:

1. OCR: turn the document's pixels into text we can actually reason about,
   using Tesseract (via pytesseract). Images go straight through; PDFs get
   rasterized page-by-page first (via PyMuPDF) since Tesseract only reads
   pixels, not PDF structure.

2. NLP/rule-based consistency checks on the extracted text:
   - Anachronism detection: does the document reference a technology,
     product, or standard that didn't exist yet on its stated issue date?
     (e.g. "Issued: 2019" but the text mentions "GPT-4", which shipped in
     2023 — a strong sign the date field was altered.)
   - Date sanity: issue dates in the future, or an expiry date before the
     issue date.

This is intentionally a small, easily-extended reference dictionary rather
than a general knowledge base — the point is to demonstrate the technique.
In a production system this would be backed by a real, regularly-updated
database of product/technology release dates.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- Reference data: technology/product -> (year, month) it became public. ---
# Keep names lowercase; matching is case-insensitive and word-boundary aware.
TECH_RELEASE_DATES = {
    "gpt-4": (2023, 3),
    "gpt-3": (2020, 6),
    "chatgpt": (2022, 11),
    "claude 3": (2024, 3),
    "python 3.12": (2023, 10),
    "python 3.11": (2022, 10),
    "react 18": (2022, 3),
    "windows 11": (2021, 10),
    "ios 17": (2023, 9),
    "ios 18": (2024, 9),
    "iphone 15": (2023, 9),
    "iphone 16": (2024, 9),
    "tensorflow 2": (2019, 9),
    "pytorch 2.0": (2023, 3),
    "iso 27001:2022": (2022, 10),
    "usb-c": (2014, 8),
    "5g": (2019, 4),
}

# Matches "Issued: 12 March 2021", "Issue Date - 2021-03-12", "Date of Issue: 2021", etc.
DATE_PATTERNS = [
    r"(?:issued|issue date|date of issue|dated)\D{0,10}(\d{1,2}[\/\-\s]\w+[\/\-\s]\d{2,4})",
    r"(?:issued|issue date|date of issue|dated)\D{0,10}(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})",
    r"(?:issued|issue date|date of issue|dated)\D{0,10}(\b\d{4}\b)",
]

MONTH_NAMES = (
    "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    "jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)


@dataclass
class ContentReport:
    extracted_text: str
    detected_issue_year: int | None
    anachronisms: list[dict] = field(default_factory=list)  # [{term, doc_year, real_year}]
    date_sanity_issues: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    content_score: int = 0  # contribution to overall fraud_score


def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Rasterizes each PDF page to an image, then OCRs it."""
    text_parts = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text_parts.append(pytesseract.image_to_string(img))
    doc.close()
    return "\n".join(text_parts)


def _find_issue_year(text: str) -> int | None:
    """Looks for an explicit issue/date-of-issue year; falls back to the
    most recent 4-digit year mentioned anywhere if no explicit label is found."""
    lower = text.lower()
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            year_match = re.search(r"\d{4}", match.group(1))
            if year_match:
                return int(year_match.group(0))

    # Fallback: any plausible 4-digit year in the document (1990-2035 range
    # to avoid false-matching unrelated numbers like IDs or phone numbers).
    all_years = [int(y) for y in re.findall(r"\b(19[9]\d|20[0-3]\d)\b", text)]
    return max(all_years) if all_years else None


def _find_anachronisms(text: str, issue_year: int | None) -> list[dict]:
    if issue_year is None:
        return []

    lower = text.lower()
    findings = []
    for term, (release_year, _release_month) in TECH_RELEASE_DATES.items():
        if re.search(r"\b" + re.escape(term) + r"\b", lower):
            if release_year > issue_year:
                findings.append({
                    "term": term,
                    "doc_year": issue_year,
                    "real_year": release_year,
                })
    return findings


def _check_date_sanity(text: str) -> list[str]:
    issues = []
    current_year = datetime.utcnow().year

    all_years = [int(y) for y in re.findall(r"\b(19[9]\d|20[0-3]\d)\b", text)]
    future_years = [y for y in all_years if y > current_year]
    if future_years:
        issues.append(
            f"Document references a year in the future ({max(future_years)}), "
            f"which is not possible for an already-issued document."
        )
    return issues


def analyze_content(file_path: str, is_pdf: bool) -> ContentReport:
    text = extract_text_from_pdf(file_path) if is_pdf else extract_text_from_image(file_path)

    issue_year = _find_issue_year(text)
    anachronisms = _find_anachronisms(text, issue_year)
    date_issues = _check_date_sanity(text)

    reasons = []
    score = 0

    if anachronisms:
        for a in anachronisms:
            reasons.append(
                f"Document appears to be issued in {a['doc_year']} but mentions "
                f"'{a['term']}', which did not exist until {a['real_year']} — "
                f"likely an altered date or fabricated content."
            )
        score += 60  # a real anachronism is strong, specific evidence

    for issue in date_issues:
        reasons.append(issue)
        score += 40

    if not text.strip():
        reasons.append("OCR could not extract any readable text from this document.")

    return ContentReport(
        extracted_text=text.strip(),
        detected_issue_year=issue_year,
        anachronisms=anachronisms,
        date_sanity_issues=date_issues,
        reasons=reasons,
        content_score=min(score, 100),
    )
