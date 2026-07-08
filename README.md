# Document Fraud Detection Pipeline — Phase 1–3

**Phase 1**: upload UI + FastAPI backend storing documents.
**Phase 2**: image forgery checks — Error Level Analysis (ELA) + EXIF metadata.
**Phase 3**: OCR (Tesseract) extracts document text; NLP-style rule checks
flag anachronisms (e.g. a 2021-dated certificate mentioning GPT-4, which
didn't exist until 2023) and nonsensical dates. Works on images *and* PDFs
now (PDFs get rasterized page-by-page before OCR).

## Project structure

```
fraud-detector/
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI app entry point
│   │   ├── routers/
│   │   │   └── documents.py               # /api/documents/upload endpoint
│   │   ├── services/
│   │   │   ├── forgery_detection.py       # ELA + EXIF analysis (Phase 2)
│   │   │   └── content_analysis.py        # OCR + anachronism/date checks (Phase 3)
│   │   └── uploads/                       # uploaded files + ELA overlays land here
│   └── requirements.txt
└── frontend/
    └── index.html              # upload UI (plain HTML/CSS/JS, no build step)
```

## Running it

**0. System dependency — Tesseract OCR** (not a Python package, install separately)

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
# and make sure the install folder is on your PATH.
```

**1. Backend**

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs.

**2. Frontend**

```bash
cd frontend
python3 -m http.server 8080
```

Visit `http://localhost:8080`, drag in a JPG/PNG/PDF. You'll see:
- A **fraud score** (0–100) with a colored bar (green < 20, amber, red ≥ 50)
- A **status stamp**: Approved / Pending / Flagged for Review
- A **findings list** explaining exactly why
- An **ELA overlay** image (images only)
- The **extracted OCR text** (images and PDFs)

## How the scoring works

| Signal | Weight | Source | Why |
|---|---|---|---|
| EXIF shows editing software | +70 | Phase 2 | Direct, reliable evidence |
| Anachronism (tech mentioned predates issue date) | +60 | Phase 3 | Specific, verifiable evidence |
| Future-dated document | +40 | Phase 3 | Logically impossible |
| No EXIF metadata at all | +5 | Phase 2 | Weak signal, common in legitimate scans |
| ELA hotspot | +15 | Phase 2 | See limitation below — kept low-weight |

All applicable signals are summed and capped at 100.
`fraud_score >= 50` → Flagged for Review · `< 20` → Approved · else → Pending.

## Known limitations

**ELA and text edges** (Phase 2): naive Error Level Analysis flags any
high-contrast edge — including untampered text — not just genuinely edited
regions. In testing, a clean and a tampered certificate produced similar
ELA scores because both simply contain printed text. This is a documented
characteristic of ELA in the forensics literature — strong for spliced
*photographic* content, weak for flat text-on-white documents. It's
weighted low here and kept mainly as a visual aid for the reviewer.

**Anachronism detection is only as good as its reference dictionary**
(Phase 3): `TECH_RELEASE_DATES` in `content_analysis.py` is a small,
hand-curated list for demonstration. A document mentioning a real
anachronism *not* in that dictionary won't be caught. In production this
would be backed by a maintained database of product/release dates.

**OCR accuracy**: Tesseract isn't perfect, especially on stylized fonts or
low-resolution scans — it occasionally misreads characters (we saw "to" →
"ta" in testing). This mainly affects date/entity extraction precision,
which is why date-extraction regex is deliberately permissive.

Both limitations point at the same thing: **Phase 4's trained CNN** exists
specifically to learn patterns (real tampering vs. normal text edges) that
hand-written rules can't reliably capture.

## What's next

- **Phase 4**: label a small tampered/untampered image dataset and train a
  CNN to replace/augment the ELA heuristic.
- **Phase 5**: SQL-backed status persistence + polished reviewer dashboard.


