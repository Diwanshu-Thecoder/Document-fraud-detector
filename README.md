# Document Fraud Detection Pipeline — Phase 1 + 2

**Phase 1**: upload UI + FastAPI backend storing documents.
**Phase 2**: every upload now runs through two forgery-detection signals
(Error Level Analysis + EXIF metadata) and returns a `fraud_score` (0–100),
a human-readable list of `reasons`, and an ELA visualization image.

## Project structure

```
fraud-detector/
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI app entry point
│   │   ├── routers/
│   │   │   └── documents.py               # /api/documents/upload endpoint
│   │   ├── services/
│   │   │   └── forgery_detection.py       # ELA + EXIF analysis (Phase 2)
│   │   └── uploads/                       # uploaded files + ELA overlays land here
│   └── requirements.txt
└── frontend/
    └── index.html              # upload UI (plain HTML/CSS/JS, no build step)
```

## Running it

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

Visit `http://localhost:8080`, drag in a JPG/PNG. You'll see:
- A **fraud score** (0–100) with a colored bar (green < 20, amber, red ≥ 50)
- A **status stamp**: Approved / Pending / Flagged for Review
- A **findings list** explaining exactly why
- An **ELA overlay** image you can inspect visually

## How the scoring works

| Signal | Weight | Why |
|---|---|---|
| EXIF shows editing software (Photoshop, GIMP, Canva, etc.) | +70 | Direct, reliable evidence — verified in testing |
| No EXIF metadata at all | +5 | Weak signal — common even in legitimate scans |
| ELA hotspot (localized compression anomaly) | +15 | See limitation below — kept low-weight on purpose |

`fraud_score >= 50` → Flagged for Review · `< 20` → Approved · else → Pending.

## Known limitation: ELA and text edges

While building this, testing showed that naive Error Level Analysis flags
**any high-contrast edge** — including completely untampered text and
borders — not just genuinely edited regions. A clean, never-edited
certificate and a deliberately tampered one produced similar ELA hotspot
scores in our tests, because both simply contain printed text.

This is a documented characteristic of ELA in the forensics literature: it's
a strong tool for spotting spliced *photographic* content (e.g. a face
pasted from another photo) but a weak, noisy one for flat text-on-white
documents. Because of this, ELA is weighted low in the score and its main
value here is the **visual overlay** — a human reviewer can look at it
alongside the metadata evidence, rather than trusting a single automated
number.

This is exactly the gap **Phase 4** (a trained CNN) is meant to close: a
model trained on labeled tampered/untampered examples can learn to tell a
normal text edge apart from a spliced one, which hand-written statistical
rules can't reliably do.

## What's next

- **Phase 3**: OCR (Tesseract/EasyOCR) + NLP consistency checks (e.g. dates,
  fonts, logical contradictions in the extracted text).
- **Phase 4**: label a small tampered/untampered image dataset and train a
  CNN to replace/augment the ELA heuristic.
- **Phase 5**: SQL-backed status persistence + polished reviewer dashboard.

