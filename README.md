# Document Fraud Detection Pipeline — Phase 1–5 (Complete)

**Phase 1**: upload UI + FastAPI backend storing documents.
**Phase 2**: image forgery checks — Error Level Analysis (ELA) + EXIF metadata.
**Phase 3**: OCR (Tesseract) + anachronism/date consistency checks.
**Phase 4**: a CNN trained from scratch on a synthetic tampered/authentic
dataset, served as an experimental (non-scoring) signal.
**Phase 5**: SQLite persistence (survives restarts) + a reviewer dashboard
with a manual approve/reject override, layered on top of everything above.

## Project structure

```
fraud-detector/
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI app entry point
│   │   ├── database.py                    # SQLAlchemy engine/session (Phase 5)
│   │   ├── models.py                      # Document ORM model (Phase 5)
│   │   ├── routers/
│   │   │   └── documents.py               # upload/list/detail/status-override endpoints
│   │   ├── services/
│   │   │   ├── forgery_detection.py       # ELA + EXIF analysis (Phase 2)
│   │   │   ├── content_analysis.py        # OCR + anachronism checks (Phase 3)
│   │   │   └── cnn_inference.py           # CNN inference (Phase 4)
│   │   ├── ml_models/
│   │   │   ├── model_def.py               # CNN architecture (must match ml/train.py)
│   │   │   └── tamper_classifier.pt       # trained weights
│   │   ├── uploads/                       # uploaded files + ELA overlays land here
│   │   └── fraud_detector.db              # SQLite DB (created on first run, gitignored)
│   └── requirements.txt
├── ml/                                     # training pipeline (separate from the served app)
│   ├── dataset_generation/
│   │   └── generate_dataset.py            # synthetic tampered/authentic dataset generator
│   ├── data/authentic/, data/tampered/    # generated training images (not committed — regenerate)
│   ├── train.py                           # resumable, checkpointed training script
│   └── models/tamper_classifier.pt        # training output (copied into backend/app/ml_models/)
└── frontend/
    └── index.html              # Intake + Review Dashboard tabs (plain HTML/CSS/JS, no build step)
```

## Running it

**0. System dependency — Tesseract OCR**

```bash
# macOS: brew install tesseract
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

**1. Backend**

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Note: the first request after starting the server will be slow (PyTorch's
import/initialization can take 10-60 seconds depending on your machine) —
this is normal, not a bug.

**2. Frontend**

```bash
cd frontend
python3 -m http.server 8080
```

## How the automated score works

| Signal | Weight | Source | Reliability |
|---|---|---|---|
| EXIF shows editing software | +70 | Phase 2 | High — direct evidence |
| Anachronism (tech predates issue date) | +60 | Phase 3 | High — specific, verifiable |
| Future-dated document | +40 | Phase 3 | High — logically certain |
| No EXIF metadata at all | +5 | Phase 2 | Low — common in legit scans |
| ELA hotspot | +15 | Phase 2 | Low — fires on normal text edges too |
| **CNN tampering probability** | **0 (not scored)** | Phase 4 | **Not established on real data — see below** |

`fraud_score >= 50` → Flagged for Review · `< 20` → Approved · else → Pending.

## Phase 4: training the CNN — what actually happened

This section is here on purpose. Building this surfaced three real bugs in
sequence, and fixing them (rather than shipping a plausible-looking but
broken result) is the actual point of the phase.

**1. Weak synthetic tampering signal.** The first dataset version used a
JPEG-recompression trick (double-vs-single compression) to simulate
splicing, mirroring the Phase 2 ELA logic. Measured directly, the signal
was barely above noise (authentic mean z≈3.9 vs tampered mean z≈4.0,
heavily overlapping) — too weak for anything to learn reliably. Fixed by
switching the primary tampering signature to **noise-level inconsistency**
(the pasted patch is rendered with much less scan-noise than the
surrounding document) — a different, independently well-established
forensic cue.

**2. Model collapse from a normalization bug.** Even after fixing the
data, training accuracy itself stayed stuck at chance (~50%) — not just
validation. Debugging traced this to `transforms.Normalize`, which was
using generic natural-image constants (mean/std ≈ 0.5). This document
dataset is near-white with very low contrast (actual mean ≈0.955,
std≈0.062) — those mismatched constants left the network's input
effectively still un-centered, strangling gradient flow at initialization.
Fixed by measuring the dataset's real statistics and normalizing to them.
A quick sanity check (logistic regression on raw pixels hitting 56% test
accuracy) confirmed real signal existed before spending more time on the
CNN itself, isolating "data problem" from "model bug."

**3. Best-epoch checkpointing.** Training was unstable epoch-to-epoch
(some epochs saw validation accuracy swing from 89% down to 50% and back).
The final epoch (20) actually scored *worse* (52.5% val accuracy) than
epoch 18 (95%). The training script now checkpoints and keeps the
best-by-validation-F1 model, not just whatever the last epoch happens to
produce — a real, common ML practice this project surfaced the need for
directly, rather than by assumption.

**Final result, held-out validation (same synthetic distribution as
training):** accuracy 95.0%, precision 96.2%, recall 93.8%, F1 94.97%
(epoch 18/20, 1,280 train / 320 val images).

### The honest limitation: this hasn't been shown to generalize

That 95% is measured on data generated by the **same script** as the
training data — same layout, same synthetic noise model, same rendering
quirks. When tested against differently-generated synthetic certificates
from earlier phases (still synthetic, just a different generator), the
model's predictions collapsed to calling almost everything "tampered" —
including genuinely clean documents. This is a textbook case of a model
that learned its training distribution well without learning a
generalizable notion of "tampering."

Because of this, **the CNN's score is deliberately excluded from the
automated `fraud_score`** — it's surfaced to the reviewer as clearly-labeled
experimental output only. Folding an unreliable signal into automated
scoring would let it override the far more trustworthy EXIF and
anachronism evidence. Closing this gap for real use would need labeled
data from real documents (or at minimum a much more visually diverse
synthetic generator: multiple fonts, layouts, backgrounds, capture
devices) — which is exactly the kind of dataset the original project brief
notes is scarce.

### Regenerating / retraining

```bash
cd ml
pip install torch torchvision pillow numpy
python3 dataset_generation/generate_dataset.py   # writes to ml/data/
python3 train.py --epochs 20 --max-time 600      # writes ml/models/tamper_classifier.pt
cp models/tamper_classifier.pt ../backend/app/ml_models/
```

`--max-time` lets you cap wall-clock time per run; re-running the same
command resumes from the last checkpoint (`ml/models/_checkpoint.pt`)
rather than starting over.

## Other known limitations (Phases 2-3)

**ELA and text edges**: naive Error Level Analysis flags any high-contrast
edge, tampered or not — see Phase 2 code comments. Kept low-weight, used
mainly as a visual aid for the reviewer.

**Anachronism dictionary**: `TECH_RELEASE_DATES` in `content_analysis.py`
is a small hand-curated demo list, not a general knowledge base.

**OCR accuracy**: Tesseract occasionally misreads characters on stylized
fonts, affecting text-extraction precision.

## Phase 5: persistence + reviewer dashboard

The in-memory dict from Phases 1-4 is gone — documents now live in a real
SQLite database (`backend/app/fraud_detector.db`, created automatically on
first run) via SQLAlchemy. Verified this actually persists by uploading a
document, fully restarting the server process, and confirming a fresh
`GET /api/documents/` still returned it.

**New endpoints:**
- `GET /api/documents/?status=Flagged for Review` — list, optionally
  filtered by status
- `PATCH /api/documents/{id}/status` — manual reviewer override (body:
  `{"status": "Approved"}`). Valid values: `Approved`, `Pending`,
  `Flagged for Review`, `Rejected`. The original score and findings are
  preserved regardless — they're the evidence a decision was based on, not
  a verdict to overwrite.

**Dashboard (frontend, "Review Dashboard" tab):** lists every case with
filename, score, status, and timestamp; filterable by status; click a row
to open the full case-file detail (same view as Intake — ELA overlay, OCR
text, CNN probability, findings); Approve / Reject / Re-flag buttons call
the override endpoint directly and show a "reviewed by a human" tag once
used.

## What's next

- Closing the CNN's generalization gap with more diverse training data.
- Swapping SQLite for Postgres for concurrent multi-reviewer use (the
  SQLAlchemy layer makes this a connection-string change, not a rewrite).
- Auth/login so `reviewed_by_human` can record *who* reviewed a case.



