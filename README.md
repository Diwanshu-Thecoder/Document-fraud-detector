# Document Fraud Detection Pipeline — Phase 1

Foundation: upload a document through a web UI, backend stores it and returns
its metadata. No fraud detection logic yet — that's Phase 2.

## Project structure

```
fraud-detector/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── routers/
│   │   │   └── documents.py   # /api/documents/upload endpoint
│   │   ├── services/          # (empty for now — Phase 2 logic goes here)
│   │   └── uploads/           # uploaded files land here
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

Visit `http://localhost:8000/docs` to see the interactive API docs (FastAPI
generates this automatically — try the upload endpoint right from there).

**2. Frontend**

In a separate terminal:

```bash
cd frontend
python3 -m http.server 8080
```

Visit `http://localhost:8080` and drag a document in. It should appear in
the case-file panel with its ID, size, and timestamp, and the backend
terminal will log the request.

## What's next (Phase 2)

We'll add a `services/forgery_detection.py` module implementing:
- **Error Level Analysis (ELA)** — resave the image at a known JPEG quality
  and diff it against the original; edited regions light up.
- **EXIF metadata inspection** — check if Photoshop/GIMP touched the file.

The upload endpoint will call this module and return a fraud score alongside
the existing metadata.
