"""
Phase 4: CNN-based tampering classification (inference).

Loads the model trained in ml/train.py and runs it on uploaded images.

IMPORTANT — read this before trusting the numbers this returns:
This model was trained AND validated entirely on synthetically generated
certificates (see ml/dataset_generation/generate_dataset.py) — a specific,
narrow visual style (one layout, programmatically rendered text, synthetic
paper-noise texture). Its 95% validation accuracy is measured on held-out
data from that SAME synthetic generator, not on real-world scanned/
photographed documents. This is a meaningful, common gap in ML projects:
a model can genuinely learn its training distribution well while still
performing much worse on real-world data with a different visual style
("domain shift"). Treat this model's score as a demonstration of the
technique, not as a production-ready tampering detector — closing that
gap would require real (or far more visually diverse) labeled training
data, which is exactly the kind of dataset the original project brief
notes is scarce and hard to obtain.
"""
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from app.ml_models.model_def import TamperCNN

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "tamper_classifier.pt")
IMG_SIZE = 128
# Must match training normalization exactly (see ml/train.py) — these were
# measured from the synthetic training set, not generic image statistics.
DATA_MEAN, DATA_STD = 0.9554, 0.0616

_model = None
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[DATA_MEAN] * 3, std=[DATA_STD] * 3),
])


def _get_model():
    global _model
    if _model is None:
        model = TamperCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        _model = model
    return _model


def classify_tampering(image_path: str) -> dict:
    """
    Returns {"tampered_probability": float 0-1, "predicted_label": str}.
    Wrapped in a try/except by the caller — this is a demo-quality model
    on synthetic data and shouldn't take down the pipeline if it errors.
    """
    model = _get_model()
    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    tampered_prob = float(probs[1])
    return {
        "tampered_probability": round(tampered_prob, 3),
        "predicted_label": "tampered" if tampered_prob > 0.5 else "authentic",
    }
