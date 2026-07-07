"""
Phase 2: Digital Forgery & Manipulation Detection.

Two independent, rule-based signals — no trained model needed yet:

1. Error Level Analysis (ELA): resaves the image at a known JPEG quality and
   diffs it against the original. Edited regions compress differently than
   the rest of the image and show up brighter in the diff.

2. EXIF metadata inspection: checks whether editing software (Photoshop,
   GIMP, Canva, etc.) touched the file, and whether basic capture metadata
   is present at all (scanned documents from real scanners/phones usually
   carry EXIF; a "flat" screenshot-like file with none can itself be a
   mild signal, though it's common enough not to weight heavily alone).

Both signals are combined into a single 0-100 fraud_score. This is a
heuristic, explainable baseline — Phase 4 will add a trained CNN that
learns patterns humans wouldn't think to hand-code.
"""
import io
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageChops, ExifTags

# Software strings that indicate the file was edited after capture/scan.
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "canva", "illustrator", "affinity photo",
    "paint.net", "pixlr", "lightroom", "snapseed",
]

ELA_JPEG_QUALITY = 90
# Block size (pixels) used to look for *localized* hotspots, rather than a
# single global max — a global max is easily swamped by whole-image noise
# and misses the real signal, which is one small region behaving
# differently than the rest of the document.
ELA_BLOCK_SIZE = 16
# A block is a "hotspot" if its mean error is this many standard deviations
# above the image's own background mean error.
ELA_HOTSPOT_ZSCORE = 4.0


@dataclass
class ForgeryReport:
    fraud_score: int                  # 0 (clean) - 100 (highly suspicious)
    ela_hotspot_zscore: float         # how much the worst block stands out vs. background
    ela_suspicious: bool
    exif_software: str | None
    exif_suspicious: bool
    exif_present: bool
    reasons: list[str] = field(default_factory=list)
    ela_image_path: str | None = None  # path to the saved ELA visualization


def _run_ela(image_path: str, output_dir: str, doc_id: str) -> tuple[int, bool, str]:
    """
    Resaves the image at ELA_JPEG_QUALITY and diffs it against the original,
    then looks for a *localized* hotspot block whose error is a statistical
    outlier compared to the rest of the document.

    Why block-based instead of a single global max: a whole-image max is
    easily dominated by ordinary scan noise or texture (real documents
    aren't perfectly flat), which drowns out the actual signal — one small
    region that was edited and re-compressed separately from the rest.
    Splitting into blocks and comparing each block's error against the
    image's own background lets a small tampered patch stand out even when
    the whole image has plenty of natural noise.

    Returns (hotspot_zscore_rounded, is_suspicious, path_to_ela_visualization).
    """
    original = Image.open(image_path).convert("RGB")

    resaved_buffer = io.BytesIO()
    original.save(resaved_buffer, "JPEG", quality=ELA_JPEG_QUALITY)
    resaved_buffer.seek(0)
    resaved = Image.open(resaved_buffer)

    diff = ImageChops.difference(original, resaved)
    diff_arr = np.asarray(diff).astype(np.float32).mean(axis=2)  # grayscale error map

    h, w = diff_arr.shape
    bs = ELA_BLOCK_SIZE
    n_rows, n_cols = h // bs, w // bs

    if n_rows == 0 or n_cols == 0:
        block_means = np.array([diff_arr.mean()])
    else:
        trimmed = diff_arr[: n_rows * bs, : n_cols * bs]
        blocks = trimmed.reshape(n_rows, bs, n_cols, bs).transpose(0, 2, 1, 3)
        block_means = blocks.reshape(n_rows * n_cols, bs, bs).mean(axis=(1, 2))

    bg_mean = float(np.median(block_means))
    bg_std = float(np.std(block_means)) or 1e-6
    hotspot_z = float((block_means.max() - bg_mean) / bg_std)

    extrema = diff.getextrema()
    max_diff = max(channel_max for _, channel_max in extrema)
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela_visual = diff.point(lambda p: min(255, int(p * scale)))

    ela_filename = f"{doc_id}_ela.jpg"
    ela_path = os.path.join(output_dir, ela_filename)
    ela_visual.save(ela_path, "JPEG")

    is_suspicious = hotspot_z > ELA_HOTSPOT_ZSCORE
    return round(hotspot_z, 1), is_suspicious, ela_filename


def _check_exif(image_path: str) -> tuple[str | None, bool, bool]:
    """
    Reads EXIF data and looks for editing-software signatures.
    Returns (software_string_or_None, is_suspicious, exif_present).
    """
    image = Image.open(image_path)
    raw_exif = image.getexif()

    if not raw_exif:
        return None, False, False

    tags = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()}
    software = tags.get("Software")

    is_suspicious = False
    if software and any(kw in str(software).lower() for kw in SUSPICIOUS_SOFTWARE_KEYWORDS):
        is_suspicious = True

    return software, is_suspicious, True


def analyze_document(image_path: str, output_dir: str, doc_id: str) -> ForgeryReport:
    """
    Runs all Phase 2 checks on a document and returns a combined report.
    Only handles image files for now (JPG/PNG) — PDF page rasterization
    comes with the OCR step in Phase 3.
    """
    reasons = []

    hotspot_z, ela_suspicious, ela_filename = _run_ela(image_path, output_dir, doc_id)
    if ela_suspicious:
        reasons.append(
            f"ELA found a region compressing differently than the rest of the "
            f"document ({hotspot_z} std. dev. above background). NOTE: plain "
            f"text and hard edges also trigger this on their own, even with "
            f"no tampering — treat this as 'worth a human look', not proof. "
            f"See the ela_image for a visual overlay."
        )

    software, exif_suspicious, exif_present = _check_exif(image_path)
    if exif_suspicious:
        reasons.append(f"File metadata shows it was last saved by '{software}'.")
    elif not exif_present:
        reasons.append(
            "No EXIF metadata found at all — expected for a plain scan or "
            "screenshot, but also consistent with metadata being stripped."
        )

    # Weighted combination. EXIF software evidence is direct and reliable,
    # so it carries most of the weight. ELA is kept deliberately low-weight:
    # our own testing showed it fires on ordinary text/edges just as often
    # as on real tampering (see README "Known limitations"), so on its own
    # it isn't trustworthy enough to drive an automatic decision — it's
    # included mainly so the visual overlay reaches the human reviewer.
    # Phase 4's trained CNN is what actually learns to tell a normal text
    # edge apart from a spliced one; until then, treat ELA as advisory.
    score = 0
    if ela_suspicious:
        score += 15
    if exif_suspicious:
        score += 70
    if not exif_present:
        score += 5
    score = min(score, 100)

    return ForgeryReport(
        fraud_score=score,
        ela_hotspot_zscore=hotspot_z,
        ela_suspicious=ela_suspicious,
        exif_software=software,
        exif_suspicious=exif_suspicious,
        exif_present=exif_present,
        reasons=reasons,
        ela_image_path=ela_filename,
    )
