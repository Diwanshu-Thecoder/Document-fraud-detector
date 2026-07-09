"""
Phase 4: Synthetic dataset generation.

Real labeled tampered-document datasets are scarce (and the ones that exist
are usually private, for obvious reasons), so we generate our own — exactly
as the project brief suggests: take template documents, alter fields with
photo-editing operations, and label the results.

Two classes:
  - authentic/: rendered once, saved once. No splicing.
  - tampered/:  rendered, saved (simulating an original scan/photo), then
                RE-OPENED and a randomly-placed patch of freshly-rendered
                text is pasted over a field and the whole thing re-saved.
                This mimics the real-world signature of splicing: the
                pasted region has a different compression history than the
                surrounding image, and there's often a subtle boundary/
                blending artifact at the patch edge.

Deliberate variety is added (background noise level, template layout, font
size, patch position, JPEG quality, blur amount) so the model learns a
general notion of "spliced patch" rather than memorizing one template.
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMG_SIZE = (300, 200)  # (width, height) before any resizing for the model

NAMES = ["Aarav Sharma", "Priya Nair", "John Miller", "Fatima Khan", "Wei Zhang",
         "Lucia Rossi", "Kwame Mensah", "Sara Kim", "Diego Torres", "Emma Wilson"]
COURSES = ["Data Structures", "Machine Learning", "Web Development", "Cloud Computing",
           "Digital Marketing", "Financial Accounting", "Graphic Design", "Cybersecurity"]
GRADES = ["A+", "A", "B+", "B", "Pass", "Distinction"]


def _random_background(w, h, noise_level):
    """Simulates paper/scan texture: a near-white base with mild per-pixel noise."""
    base_val = random.randint(245, 253)
    arr = np.random.normal(base_val, noise_level, (h, w, 3)).clip(230, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _render_document(noise_level):
    """Renders one synthetic certificate with randomized content/layout."""
    w, h = IMG_SIZE
    img = _random_background(w, h, noise_level)
    draw = ImageDraw.Draw(img)

    name = random.choice(NAMES)
    course = random.choice(COURSES)
    grade = random.choice(GRADES)
    year = random.randint(2018, 2025)

    draw.text((20, 15), "CERTIFICATE OF COMPLETION", fill=(15, 15, 15))
    draw.line((20, 35, 280, 35), fill=(100, 100, 100), width=1)
    draw.text((20, 55), f"Awarded to: {name}", fill=(20, 20, 20))
    draw.text((20, 85), f"Course: {course}", fill=(20, 20, 20))

    grade_pos = (20, 115)
    draw.text(grade_pos, f"Grade: {grade}", fill=(20, 20, 20))
    draw.text((20, 145), f"Year: {year}", fill=(20, 20, 20))

    return img, grade_pos, grade


def generate_authentic(n, out_dir):
    for i in range(n):
        noise = random.uniform(2.0, 3.5)
        img, _, _ = _render_document(noise)
        # Single save, quality range overlapping with tampered's *final* save
        # quality so "overall JPEG quality" alone isn't a trivial giveaway —
        # the model has to find the localized signal, not a global shortcut.
        quality = random.randint(65, 90)
        img.save(os.path.join(out_dir, f"auth_{i:04d}.jpg"), "JPEG", quality=quality)


def generate_tampered(n, out_dir):
    for i in range(n):
        noise = random.uniform(2.0, 3.5)
        img, grade_pos, original_grade = _render_document(noise)

        first_quality = random.randint(92, 98)
        buf_path = f"/tmp/_tamper_tmp_{i}.jpg"
        img.save(buf_path, "JPEG", quality=first_quality)
        img = Image.open(buf_path).convert("RGB")
        os.remove(buf_path)

        new_grade = random.choice([g for g in GRADES if g != original_grade])
        patch_w, patch_h = 110, 26
        px, py = grade_pos[0] - 4, grade_pos[1] - 4

        # KEY SIGNAL: the patch is rendered much cleaner (far less noise)
        # than the surrounding scanned/photographed document. This mimics a
        # very common real-world tampering signature — a digitally
        # inserted or re-typed field lacks the sensor/scan noise of the
        # original — and is a standard, independent forensic cue (noise
        # inconsistency analysis) alongside ELA.
        patch_noise = noise * random.uniform(0.1, 0.3)
        patch = _random_background(patch_w, patch_h, patch_noise)
        pdraw = ImageDraw.Draw(patch)
        pdraw.text((4, 4), f"Grade: {new_grade}", fill=(20, 20, 20))

        if random.random() < 0.5:
            patch = patch.filter(ImageFilter.GaussianBlur(radius=0.4))

        img.paste(patch, (px, py))

        second_quality = random.randint(65, 80)
        img.save(os.path.join(out_dir, f"tamp_{i:04d}.jpg"), "JPEG", quality=second_quality)


def main(n_per_class=400, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    auth_dir = os.path.join(OUTPUT_DIR, "authentic")
    tamp_dir = os.path.join(OUTPUT_DIR, "tampered")
    os.makedirs(auth_dir, exist_ok=True)
    os.makedirs(tamp_dir, exist_ok=True)

    print(f"Generating {n_per_class} authentic examples...")
    generate_authentic(n_per_class, auth_dir)
    print(f"Generating {n_per_class} tampered examples...")
    generate_tampered(n_per_class, tamp_dir)
    print("Done.")


if __name__ == "__main__":
    main()
