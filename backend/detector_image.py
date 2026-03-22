"""
detector_image.py
-----------------
Deepfake / AI-image detection for still images.
"""

import cv2
import numpy as np
from utils import detect_faces, compute_fake_probability, safe_int, safe_float

MAX_DIM   = 720
MAX_FACES = 2


def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        s   = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def detect_image(image_path: str) -> dict:
    """
    Analyse a single image.

    Returns JSON-serialisable dict:
    {
      faces_detected, faces_analyzed,
      results: [{face_index, label, confidence, bounding_box, signals}],
      overall_label, overall_confidence, message
    }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img      = _resize(img)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces    = detect_faces(gray)

    # ── No face detected → analyse whole image ────────────────────────────
    if not faces:
        result = compute_fake_probability(img, full_gray=gray)
        return {
            "faces_detected":     0,
            "faces_analyzed":     1,
            "results": [{
                "face_index":   0,
                "label":        result["label"],
                "confidence":   safe_float(result["confidence"]),
                "bounding_box": [0, 0, safe_int(img.shape[1]), safe_int(img.shape[0])],
                "signals":      {k: safe_float(v) for k, v in result["signals"].items()},
            }],
            "overall_label":      result["label"],
            "overall_confidence": safe_float(result["confidence"]),
            "message": "No face detected — full-image analysis used.",
        }

    # ── Analyse each detected face ────────────────────────────────────────
    results    = []
    fake_scores = []
    real_scores = []

    for i, (x, y, w, h) in enumerate(faces[:MAX_FACES]):
        pad = int(0.12 * min(w, h))
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(img.shape[1], x + w + pad)
        y2  = min(img.shape[0], y + h + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        res = compute_fake_probability(crop, full_gray=gray)

        results.append({
            "face_index":   i,
            "label":        res["label"],
            "confidence":   safe_float(res["confidence"]),
            "bounding_box": [safe_int(x), safe_int(y), safe_int(w), safe_int(h)],
            "signals":      {k: safe_float(v) for k, v in res["signals"].items()},
        })

        if res["label"] == "Fake":
            fake_scores.append(res["fake_probability"])
        else:
            real_scores.append(res["fake_probability"])

    if not results:
        raise RuntimeError("All face crops were empty after padding.")

    # Overall verdict: highest fake_probability wins
    all_probs = [r["signals"] for r in results]
    # Recompute from stored fake_probability via confidence + label
    fake_probs_raw = []
    for r in results:
        if r["label"] == "Fake":
            fake_probs_raw.append(r["confidence"])
        else:
            fake_probs_raw.append(1.0 - r["confidence"])

    mean_fake_prob = float(np.mean(fake_probs_raw))
    from utils import _THRESHOLD
    overall_label = "Fake" if mean_fake_prob >= _THRESHOLD else "Real"
    overall_conf  = safe_float(
        mean_fake_prob if overall_label == "Fake" else 1.0 - mean_fake_prob
    )

    return {
        "faces_detected":     safe_int(len(faces)),
        "faces_analyzed":     safe_int(len(results)),
        "results":            results,
        "overall_label":      overall_label,
        "overall_confidence": overall_conf,
        "message": f"Analysed {len(results)} of {len(faces)} detected face(s).",
    }