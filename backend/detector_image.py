"""
detector_image.py
-----------------
Deepfake / AI-image detection for still images using multi-signal fusion.
"""

import cv2
import numpy as np
from utils import detect_faces, compute_fake_probability, safe_int, safe_float

MAX_DIM   = 640   # resize long side to this before any processing
MAX_FACES = 2     # analyse at most this many faces per image


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale  = MAX_DIM / max(h, w)
        img    = cv2.resize(img, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
    return img


def detect_image(image_path: str) -> dict:
    """
    Analyse an image for AI/deepfake content.

    Returns JSON-serialisable dict:
    {
      "faces_detected":   int,
      "faces_analyzed":   int,
      "results": [
        {
          "face_index":   int,
          "label":        "Fake" | "Real",
          "confidence":   float,
          "fake_probability": float,
          "signals":      { freq, texture, colour, neural, noise },
          "bounding_box": [x, y, w, h]
        }
      ],
      "overall_label":       "Fake" | "Real",
      "overall_confidence":  float,
      "overall_fake_prob":   float,
      "message": str
    }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img  = _resize_if_needed(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    # ── No face found → analyse whole image ──────────────────────────────
    if not faces:
        result = compute_fake_probability(img, gray)
        return {
            "faces_detected":      0,
            "faces_analyzed":      1,
            "results": [{
                "face_index":       0,
                "label":            result["label"],
                "confidence":       safe_float(result["confidence"]),
                "fake_probability": safe_float(result["fake_probability"]),
                "signals":          {k: safe_float(v) for k, v in result["signals"].items()},
                "bounding_box":     [0, 0, safe_int(img.shape[1]), safe_int(img.shape[0])],
            }],
            "overall_label":      result["label"],
            "overall_confidence": safe_float(result["confidence"]),
            "overall_fake_prob":  safe_float(result["fake_probability"]),
            "message": "No face detected — whole-image analysis used.",
        }

    # ── Analyse up to MAX_FACES faces ────────────────────────────────────
    results     = []
    fake_probs  = []

    for i, (x, y, w, h) in enumerate(faces[:MAX_FACES]):
        pad = int(0.10 * min(w, h))
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(img.shape[1], x + w + pad)
        y2  = min(img.shape[0], y + h + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        result = compute_fake_probability(crop, gray)
        fake_probs.append(result["fake_probability"])

        results.append({
            "face_index":       i,
            "label":            result["label"],
            "confidence":       safe_float(result["confidence"]),
            "fake_probability": safe_float(result["fake_probability"]),
            "signals":          {k: safe_float(v) for k, v in result["signals"].items()},
            "bounding_box":     [safe_int(x), safe_int(y), safe_int(w), safe_int(h)],
        })

    if not results:
        raise RuntimeError("All face crops were empty — cannot analyse.")

    # Overall: average fake probability across analysed faces
    avg_fake_prob = float(np.mean(fake_probs))
    from utils import _FAKE_THRESHOLD  # re-use same threshold
    overall_label = "Fake" if avg_fake_prob >= _FAKE_THRESHOLD else "Real"
    overall_conf  = avg_fake_prob if overall_label == "Fake" else 1.0 - avg_fake_prob

    return {
        "faces_detected":      safe_int(len(faces)),
        "faces_analyzed":      safe_int(len(results)),
        "results":             results,
        "overall_label":       overall_label,
        "overall_confidence":  round(safe_float(overall_conf), 4),
        "overall_fake_prob":   round(safe_float(avg_fake_prob), 4),
        "message": f"Analysed {len(results)} of {len(faces)} detected face(s). "
                   f"Average fake probability: {avg_fake_prob:.2%}",
    }