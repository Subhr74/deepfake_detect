"""
detector_image.py
-----------------
Deepfake detection for still images.
Returns a JSON-serialisable dict with face-level results.
"""

import cv2
import numpy as np
from utils import detect_faces, preprocess_face, run_inference, safe_int, safe_float

# Maximum long-side dimension before analysis (keeps inference fast)
MAX_DIM = 640
# Maximum number of faces to analyse per image
MAX_FACES = 2


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def detect_image(image_path: str) -> dict:
    """
    Analyse an image file for deepfake faces.

    Returns
    -------
    {
      "faces_detected": int,
      "faces_analyzed": int,
      "results": [
          {
            "face_index": int,
            "label": "Fake" | "Real",
            "confidence": float,          # confidence in the returned label
            "bounding_box": [x, y, w, h]
          },
          ...
      ],
      "overall_label": "Fake" | "Real",
      "overall_confidence": float,
      "message": str
    }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = _resize_if_needed(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    # ── No face found: run whole-image inference ──────────────────────────
    if not faces:
        tensor = preprocess_face(img)
        label, confidence = run_inference(tensor)
        return {
            "faces_detected": 0,
            "faces_analyzed": 1,
            "results": [
                {
                    "face_index": 0,
                    "label": label,
                    "confidence": safe_float(confidence),
                    "bounding_box": [0, 0, safe_int(img.shape[1]), safe_int(img.shape[0])],
                }
            ],
            "overall_label": label,
            "overall_confidence": safe_float(confidence),
            "message": "No face detected; whole-image analysis used.",
        }

    # ── Analyse up to MAX_FACES faces ─────────────────────────────────────
    results = []
    fake_confs = []
    real_confs = []

    for i, (x, y, w, h) in enumerate(faces[:MAX_FACES]):
        # Slight padding around the crop
        pad = int(0.1 * min(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        tensor = preprocess_face(face_crop)
        label, confidence = run_inference(tensor)

        results.append({
            "face_index": i,
            "label": label,
            "confidence": safe_float(confidence),
            "bounding_box": [safe_int(x), safe_int(y), safe_int(w), safe_int(h)],
        })

        if label == "Fake":
            fake_confs.append(confidence)
        else:
            real_confs.append(confidence)

    if not results:
        raise RuntimeError("Face crops were empty after padding – cannot analyse.")

    # Overall verdict: majority vote weighted by confidence
    fake_score = sum(fake_confs)
    real_score = sum(real_confs)
    if fake_score >= real_score:
        overall_label = "Fake"
        overall_confidence = safe_float(fake_score / len(results))
    else:
        overall_label = "Real"
        overall_confidence = safe_float(real_score / len(results))

    return {
        "faces_detected": safe_int(len(faces)),
        "faces_analyzed": safe_int(len(results)),
        "results": results,
        "overall_label": overall_label,
        "overall_confidence": overall_confidence,
        "message": f"Analysed {len(results)} of {len(faces)} detected face(s).",
    }