# detector_image.py — Image Deepfake Detector
# Public API: detect_image(image_path: str) -> dict

import cv2
import io
import numpy as np
from PIL import Image as PILImage

from utils import (
    detect_faces,
    compute_fake_probability,
    safe_int,
    safe_float,
    _THRESHOLD,
)

MAX_DIM   = 1024
MAX_FACES = 4


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        s   = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_LANCZOS4)
    return img


def _ela(img_bgr: np.ndarray, quality: int = 75):
    """
    Error Level Analysis.
    Returns (ela_mean, ela_std, ela_fake_score).
    Low mean + low std  →  AI-generated (no authentic JPEG history).
    Calibration: Corvi et al. 2023 / Farid 2009.
    """
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    rc  = np.array(PILImage.open(buf).convert("RGB"))
    rc  = cv2.cvtColor(rc, cv2.COLOR_RGB2BGR)

    diff     = cv2.absdiff(img_bgr.astype(np.float32), rc.astype(np.float32))
    ela_gray = np.clip(diff.mean(axis=2) * 12.0, 0, 255).astype(np.uint8)
    ela_mean = float(ela_gray.mean())
    ela_std  = float(ela_gray.std())

    mean_p = 1.0 / (1.0 + np.exp((ela_mean - 6.0) * 0.40))
    std_p  = 1.0 / (1.0 + np.exp((ela_std  - 5.0) * 0.45))
    score  = float(np.clip(0.55 * mean_p + 0.45 * std_p, 0, 1))
    return ela_mean, ela_std, score


def _global_cues(img_bgr: np.ndarray) -> float:
    """
    Whole-image noise (Laplacian std) + AI-typical dimension check.
    Returns fake_score in [0,1].
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap  = cv2.Laplacian(gray, cv2.CV_32F)
    ns   = float(lap.std())
    noise_p = 1.0 / (1.0 + np.exp((ns - 8.0) * 0.25))

    h, w = img_bgr.shape[:2]
    ai_dims = {512, 768, 1024, 1280, 2048}
    size_p  = 0.35 if (h in ai_dims or w in ai_dims) else 0.10

    return float(np.clip(0.70 * noise_p + 0.30 * size_p, 0, 1))


def _grid_analyse(img_bgr: np.ndarray, gray: np.ndarray) -> dict:
    """Analyse 5 overlapping patches; return the most suspicious result."""
    h, w = img_bgr.shape[:2]
    patches = [
        img_bgr[: h // 2 + h // 8, : w // 2 + w // 8],
        img_bgr[: h // 2 + h // 8,   w // 2 - w // 8 :],
        img_bgr[h // 2 - h // 8 :, : w // 2 + w // 8],
        img_bgr[h // 2 - h // 8 :,   w // 2 - w // 8 :],
        img_bgr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4],
    ]
    results = []
    for p in patches:
        if p.size == 0:
            continue
        try:
            results.append(compute_fake_probability(p, full_gray=gray))
        except Exception:
            pass
    if not results:
        return compute_fake_probability(img_bgr, full_gray=gray)
    return max(results, key=lambda r: r["fake_probability"])


# ── Public entry point ────────────────────────────────────────────────────────

def detect_image(image_path: str) -> dict:
    """
    Full forensic analysis of a single image file.

    Returns a JSON-serialisable dict containing:
      overall_label, overall_confidence, ela_*, pillar_scores, results[], message
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img  = _resize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Full-image ELA and global cues
    ela_mean, ela_std, ela_score = _ela(img)
    global_score = _global_cues(img)

    # Face detection
    faces = detect_faces(gray)

    # ── No face found → grid analysis ────────────────────────────────────
    if not faces:
        best = _grid_analyse(img, gray)
        fp   = float(np.clip(
            0.60 * best["fake_probability"] +
            0.28 * ela_score +
            0.12 * global_score, 0, 1))
        lbl  = "Fake" if fp >= _THRESHOLD else "Real"
        conf = fp if lbl == "Fake" else 1.0 - fp

        return {
            "faces_detected":     0,
            "faces_analyzed":     1,
            "overall_label":      lbl,
            "overall_confidence": safe_float(round(conf, 4)),
            "ela_mean":           safe_float(round(ela_mean, 2)),
            "ela_std":            safe_float(round(ela_std, 2)),
            "ela_fake_score":     safe_float(round(ela_score, 4)),
            "pillar_scores":      {k: safe_float(v) for k, v in best["pillar_scores"].items()},
            "results": [{
                "face_index":     0,
                "label":          lbl,
                "confidence":     safe_float(round(conf, 4)),
                "fake_prob":      safe_float(round(fp, 4)),
                "bounding_box":   [0, 0, safe_int(img.shape[1]), safe_int(img.shape[0])],
                "signals":        {k: safe_float(v) for k, v in best["signals"].items()},
                "pillar_scores":  {k: safe_float(v) for k, v in best["pillar_scores"].items()},
                "ela_face_score": safe_float(round(ela_score, 4)),
            }],
            "message": "No face detected — full-image forensic analysis applied.",
        }

    # ── Per-face analysis ─────────────────────────────────────────────────
    face_results = []

    for i, (x, y, w, h) in enumerate(faces[:MAX_FACES]):
        pad  = int(0.18 * min(w, h))
        x1   = max(0, x - pad)
        y1   = max(0, y - pad)
        x2   = min(img.shape[1], x + w + pad)
        y2   = min(img.shape[0], y + h + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Per-face ELA
        ef_mean, ef_std, ef_score = _ela(crop)

        # Forensic analysis
        res = compute_fake_probability(crop, full_gray=gray)

        # Ensemble
        fp   = float(np.clip(
            0.65 * res["fake_probability"] +
            0.25 * ef_score +
            0.10 * global_score, 0, 1))
        lbl  = "Fake" if fp >= _THRESHOLD else "Real"
        conf = fp if lbl == "Fake" else 1.0 - fp

        face_results.append({
            "face_index":     i,
            "label":          lbl,
            "confidence":     safe_float(round(conf, 4)),
            "fake_prob":      safe_float(round(fp, 4)),
            "bounding_box":   [safe_int(x), safe_int(y), safe_int(w), safe_int(h)],
            "signals":        {k: safe_float(v) for k, v in res["signals"].items()},
            "pillar_scores":  {k: safe_float(v) for k, v in res["pillar_scores"].items()},
            "ela_face_score": safe_float(round(ef_score, 4)),
        })

    if not face_results:
        raise RuntimeError("All face crops were empty after padding.")

    # ── Overall verdict ───────────────────────────────────────────────────
    all_fp    = [r["fake_prob"] for r in face_results]
    max_fp    = max(all_fp)
    mean_fp   = float(np.mean(all_fp))
    overall_p = float(np.clip(0.65 * max_fp + 0.35 * mean_fp, 0, 1))
    ovr_lbl   = "Fake" if overall_p >= _THRESHOLD else "Real"
    ovr_conf  = overall_p if ovr_lbl == "Fake" else 1.0 - overall_p

    # Average pillar scores across faces
    all_cats  = set(k for r in face_results for k in r["pillar_scores"])
    avg_pillars = {
        cat: safe_float(round(float(
            np.mean([r["pillar_scores"].get(cat, 0.4) for r in face_results])), 4))
        for cat in all_cats
    }

    n_fake = sum(1 for r in face_results if r["label"] == "Fake")
    n_real = len(face_results) - n_fake

    return {
        "faces_detected":     safe_int(len(faces)),
        "faces_analyzed":     safe_int(len(face_results)),
        "overall_label":      ovr_lbl,
        "overall_confidence": safe_float(round(ovr_conf, 4)),
        "ela_mean":           safe_float(round(ela_mean, 2)),
        "ela_std":            safe_float(round(ela_std, 2)),
        "ela_fake_score":     safe_float(round(ela_score, 4)),
        "pillar_scores":      avg_pillars,
        "results":            face_results,
        "message": (
            f"Analysed {len(face_results)} face(s) — "
            f"{n_fake} fake, {n_real} real. "
            f"ELA score: {ela_score:.2f}. "
            f"Peak fake prob: {max_fp:.3f}."
        ),
    }