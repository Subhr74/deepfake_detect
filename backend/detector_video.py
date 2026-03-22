# detector_video.py — Video Deepfake Detector
# Public API: detect_video(video_path: str) -> dict

import cv2
import io
import time
import numpy as np
from PIL import Image as PILImage

from utils import (
    detect_faces,
    compute_fake_probability,
    safe_int,
    safe_float,
    _THRESHOLD,
)

FRAME_STEP   = 8
MAX_FRAMES   = 80
MAX_WALL_SEC = 20.0
RESIZE_W     = 512


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > RESIZE_W:
        s     = RESIZE_W / w
        frame = cv2.resize(frame, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_AREA)
    return frame


def _get_best_face(frame: np.ndarray):
    """Return (largest_face_crop, full_gray, bbox) or (None, gray, None)."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        return None, gray.astype(np.float32), None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad  = int(0.15 * min(w, h))
    x1   = max(0, x - pad);        y1 = max(0, y - pad)
    x2   = min(frame.shape[1], x + w + pad)
    y2   = min(frame.shape[0], y + h + pad)
    crop = frame[y1:y2, x1:x2]
    return (crop if crop.size > 0 else None), gray.astype(np.float32), (x, y, w, h)


def _ela_score_fast(face_bgr: np.ndarray) -> float:
    """Quick JPEG Ghost ELA score for a face crop."""
    try:
        pil  = PILImage.fromarray(
            cv2.cvtColor(cv2.resize(face_bgr, (128, 128)), cv2.COLOR_BGR2RGB))
        orig = np.array(pil).astype(np.float32)
        diffs = []
        for q in [65, 75, 85]:
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            rc = np.array(PILImage.open(buf)).astype(np.float32)
            diffs.append(float(np.abs(orig - rc).mean()))
        md = float(np.mean(diffs))
        return float(np.clip(1.0 / (1.0 + np.exp((md - 10) * 0.28)), 0, 1))
    except Exception:
        return 0.45


def _optical_flow_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Optical flow variance between consecutive face crops.
    Deepfake swap: abrupt texture → high flow variance.
    AI static gen: no natural jitter → very low flow variance.
    """
    try:
        p    = cv2.resize(prev_gray, (64, 64)).astype(np.uint8)
        c    = cv2.resize(curr_gray, (64, 64)).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        fv = float(mag.var())
        if fv > 10:
            return float(np.clip((fv - 10) / 20, 0, 0.9))
        elif fv < 0.12:
            return float(np.clip((0.12 - fv) / 0.12, 0, 0.7))
        return 0.10
    except Exception:
        return 0.35


def _texture_consistency(crops: list) -> float:
    """
    Brightness std across face crops — deepfake swaps cause jumps.
    Real video: brightness_std < 8.   Deepfake: > 15.
    """
    if len(crops) < 3:
        return 0.35
    means = []
    for c in crops:
        if c is not None and c.size > 0:
            means.append(float(
                cv2.cvtColor(cv2.resize(c, (64, 64)),
                             cv2.COLOR_BGR2GRAY).mean()))
    if len(means) < 3:
        return 0.35
    bs = float(np.std(means))
    return float(np.clip(1.0 / (1.0 + np.exp(-(bs - 14) * 0.18)), 0, 1))


def _noise_corr(crops: list) -> float:
    """
    Temporal noise correlation — same camera PRNU across frames = real.
    Different noise (swap) = low correlation.
    """
    if len(crops) < 2:
        return 0.35
    residuals = []
    for c in crops[:8]:
        if c is None or c.size == 0:
            continue
        g = cv2.cvtColor(cv2.resize(c, (64, 64)),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        blurred = cv2.GaussianBlur(g, (5, 5), 1.0)
        residuals.append((g - blurred).flatten())
    if len(residuals) < 2:
        return 0.35
    corrs = []
    for i in range(len(residuals) - 1):
        r1, r2 = residuals[i], residuals[i + 1]
        if r1.std() < 0.1 or r2.std() < 0.1:
            corrs.append(0.5)
            continue
        corrs.append(float(np.corrcoef(r1, r2)[0, 1]))
    mc = float(np.mean(corrs))
    return float(np.clip(1.0 / (1.0 + np.exp((mc - 0.40) * 8)), 0, 1))


# ── Public entry point ────────────────────────────────────────────────────────

def detect_video(video_path: str) -> dict:
    """
    Analyse a video file for deepfake content.

    Returns a JSON-serialisable dict containing:
      label, confidence, fake_count, real_count, fake_ratio,
      frames_sampled, total_frames, temporal_fake_score, message
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames  = safe_int(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_labels  = []
    crops_list    = []
    prev_gray     = None
    flow_scores   = []

    frame_idx   = 0
    frames_done = 0
    start       = time.time()

    while True:
        if frames_done >= MAX_FRAMES or time.time() - start > MAX_WALL_SEC:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_STEP != 0:
            continue

        frame = _resize_frame(frame)
        crop, full_gray, _ = _get_best_face(frame)
        src = crop if crop is not None else frame

        # Optical flow between consecutive face regions
        curr_gray_small = cv2.cvtColor(
            cv2.resize(src, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            flow_scores.append(_optical_flow_score(prev_gray, curr_gray_small))
        prev_gray = curr_gray_small

        crops_list.append(crop)

        # Per-frame forensics
        try:
            res   = compute_fake_probability(src, full_gray=full_gray)
            ela   = _ela_score_fast(src)
            fp    = float(np.clip(0.65 * res["fake_probability"] + 0.35 * ela, 0, 1))
            frame_labels.append("Fake" if fp >= _THRESHOLD else "Real")
            frames_done += 1
        except Exception:
            continue

    cap.release()
    elapsed = time.time() - start

    if frames_done == 0:
        return {
            "total_frames": total_frames, "frames_sampled": 0,
            "fake_count": 0, "real_count": 0, "fake_ratio": 0.0,
            "label": "Unknown", "confidence": 0.0,
            "temporal_fake_score": 0.0,
            "message": "No frames could be analysed.",
        }

    # Temporal signals
    flow_score    = float(np.mean(flow_scores)) if flow_scores else 0.35
    tex_score     = _texture_consistency(crops_list)
    noise_score   = _noise_corr(crops_list)
    temporal_score = float(np.clip(
        0.40 * flow_score + 0.35 * tex_score + 0.25 * noise_score, 0, 1))

    # Frame vote ratio
    fake_count = sum(1 for l in frame_labels if l == "Fake")
    real_count = frames_done - fake_count
    fake_ratio = fake_count / frames_done

    # Final ensemble
    final_score = float(np.clip(
        0.70 * fake_ratio + 0.30 * temporal_score, 0, 1))
    label = "Fake" if final_score >= _THRESHOLD else "Real"
    conf  = final_score if label == "Fake" else 1.0 - final_score

    return {
        "total_frames":        total_frames,
        "frames_sampled":      safe_int(frames_done),
        "fake_count":          safe_int(fake_count),
        "real_count":          safe_int(real_count),
        "fake_ratio":          round(float(fake_ratio), 4),
        "label":               label,
        "confidence":          round(float(conf), 4),
        "temporal_fake_score": round(float(temporal_score), 4),
        "message": (
            f"Analysed {frames_done} frames (every {FRAME_STEP}th). "
            f"Frame verdict: {fake_count} fake / {real_count} real. "
            f"Temporal score: {temporal_score:.2f}. "
            f"Wall time: {elapsed:.1f}s."
        ),
    }