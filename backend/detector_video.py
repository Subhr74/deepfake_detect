"""
detector_video.py
-----------------
Deepfake detection for video files using multi-signal fusion per frame.
"""

import cv2
import time
import numpy as np
from utils import detect_faces, compute_fake_probability, safe_int, safe_float, _FAKE_THRESHOLD

FRAME_STEP       = 10
MAX_FRAMES       = 60
MAX_WALL_SECONDS = 10.0
RESIZE_WIDTH     = 480


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > RESIZE_WIDTH:
        scale  = RESIZE_WIDTH / w
        frame  = cv2.resize(frame, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
    return frame


def _best_face_crop(frame: np.ndarray):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.10 * min(w, h))
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(frame.shape[1], x + w + pad)
    y2  = min(frame.shape[0], y + h + pad)
    crop = frame[y1:y2, x1:x2]
    return (crop if crop.size > 0 else None), gray


def detect_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = safe_int(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    fake_count   = 0
    real_count   = 0
    fake_probs   = []
    frame_idx    = 0
    frames_done  = 0
    start_time   = time.time()

    while True:
        if frames_done >= MAX_FRAMES:
            break
        if time.time() - start_time > MAX_WALL_SECONDS:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % FRAME_STEP != 0:
            continue

        frame = _resize_frame(frame)
        crop, gray = _best_face_crop(frame)

        # Fall back to whole frame if no face
        if crop is None:
            crop = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            result = compute_fake_probability(crop, gray)
        except Exception:
            continue

        fake_probs.append(result["fake_probability"])
        if result["label"] == "Fake":
            fake_count += 1
        else:
            real_count += 1
        frames_done += 1

    cap.release()
    elapsed = time.time() - start_time

    if frames_done == 0:
        return {
            "total_frames":   total_frames,
            "frames_sampled": 0,
            "fake_count":     0,
            "real_count":     0,
            "fake_ratio":     0.0,
            "avg_fake_prob":  0.0,
            "label":          "Unknown",
            "confidence":     0.0,
            "message":        "No frames could be analysed.",
        }

    avg_fake_prob = float(np.mean(fake_probs))
    fake_ratio    = safe_float(fake_count / frames_done)
    label         = "Fake" if avg_fake_prob >= _FAKE_THRESHOLD else "Real"
    confidence    = avg_fake_prob if label == "Fake" else 1.0 - avg_fake_prob

    return {
        "total_frames":   total_frames,
        "frames_sampled": safe_int(frames_done),
        "fake_count":     safe_int(fake_count),
        "real_count":     safe_int(real_count),
        "fake_ratio":     round(fake_ratio, 4),
        "avg_fake_prob":  round(safe_float(avg_fake_prob), 4),
        "label":          label,
        "confidence":     round(safe_float(confidence), 4),
        "message": (
            f"Sampled {frames_done} frames every {FRAME_STEP}th frame "
            f"(max {MAX_FRAMES}). Wall time: {elapsed:.1f}s. "
            f"Average fake probability: {avg_fake_prob:.2%}."
        ),
    }