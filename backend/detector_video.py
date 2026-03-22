"""
detector_video.py
-----------------
Deepfake detection for video files — frame sampling + per-frame analysis.
"""

import cv2
import time
import numpy as np
from utils import detect_faces, compute_fake_probability, safe_int, safe_float

FRAME_STEP       = 10
MAX_FRAMES       = 60
MAX_WALL_SECONDS = 12.0
RESIZE_WIDTH     = 480


def _resize_frame(frame):
    h, w = frame.shape[:2]
    if w > RESIZE_WIDTH:
        s = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return frame


def _best_face_crop(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        return None, gray
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.12 * min(w, h))
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

    fake_count  = 0
    real_count  = 0
    frame_idx   = 0
    frames_done = 0
    start       = time.time()

    while True:
        if frames_done >= MAX_FRAMES or time.time() - start > MAX_WALL_SECONDS:
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_STEP != 0:
            continue

        frame = _resize_frame(frame)
        crop, gray = _best_face_crop(frame)
        src = crop if crop is not None else frame

        try:
            res = compute_fake_probability(src, full_gray=gray)
            if res["label"] == "Fake":
                fake_count += 1
            else:
                real_count += 1
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
            "message": "No frames could be analysed.",
        }

    fake_ratio = safe_float(fake_count / frames_done)
    label      = "Fake" if fake_ratio >= 0.50 else "Real"
    confidence = safe_float(fake_ratio if label == "Fake" else 1.0 - fake_ratio)

    return {
        "total_frames":   total_frames,
        "frames_sampled": safe_int(frames_done),
        "fake_count":     safe_int(fake_count),
        "real_count":     safe_int(real_count),
        "fake_ratio":     round(fake_ratio, 4),
        "label":          label,
        "confidence":     round(confidence, 4),
        "message": (
            f"Processed {frames_done} frames "
            f"(every {FRAME_STEP}th, cap {MAX_FRAMES}). "
            f"Wall time: {elapsed:.1f}s"
        ),
    }