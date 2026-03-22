"""
detector_video.py
-----------------
Deepfake detection for video files.
Samples every Nth frame, analyses the best face per sampled frame,
and returns aggregate statistics.
"""

import cv2
import time
import numpy as np
from utils import detect_faces, preprocess_face, run_inference, safe_int, safe_float

# ── Tuning knobs ───────────────────────────────────────────────────────────
FRAME_STEP        = 10      # analyse every 10th frame
MAX_FRAMES        = 60      # hard cap on frames analysed
MAX_WALL_SECONDS  = 10.0    # bail out after this many wall-clock seconds
RESIZE_WIDTH      = 480     # rescale each frame to this width for speed
# ──────────────────────────────────────────────────────────────────────────


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > RESIZE_WIDTH:
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame


def _best_face_crop(frame: np.ndarray):
    """
    Detect faces in *frame* and return the crop for the largest one.
    Returns None if no face found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if not faces:
        return None

    # Pick the largest face by area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.1 * min(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def detect_video(video_path: str) -> dict:
    """
    Analyse a video file for deepfakes by sampling frames.

    Returns
    -------
    {
      "total_frames":   int,   # total frames in the video
      "frames_sampled": int,   # frames actually passed to the model
      "fake_count":     int,
      "real_count":     int,
      "fake_ratio":     float, # fake_count / frames_sampled
      "label":          str,   # "Fake" | "Real"
      "confidence":     float,
      "message":        str
    }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = safe_int(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fake_count  = 0
    real_count  = 0
    frame_idx   = 0
    frames_done = 0
    start_time  = time.time()

    while True:
        # ── Time / count guard ──────────────────────────────────────────
        if frames_done >= MAX_FRAMES:
            break
        if time.time() - start_time > MAX_WALL_SECONDS:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Only process every FRAME_STEP-th frame
        if frame_idx % FRAME_STEP != 0:
            continue

        frame = _resize_frame(frame)
        crop  = _best_face_crop(frame)

        # Fall back to whole frame if no face detected
        if crop is None:
            crop = frame

        try:
            tensor = preprocess_face(crop)
            label, _ = run_inference(tensor)
        except Exception:
            # Skip corrupt frames silently
            continue

        if label == "Fake":
            fake_count += 1
        else:
            real_count += 1

        frames_done += 1

    cap.release()

    if frames_done == 0:
        return {
            "total_frames":   total_frames,
            "frames_sampled": 0,
            "fake_count":     0,
            "real_count":     0,
            "fake_ratio":     0.0,
            "label":          "Unknown",
            "confidence":     0.0,
            "message":        "No frames could be analysed.",
        }

    fake_ratio  = safe_float(fake_count / frames_done)
    label       = "Fake" if fake_ratio >= 0.5 else "Real"
    confidence  = safe_float(fake_ratio if label == "Fake" else 1.0 - fake_ratio)

    return {
        "total_frames":   total_frames,
        "frames_sampled": safe_int(frames_done),
        "fake_count":     safe_int(fake_count),
        "real_count":     safe_int(real_count),
        "fake_ratio":     round(fake_ratio, 4),
        "label":          label,
        "confidence":     round(confidence, 4),
        "message":        (
            f"Processed {frames_done} frames "
            f"(every {FRAME_STEP}th, capped at {MAX_FRAMES}). "
            f"Wall time: {time.time() - start_time:.1f}s"
        ),
    }