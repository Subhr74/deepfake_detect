"""
utils.py
--------
Multi-signal deepfake / AI-image detector for DeepScan.

Detection pipeline (5 fused signals):
  1. Frequency-domain  — FFT + DCT block regularity (GAN/diffusion leave periodic artefacts)
  2. Facial texture    — Laplacian variance, LBP uniformity, edge density
  3. Colour anomaly    — Saturation smoothness + hue uniformity
  4. Neural feature    — MobileNetV2 feature-norm / sparsity / entropy deviation
  5. Noise / PRNU     — Absence of camera sensor noise in AI images

Each signal → [0..1] fake probability → weighted sum → label + confidence.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms

# ── Device ────────────────────────────────────────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pre-processing (MobileNetV2 canonical) ────────────────────────────────
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Haar Cascade ──────────────────────────────────────────────────────────
_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade  = cv2.CascadeClassifier(_cascade_path)

# ── MobileNetV2 feature extractor (no head) ───────────────────────────────
_feature_extractor = None

def _get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        m = models.mobilenet_v2(weights=weights)
        m.classifier = nn.Identity()
        m.eval()
        m.to(_device)
        _feature_extractor = m
    return _feature_extractor


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 1 — Frequency-domain artefacts
# ═══════════════════════════════════════════════════════════════════════════

def _dct_block_regularity(gray_f32: np.ndarray) -> float:
    h, w   = gray_f32.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    if h8 < 8 or w8 < 8:
        return 0.5
    blocks = gray_f32[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8)
    bvars  = blocks.var(axis=(1, 3))
    cv     = float(bvars.std() / (bvars.mean() + 1e-6))
    return float(np.clip(1.0 - cv / 1.2, 0.0, 1.0))


def _fft_mid_high_ratio(gray_f32: np.ndarray) -> float:
    arr    = cv2.resize(gray_f32, (256, 256))
    fft    = np.fft.fft2(arr)
    mag    = np.log1p(np.abs(np.fft.fftshift(fft)))
    h, w   = mag.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    mid  = mag[(dist > 15) & (dist <= 50)].mean()
    high = mag[(dist > 50) & (dist <= 90)].mean() + 1e-6
    ratio = float(mid / high)
    return float(np.clip((ratio - 1.05) / 1.3, 0.0, 1.0))


def _freq_fake_score(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    return float(np.clip(0.55 * _fft_mid_high_ratio(g) + 0.45 * _dct_block_regularity(g), 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 2 — Facial texture & blending
# ═══════════════════════════════════════════════════════════════════════════

def _lbp_uniformity(gray: np.ndarray) -> float:
    h, w  = gray.shape
    count = 0
    total = 0
    step  = 3
    for y in range(1, h - 1, step):
        for x in range(1, w - 1, step):
            centre = gray[y, x]
            ring   = [gray[y-1,x-1], gray[y-1,x], gray[y-1,x+1],
                      gray[y,  x+1], gray[y+1,x+1], gray[y+1,x],
                      gray[y+1,x-1], gray[y,  x-1]]
            bits = sum(1 for v in ring if v >= centre)
            if bits in (0, 1, 7, 8):
                count += 1
            total += 1
    ratio = count / max(total, 1)
    return float(np.clip((ratio - 0.55) / 0.30, 0.0, 1.0))


def _texture_fake_score(face_bgr: np.ndarray) -> float:
    face  = cv2.resize(face_bgr, (128, 128))
    gray  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    if lap_var < 120:
        sharpness_score = float(np.clip((120 - lap_var) / 120.0, 0.0, 1.0))
    elif lap_var > 4000:
        sharpness_score = float(np.clip((lap_var - 4000) / 4000.0, 0.0, 0.7))
    else:
        sharpness_score = 0.1

    lbp_score    = _lbp_uniformity(gray.astype(np.uint8))
    edges        = cv2.Canny(face, 40, 120).astype(np.float32)
    edge_density = float(edges.mean())
    edge_score   = float(np.clip(abs(edge_density - 11.0) / 14.0, 0.0, 0.7))

    return float(np.clip(0.40 * sharpness_score + 0.35 * lbp_score + 0.25 * edge_score, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 3 — Colour / saturation anomalies
# ═══════════════════════════════════════════════════════════════════════════

def _colour_fake_score(face_bgr: np.ndarray) -> float:
    face = cv2.resize(face_bgr, (128, 128))
    hsv  = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat  = hsv[:, :, 1]
    val  = hsv[:, :, 2]
    hue  = hsv[:, :, 0]

    sat_mean = float(sat.mean())
    sat_std  = float(sat.std())
    val_std  = float(val.std())
    hue_std  = float(hue.std())

    sat_score = 0.0
    if sat_mean > 130:
        sat_score = float(np.clip((sat_mean - 130) / 70.0, 0.0, 1.0))
    elif sat_std < 20:
        sat_score = float(np.clip((20 - sat_std) / 20.0, 0.0, 0.85))

    val_score = float(np.clip(1.0 - val_std / 55.0, 0.0, 0.75)) if val_std < 55 else 0.0
    hue_score = float(np.clip((25 - hue_std) / 25.0, 0.0, 0.7)) if hue_std < 25 else 0.0

    return float(np.clip(0.45 * sat_score + 0.30 * val_score + 0.25 * hue_score, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 4 — Neural feature anomaly
# ═══════════════════════════════════════════════════════════════════════════

_REAL_FEAT_NORM_MEAN = 40.0
_REAL_FEAT_NORM_STD  =  9.0

def _neural_fake_score(face_bgr: np.ndarray) -> float:
    fe     = _get_feature_extractor()
    rgb    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0).to(_device)

    with torch.no_grad():
        feats = fe(tensor).squeeze()

    feat_np = feats.cpu().numpy()
    norm    = float(np.linalg.norm(feat_np))
    z_norm  = abs(norm - _REAL_FEAT_NORM_MEAN) / (_REAL_FEAT_NORM_STD + 1e-6)

    sparsity       = float((feat_np == 0).mean())
    sparsity_score = float(np.clip(abs(sparsity - 0.52) / 0.28, 0.0, 1.0))

    feat_abs  = np.abs(feat_np)
    feat_prob = feat_abs / (feat_abs.sum() + 1e-9)
    entropy   = -float(np.sum(feat_prob * np.log(feat_prob + 1e-9)))
    max_ent   = float(np.log(len(feat_np)))
    entropy_score = float(np.clip(1.0 - entropy / max_ent * 1.4, 0.0, 1.0))

    z_score = float(np.clip((z_norm - 0.4) / 3.5, 0.0, 1.0))
    return float(np.clip(0.40 * z_score + 0.35 * sparsity_score + 0.25 * entropy_score, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  Signal 5 — Noise / PRNU fingerprint
# ═══════════════════════════════════════════════════════════════════════════

def _noise_fake_score(face_bgr: np.ndarray) -> float:
    gray      = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    residual  = gray - blurred
    noise_std = float(residual.std())

    if noise_std < 2.0:
        low_noise_score = float(np.clip((2.0 - noise_std) / 2.0, 0.0, 1.0))
    elif noise_std > 14.0:
        low_noise_score = float(np.clip((noise_std - 14.0) / 10.0, 0.0, 0.7))
    else:
        low_noise_score = 0.05

    h, w   = residual.shape
    q_h, q_w = max(1, h // 4), max(1, w // 4)
    quads  = [
        residual[:q_h,   :q_w],   residual[:q_h,   w-q_w:],
        residual[h-q_h:, :q_w],   residual[h-q_h:, w-q_w:],
    ]
    quad_stds = np.array([q.std() for q in quads if q.size > 0])
    if len(quad_stds) > 1:
        uniformity_cv    = float(quad_stds.std() / (quad_stds.mean() + 1e-6))
        uniformity_score = float(np.clip(1.0 - uniformity_cv / 0.5, 0.0, 0.7))
    else:
        uniformity_score = 0.3

    return float(np.clip(0.65 * low_noise_score + 0.35 * uniformity_score, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
#  Fusion
# ═══════════════════════════════════════════════════════════════════════════

_SIGNAL_WEIGHTS = {
    "freq":    0.28,
    "texture": 0.22,
    "colour":  0.18,
    "neural":  0.17,
    "noise":   0.15,
}

# Threshold: fused fake_prob >= this → label "Fake"
_FAKE_THRESHOLD = 0.42


def compute_fake_probability(face_bgr: np.ndarray,
                              full_image_gray: np.ndarray = None) -> dict:
    gray_src = full_image_gray if full_image_gray is not None else \
               cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    scores = {
        "freq":    _freq_fake_score(gray_src),
        "texture": _texture_fake_score(face_bgr),
        "colour":  _colour_fake_score(face_bgr),
        "neural":  _neural_fake_score(face_bgr),
        "noise":   _noise_fake_score(face_bgr),
    }

    fake_prob = float(np.clip(
        sum(_SIGNAL_WEIGHTS[k] * v for k, v in scores.items()), 0.0, 1.0
    ))

    label      = "Fake" if fake_prob >= _FAKE_THRESHOLD else "Real"
    confidence = fake_prob if label == "Fake" else 1.0 - fake_prob

    return {
        "fake_probability": round(fake_prob, 4),
        "label":            label,
        "confidence":       round(float(confidence), 4),
        "signals":          {k: round(v, 4) for k, v in scores.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Public helpers
# ═══════════════════════════════════════════════════════════════════════════

def detect_faces(gray: np.ndarray):
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def safe_int(val)   -> int:   return int(val)
def safe_float(val) -> float: return float(val)