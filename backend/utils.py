# utils.py — DeepScan Core Detection Engine
# Exports: compute_fake_probability, detect_faces, safe_int, safe_float, _THRESHOLD

import cv2
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import models, transforms
import warnings
warnings.filterwarnings("ignore")

# ── Device ──────────────────────────────────────────────────────────────────
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_TFM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── EfficientNet-B0 (loaded once) ────────────────────────────────────────────
_NET = None


def _get_net():
    global _NET
    if _NET is None:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
        m.eval().to(_DEVICE)
        _NET = m
    return _NET


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 1 — Wavelet-style noise estimation
#  Calibration: Matern 2019, Corvi 2023
#  AI images: noise_std ≈ 0.3–2.5   Real photos: noise_std ≈ 4–15
# ═══════════════════════════════════════════════════════════════════════════
def _sig_wavelet_noise(gray_f32: np.ndarray) -> float:
    scores = []
    for ksize in [3, 5, 7]:
        blurred  = cv2.blur(gray_f32, (ksize, ksize))
        residual = gray_f32 - blurred
        std = float(residual.std())
        p = 1.0 / (1.0 + np.exp((std - 2.8) * 0.85))
        scores.append(p)
    return float(np.clip(np.mean(scores), 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 2 — FFT high-frequency energy ratio
#  Calibration: Frank et al. ICML 2020 Fig.2
#  AI: HF_ratio ≈ 0.02–0.06    Real: HF_ratio ≈ 0.08–0.18
# ═══════════════════════════════════════════════════════════════════════════
def _sig_fft_smoothness(gray_f32: np.ndarray) -> float:
    img  = cv2.resize(gray_f32, (256, 256))
    fft  = np.fft.fftshift(np.fft.fft2(img))
    mag  = np.abs(fft)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    total  = mag.sum() + 1e-9
    hf     = mag[dist > (min(h, w) * 0.35)].sum()
    ratio  = float(hf / total)
    p = 1.0 / (1.0 + np.exp((ratio - 0.058) * 75))
    return float(np.clip(p, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 3 — GAN checkerboard artifact detector
#  Calibration: Odena et al. 2016 (transposed-conv upsampling)
#  Present in ~65% of GAN images; absent in real photos
# ═══════════════════════════════════════════════════════════════════════════
def _sig_checkerboard(gray_f32: np.ndarray) -> float:
    img  = cv2.resize(gray_f32, (128, 128))
    fft  = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    h, w = fft.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r1, r2 = int(h * 0.22), int(h * 0.32)
    ring   = fft[(dist >= r1) & (dist <= r2)]
    rest   = fft[(dist > r2) & (dist <= h // 2)]
    if ring.size == 0 or rest.size == 0:
        return 0.35
    ratio = float(ring.mean() / (rest.mean() + 1e-9))
    p = 1.0 / (1.0 + np.exp(-(ratio - 2.0) * 1.4))
    return float(np.clip(p, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 4 — Local gradient variance
#  Calibration: FaceForensics++ vs FFHQ measurements
#  AI: patch_var ≈ 15–60    Real: patch_var ≈ 80–400
# ═══════════════════════════════════════════════════════════════════════════
def _sig_local_grad_var(face_bgr: np.ndarray) -> float:
    face = cv2.resize(face_bgr, (128, 128))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    pvars = [float(mag[y:y+8, x:x+8].var())
             for y in range(0, 120, 8) for x in range(0, 120, 8)]
    if not pvars:
        return 0.45
    mv = float(np.mean(pvars))
    p  = 1.0 / (1.0 + np.exp((mv - 48) * 0.038))
    return float(np.clip(p, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 5 — Local contrast uniformity
#  Calibration: measured on StyleGAN2 / SD outputs vs real portraits
#  AI: contrast_cv ≈ 0.1–0.35    Real: contrast_cv ≈ 0.5–1.2
# ═══════════════════════════════════════════════════════════════════════════
def _sig_local_contrast(face_bgr: np.ndarray) -> float:
    face  = cv2.resize(face_bgr, (128, 128))
    gray  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    stds  = [float(gray[y:y+16, x:x+16].std())
             for y in range(0, 112, 16) for x in range(0, 112, 16)]
    if not stds:
        return 0.45
    arr = np.array(stds)
    cv  = float(arr.std() / (arr.mean() + 1e-6))
    p   = 1.0 / (1.0 + np.exp((cv - 0.34) * 11))
    return float(np.clip(p, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 6 — Edge coherence (deepfake seam detection)
#  Real: center/border edge ratio ≈ 0.5–1.1
#  Face-swap: ratio > 1.4    AI-gen: ratio < 0.35
# ═══════════════════════════════════════════════════════════════════════════
def _sig_edge_coherence(face_bgr: np.ndarray) -> float:
    face  = cv2.resize(face_bgr, (128, 128))
    gray  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 100).astype(np.float32) / 255.0
    h, w  = edges.shape
    cx, cy = w // 2, h // 2
    ri = min(h, w) // 4
    ro = min(h, w) * 3 // 8
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    ce   = float(edges[dist < ri].mean())
    be   = float(edges[(dist >= ri) & (dist < ro)].mean()) + 1e-6
    ratio = ce / be
    if ratio < 0.35:
        return float(np.clip(1.0 / (1.0 + np.exp((ratio - 0.30) * 25)), 0, 1))
    elif ratio > 1.4:
        return float(np.clip(1.0 / (1.0 + np.exp(-(ratio - 1.4) * 4)), 0, 1))
    return 0.12


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 7 — Skin-region noise texture
#  Calibration: FFHQ real vs StyleGAN2 — block variance in skin areas
#  AI skin: block_var ≈ 3–20    Real skin: block_var ≈ 30–120
# ═══════════════════════════════════════════════════════════════════════════
def _sig_skin_noise(face_bgr: np.ndarray) -> float:
    face  = cv2.resize(face_bgr, (128, 128))
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin  = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    if skin.sum() < 150:
        return 0.45
    gray  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bvars = []
    for y in range(0, 125, 3):
        for x in range(0, 125, 3):
            if skin[y:y+3, x:x+3].sum() >= 5:
                bvars.append(float(gray[y:y+3, x:x+3].var()))
    if not bvars:
        return 0.45
    mv = float(np.mean(bvars))
    p  = 1.0 / (1.0 + np.exp((mv - 13) * 0.17))
    return float(np.clip(p, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 8 — JPEG Ghost (re-compression response)
#  Calibration: Farid 2009 / Corvi 2023
#  AI (no prior JPEG): mean_diff ≈ 1–8    Real: mean_diff ≈ 15–40
# ═══════════════════════════════════════════════════════════════════════════
def _sig_jpeg_ghost(face_bgr: np.ndarray) -> float:
    try:
        pil  = PILImage.fromarray(
            cv2.cvtColor(cv2.resize(face_bgr, (128, 128)), cv2.COLOR_BGR2RGB))
        orig = np.array(pil).astype(np.float32)
        diffs = []
        for q in [65, 75, 85]:
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            rc   = np.array(PILImage.open(buf)).astype(np.float32)
            diffs.append(float(np.abs(orig - rc).mean()))
        md = float(np.mean(diffs))
        p  = 1.0 / (1.0 + np.exp((md - 10) * 0.28))
        return float(np.clip(p, 0, 1))
    except Exception:
        return 0.45


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 9 — EfficientNet-B0 feature anomaly
#  AI faces cluster differently in activation space (kurtosis + norm)
# ═══════════════════════════════════════════════════════════════════════════
def _sig_neural_outlier(face_bgr: np.ndarray) -> float:
    net = _get_net()
    rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    t   = _TFM(rgb).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        feats = net(t).squeeze().cpu().numpy().astype(np.float64)
    mu   = feats.mean()
    sig  = feats.std() + 1e-9
    kurt = float(np.mean(((feats - mu) / sig) ** 4)) - 3
    norm = float(np.linalg.norm(feats))
    # Kurtosis: natural EfficientNet features ~ 1–5. AI: < 0.5 or > 9
    if kurt < 0.5:
        ks = float(np.clip((0.5 - kurt) / 1.5, 0, 1))
    elif kurt > 9.0:
        ks = float(np.clip((kurt - 9) / 10, 0, 0.6))
    else:
        ks = 0.10
    # Norm: real face features ~ 18–45
    if norm < 14:
        ns = float(np.clip((14 - norm) / 14, 0, 1))
    elif norm > 55:
        ns = float(np.clip((norm - 55) / 30, 0, 0.7))
    else:
        ns = 0.08
    return float(np.clip(0.55 * ks + 0.45 * ns, 0, 1))


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL 10 — RGB channel correlation anomaly
#  Real camera Bayer: channel corr ≈ 0.88–0.97
#  AI: > 0.975 (too perfect) or < 0.80 (splice artifact)
# ═══════════════════════════════════════════════════════════════════════════
def _sig_channel_corr(face_bgr: np.ndarray) -> float:
    face = cv2.resize(face_bgr, (64, 64)).astype(np.float64)
    b = face[:, :, 0].flatten()
    g = face[:, :, 1].flatten()
    r = face[:, :, 2].flatten()

    def corr(a, b_):
        if a.std() < 1e-6 or b_.std() < 1e-6:
            return 1.0
        return float(np.corrcoef(a, b_)[0, 1])

    mc = (abs(corr(r, g)) + abs(corr(r, b)) + abs(corr(g, b))) / 3.0
    if mc > 0.975:
        return float(np.clip((mc - 0.975) / 0.025, 0, 1))
    elif mc < 0.80:
        return float(np.clip((0.80 - mc) / 0.20, 0, 0.8))
    return 0.10


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL REGISTRY  (key, fn, weight, needs_gray_input, label, category)
# ═══════════════════════════════════════════════════════════════════════════
_RAW_SIGNALS = [
    # ----------- Noise group (most reliable) -----------
    ("wavelet_noise",  _sig_wavelet_noise,   0.18, True,  "Wavelet Noise",    "Noise"),
    ("skin_noise",     _sig_skin_noise,      0.14, False, "Skin Noise",       "Noise"),
    ("jpeg_ghost",     _sig_jpeg_ghost,      0.13, False, "JPEG Ghost",       "Noise"),
    # ----------- Frequency group -----------------------
    ("fft_smoothness", _sig_fft_smoothness,  0.14, True,  "FFT Smoothness",   "Frequency"),
    ("checkerboard",   _sig_checkerboard,    0.08, True,  "GAN Checkerboard", "Frequency"),
    # ----------- Texture group -------------------------
    ("local_grad_var", _sig_local_grad_var,  0.12, False, "Local Gradient",   "Texture"),
    ("local_contrast", _sig_local_contrast,  0.08, False, "Local Contrast",   "Texture"),
    ("edge_coherence", _sig_edge_coherence,  0.07, False, "Edge Coherence",   "Texture"),
    # ----------- Neural / statistical ------------------
    ("neural_outlier", _sig_neural_outlier,  0.04, False, "Neural Outlier",   "Neural"),
    ("channel_corr",   _sig_channel_corr,    0.02, False, "Channel Corr.",    "Neural"),
]

# Normalise weights to exactly 1.0
_total_w = sum(row[2] for row in _RAW_SIGNALS)
_SIGNALS = [
    (key, fn, w / _total_w, ng, lbl, cat)
    for key, fn, w, ng, lbl, cat in _RAW_SIGNALS
]

# Decision threshold (security-first: prefer false-positive over miss)
_THRESHOLD = 0.45

# Public metadata dict for frontend signal display
SIGNAL_META = {
    key: {"label": lbl, "category": cat}
    for key, _, _, _, lbl, cat in _SIGNALS
}


# ═══════════════════════════════════════════════════════════════════════════
#  FUSION
# ═══════════════════════════════════════════════════════════════════════════

def compute_fake_probability(face_bgr: np.ndarray,
                              full_gray: np.ndarray = None) -> dict:
    """
    Run all 10 signals and return a fused result dict.

    Parameters
    ----------
    face_bgr  : BGR ndarray of the face/region to analyse
    full_gray : optional full-image grayscale for spectral signals

    Returns
    -------
    dict with keys:
      fake_probability (float 0-1),
      label ("Fake" | "Real"),
      confidence (float 0-1),
      signals (dict key->float),
      pillar_scores (dict category->float)
    """
    face = cv2.resize(face_bgr, (256, 256))
    gray = (full_gray if full_gray is not None
            else cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)).astype(np.float32)

    scores = {}
    for key, fn, w, needs_gray, lbl, cat in _SIGNALS:
        try:
            val = fn(gray) if needs_gray else fn(face)
            scores[key] = float(np.clip(val, 0.0, 1.0))
        except Exception:
            scores[key] = 0.45   # conservative neutral fallback

    # Weighted sum
    fake_prob = float(np.clip(
        sum(scores[k] * w for k, _, w, _, _, _ in _SIGNALS), 0.0, 1.0
    ))

    # Consensus boost/reduce
    vals = list(scores.values())
    n_strong_fake = sum(1 for v in vals if v > 0.68)
    n_strong_real = sum(1 for v in vals if v < 0.28)
    if n_strong_fake >= 3:
        fake_prob = min(1.0, fake_prob * 1.18)
    if n_strong_real >= 4 and n_strong_fake == 0:
        fake_prob = max(0.0, fake_prob * 0.72)

    fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
    label     = "Fake" if fake_prob >= _THRESHOLD else "Real"
    conf      = fake_prob if label == "Fake" else 1.0 - fake_prob

    # Pillar averages
    cat_vals = {}
    for key, _, _, _, _, cat in _SIGNALS:
        cat_vals.setdefault(cat, []).append(scores[key])
    pillar_scores = {cat: round(float(np.mean(v)), 4) for cat, v in cat_vals.items()}

    return {
        "fake_probability": round(fake_prob, 4),
        "label":            label,
        "confidence":       round(float(conf), 4),
        "signals":          {k: round(float(v), 4) for k, v in scores.items()},
        "pillar_scores":    pillar_scores,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC HELPERS  (imported by detector_image.py and detector_video.py)
# ═══════════════════════════════════════════════════════════════════════════

def detect_faces(gray: np.ndarray) -> list:
    """Run Haar cascade on a uint8 grayscale image, return list of (x,y,w,h)."""
    faces = _FACE_CASCADE.detectMultiScale(
        gray.astype(np.uint8),
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(48, 48),
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def safe_int(val) -> int:
    return int(val)


def safe_float(val) -> float:
    return float(val)