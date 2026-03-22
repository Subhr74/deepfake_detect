import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms

# ─────────────────────────────────────────────
# Model singleton – loaded once at import time
# ─────────────────────────────────────────────

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing pipeline expected by MobileNetV2
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# OpenCV Haar Cascade path (bundled with opencv-python)
_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_cascade_path)


def get_model() -> nn.Module:
    """Return the shared MobileNetV2 classifier (loaded on first call)."""
    global _model
    if _model is None:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        base = models.mobilenet_v2(weights=weights)
        # Replace classifier head: 2 output classes (Real=0, Fake=1)
        base.classifier[1] = nn.Linear(base.last_channel, 2)
        base.eval()
        base.to(_device)
        _model = base
    return _model


def preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR face crop to a normalised tensor on the correct device."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(face_rgb).unsqueeze(0).to(_device)
    return tensor


def detect_faces(gray: np.ndarray):
    """
    Run Haar-Cascade face detection on a grayscale image.
    Returns list of (x, y, w, h) tuples.
    """
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def run_inference(tensor: torch.Tensor):
    """
    Run a single forward pass; return (label_str, confidence_float).
    label: 'Fake' if class-1 probability > 0.5 else 'Real'
    """
    model = get_model()
    with torch.no_grad():
        logits = model(tensor)           # shape (1, 2)
        probs = torch.softmax(logits, dim=1)
        fake_prob = float(probs[0][1].item())

    label = "Fake" if fake_prob > 0.5 else "Real"
    confidence = round(fake_prob if label == "Fake" else 1.0 - fake_prob, 4)
    return label, confidence


def safe_int(val) -> int:
    """Coerce numpy/tensor int to Python int."""
    return int(val)


def safe_float(val) -> float:
    """Coerce numpy/tensor float to Python float."""
    return float(val)