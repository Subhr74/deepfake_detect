import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load model (replace with your trained model later)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def detect_fake(filepath):
    try:
        img = cv2.imread(filepath)

        if img is None:
            return {"label": "Invalid Image", "confidence": 0}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return {"label": "No face detected", "confidence": 0}

        results = []

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            input_tensor = transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)

            confidence = torch.softmax(output, dim=1).max().item()

            label = "Fake" if confidence < 0.7 else "Real"

            results.append({
                "label": label,
                "confidence": round(confidence * 100, 2),
                "box": [int(x), int(y), int(w), int(h)]
            })

        return {"faces": results}

    except Exception as e:
        return {"error": str(e)}