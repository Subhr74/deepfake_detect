import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from utils import extract_faces

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def analyze_face(face):
    img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    confidence = torch.softmax(output, dim=1).max().item()
    label = "Fake" if confidence < 0.7 else "Real"

    return label, round(confidence * 100, 2)


def detect_image(filepath):
    image = cv2.imread(filepath)

    if image is None:
        return {"error": "Invalid image"}

    faces = extract_faces(image)

    results = []

    for face, box in faces:
        label, conf = analyze_face(face)
        results.append({
            "label": label,
            "confidence": conf,
            "box": box
        })

    return {"faces": results}