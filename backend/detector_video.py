import cv2
from detector_image import analyze_face
from utils import extract_faces

def detect_video(filepath):
    cap = cv2.VideoCapture(filepath)

    frame_count = 0
    fake_count = 0
    real_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Analyze every 10th frame (performance)
        if frame_count % 10 == 0:
            faces = extract_faces(frame)

            for face, _ in faces:
                label, _ = analyze_face(face)

                if label == "Fake":
                    fake_count += 1
                else:
                    real_count += 1

        frame_count += 1

    cap.release()

    total = fake_count + real_count

    if total == 0:
        return {"result": "No faces detected"}

    fake_ratio = fake_count / total

    label = "Fake Video" if fake_ratio > 0.5 else "Real Video"

    return {
        "label": label,
        "fake_ratio": round(fake_ratio * 100, 2)
    }