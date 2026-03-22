# app.py — DeepScan Flask Backend
# Run: python app.py
# Make sure you deleted any old .pyc files: del /s __pycache__  (Windows)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

# ── These must match the function names in detector_image.py / detector_video.py
from detector_image import detect_image   # noqa: F401
from detector_video import detect_video   # noqa: F401

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "DeepScan API is running"})


@app.route("/detect/image", methods=["POST"])
def detect_image_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"png", "jpg", "jpeg", "bmp", "webp"}:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    try:
        result = detect_image(save_path)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route("/detect/video", methods=["POST"])
def detect_video_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"mp4", "avi", "mov", "mkv", "webm"}:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    try:
        result = detect_video(save_path)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    print("[DeepScan] Starting server on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)