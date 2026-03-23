# app.py — DeepScan Flask Backend
# Run: python app.py
# Delete __pycache__ before running after any file change

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

from detector_image import detect_image
from detector_video import detect_video
from detector_voice import detect_voice

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "DeepScan API running"})


# ── Image ────────────────────────────────────────────────────────────────────
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
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    try:
        return jsonify(detect_image(path))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path): os.remove(path)


# ── Video ────────────────────────────────────────────────────────────────────
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
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    try:
        return jsonify(detect_video(path))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path): os.remove(path)


# ── Voice ────────────────────────────────────────────────────────────────────
@app.route("/detect/voice", methods=["POST"])
def detect_voice_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"wav", "mp3", "m4a", "ogg", "flac", "aac"}:
        return jsonify({"error": f"Unsupported audio type: .{ext}"}), 400
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    try:
        return jsonify(detect_voice(path))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path): os.remove(path)


if __name__ == "__main__":
    print("[DeepScan] Starting on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)