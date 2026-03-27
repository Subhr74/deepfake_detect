# app.py — Deepfake Detection System
# Run:  python app.py
# Then open:  http://localhost:8000  in your browser
# Delete __pycache__ before running after any file change

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback

from detector_image import detect_image
from detector_video import detect_video
from detector_voice import detect_voice

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
# static_folder points Flask at the frontend directory so it can serve HTML/CSS/JS
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


# ── Serve the website ─────────────────────────────────────────────────────────
@app.route("/")
def serve_index():
    """Open http://localhost:8000 → returns the full website"""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serve style.css, script.js, and any other frontend assets"""
    return send_from_directory(FRONTEND_DIR, filename)


# ── Image Detection ───────────────────────────────────────────────────────────
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
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)
    try:
        return jsonify(detect_image(path))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── Video Detection ───────────────────────────────────────────────────────────
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
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)
    try:
        return jsonify(detect_video(path))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── Voice Detection ───────────────────────────────────────────────────────────
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
    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)
    try:
        return jsonify(detect_voice(path))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("")
    print("=" * 52)
    print("  Deepfake Detection System")
    print("  Open this link in your browser:")
    print("")
    print("  -->  http://localhost:8000  <--")
    print("")
    print("  Press CTRL+C to stop the server")
    print("=" * 52)
    print("")
    app.run(debug=True, host="0.0.0.0", port=8000)