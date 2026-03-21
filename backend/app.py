from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from detector_image import detect_image
from detector_video import detect_video

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/detect/image", methods=["POST"])
def detect_img():
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    result = detect_image(path)
    return jsonify(result)


@app.route("/detect/video", methods=["POST"])
def detect_vid():
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    result = detect_video(path)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)