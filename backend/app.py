from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from detector import detect_fake

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["file"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = detect_fake(filepath)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)