from flask import Flask, render_template, request, jsonify, url_for
import os
import uuid
import cv2

# -----------------------------
# Flask App Config
# -----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -----------------------------
# IMPORT DETECTORS
# -----------------------------
from illegal_modification_detector import detect_illegal_modification
from windscreen_detector import detect_windscreen
from horn_related_funcs import predict_audio_with_law
from legal_object_predictor import predict_objects_with_charges


# -----------------------------
# ROUTES (PAGES)
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/page1")
def page1():
    return render_template("illegalmodification.html")

@app.route("/page2")
def page2():
    return render_template("windscreenvisibility.html")

@app.route("/page3")
def page3():
    return render_template("horndetection.html")

@app.route("/page4")
def page4():
    return render_template("legal_object_detection.html")


# -----------------------------
# ILLEGAL MODIFICATION API
# -----------------------------
@app.route("/detect_illegal_modification", methods=["POST"])
def detect_illegal():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result = detect_illegal_modification(filepath, app.config["UPLOAD_FOLDER"])

    return jsonify({
        **result,
        "result_image": url_for(
            "static",
            filename=f"uploads/{result['output_image']}"
        )
    })


# -----------------------------
# WINDSCREEN VISIBILITY API
# -----------------------------
@app.route("/detect_windshield", methods=["POST"])
def detect_windshield():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    result = detect_windscreen(filepath, app.config["UPLOAD_FOLDER"])

    return jsonify({
        **result,
        "result_image": url_for(
            "static",
            filename=f"uploads/{result['output_image']}"
        )
    })


# -----------------------------
# HORN DETECTION API
# -----------------------------
@app.route("/detect_horn", methods=["POST"])
def detect_horn():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"})

    file = request.files["audio"]
    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        return jsonify(predict_audio_with_law(filepath))
    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# LEGAL OBJECT DETECTION API
# -----------------------------
@app.route("/detect_legal_objects", methods=["POST"])
def detect_legal_objects():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        return jsonify(predict_objects_with_charges(filepath))
    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
