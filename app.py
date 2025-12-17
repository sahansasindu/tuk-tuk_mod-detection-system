from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model
# model = YOLO(r"D:\New folder\my_web_app\weights\best.pt")
model = YOLO(r"..\weights\best.pt")

# Class names
class_names = [
    'orginal_three_wheeler',
    'other_vehical',
    'modify_three_Wheeler',
    'air_cleaner',
    'fire_extinguisher',
    'front_aerial',
    'rim_cap'
]

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
    return render_template("modificationcharges.html")



# -----------------------------
# DETECTION ROUTE
# -----------------------------
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]

    # Save uploaded file
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Read and run YOLO
    img = cv2.imread(filepath)
    results = model(img)
    boxes = results[0].boxes

    # Annotated image
    annotated_img = results[0].plot()
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"det_{filename}")
    cv2.imwrite(output_path, annotated_img)

    # Extract detected classes
    detected_classes = [class_names[int(c)] for c in boxes.cls] if len(boxes) else []

    # Illegal categories
    illegal_classes = {
        'modify_three_Wheeler',
        'air_cleaner',
        'fire_extinguisher',
        'front_aerial',
        'rim_cap'
    }

    illegal_detected = any(c in illegal_classes for c in detected_classes)

    # ---- FIXED INDENT ERROR HERE ----
    if illegal_detected:
        violation_message = (
            "⚠️ Detected Illegal Vehicle Modification!\n"
            "--------------------------------------------\n"
            "Gazette No: 2240/37\n"
            "Date: Saturday, August 14, 2021\n"
            "Link: https://www.documents.gov.lk/view/extra-gazettes/2021/8/2240-37_E.pdf\n\n"
            "Violation of REGULATIONS:\n"
            " 1) (a) exceed the weight, dimensions or limitations of its prototype;\n"
            "    (b) alter its shape, design or external appearance;\n"
            "    (d) loose its equilibrium;\n\n"
            " 2) Sharp-edged accessories causing danger/obstruction are prohibited.\n\n"
            " 6) Tyres, mud guards, or wheel covers must NOT protrude.\n"
            "--------------------------------------------"
        )
    else:
        violation_message = "✅ No illegal modification detected."

    # Response JSON
    return jsonify({
        "result_image": url_for('static', filename=f"uploads/det_{filename}"),
        "illegal_detected": illegal_detected,
        "detected_classes": detected_classes,
        "violation_message": violation_message
    })


# -----------------------------
# Windscreen DETECTION ROUTE
# -----------------------------
# wind_model = YOLO(r"D:\New folder\my_web_app\weights\best1.pt")
wind_model = YOLO(r"..\weights\best1.pt")

@app.route("/detect_windshield", methods=["POST"])
def detect_windshield():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["image"]

    # Save uploaded file
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = cv2.imread(filepath)
    results = wind_model(img)[0]

    windshield_area = 0
    sticker_area = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if cls == 0:  # Windshield class
            windshield_area += area
        elif cls == 1:  # Sticker class
            sticker_area += area

    # Draw annotated image
    annotated_img = results.plot()
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"wind_{filename}")
    cv2.imwrite(output_path, annotated_img)

    if windshield_area == 0:
        return jsonify({
            "visibility": 0,
            "legal": False,
            "result_image": url_for('static', filename=f"uploads/wind_{filename}"),
            "message": "❌ No windshield detected",
            "violation_message": (
                "⚠️ No windshield detected – vehicle may not comply with safety regulations.\n"
            )
        })

    # Calculate visibility percentage
    visible_area = windshield_area - sticker_area
    visibility = (visible_area / windshield_area) * 100
    legal = visibility >= 70  # Rules: must be at least 70% visible

    # Detailed violation message if illegal
    if not legal:
        violation_message = (
            "❌ ILLEGAL – Visibility below 70%\n"
            "--------------------------------------------\n"
            "Gazette No: 2240/37\n"
            "Date: Saturday, August 14, 2021\n"
            "Link: https://www.documents.gov.lk/view/extra-gazettes/2021/8/2240-37_E.pdf\n\n"
            "Regulations:\n"
            " 1) Windscreen must maintain at least 70% visibility;\n"
            " 2) Stickers or accessories must not obstruct view;\n"
            "--------------------------------------------"
        )
    else:
        violation_message = "✅ LEGAL – Visibility is acceptable."

    return jsonify({
        "visibility": round(visibility, 2),
        "legal": legal,
        "result_image": url_for('static', filename=f"uploads/wind_{filename}"),
        "message": "LEGAL ✔" if legal else "❌ ILLEGAL – Visibility below 70%",
        "violation_message": violation_message
    })




# -----------------------------
# Run application
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
