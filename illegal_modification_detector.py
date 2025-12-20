import os
import cv2
import uuid
from ultralytics import YOLO

# -----------------------------
# MODEL CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")

model = YOLO(MODEL_PATH)

CLASS_NAMES = [
    'orginal_three_wheeler',
    'other_vehical',
    'modify_three_Wheeler',
    'air_cleaner',
    'fire_extinguisher',
    'front_aerial',
    'rim_cap'
]

ILLEGAL_CLASSES = {
    'modify_three_Wheeler',
    'air_cleaner',
    'fire_extinguisher',
    'front_aerial',
    'rim_cap'
}

# -----------------------------
# DETECTION FUNCTION
# -----------------------------
def detect_illegal_modification(image_path, upload_folder):
    img = cv2.imread(image_path)
    results = model(img)
    boxes = results[0].boxes

    detected_classes = [
        CLASS_NAMES[int(c)] for c in boxes.cls
    ] if len(boxes) else []

    illegal_detected = any(cls in ILLEGAL_CLASSES for cls in detected_classes)

    # Annotated image
    annotated_img = results[0].plot()
    out_name = f"illegal_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(upload_folder, out_name)
    cv2.imwrite(out_path, annotated_img)

    # -----------------------------
    # Detailed violation message
    # -----------------------------
    if illegal_detected:
        violation_message = (
            "⚠️ Detected Illegal Vehicle Modification!\n"
            "--------------------------------------------\n"
            "Gazette No: 2240/37 (2021)\n"
            "Date: Saturday, August 14, 2021\n"
            "Link: https://www.documents.gov.lk/view/extra-gazettes/2021/8/2240-37_E.pdf\n\n"
            "Violation of REGULATIONS:\n"
            " 1) (a) Exceed the weight, dimensions, or limitations of its prototype;\n"
            "    (b) Alter its shape, design, or external appearance;\n"
            "    (d) Lose its equilibrium;\n\n"
            " 2) Sharp-edged accessories causing danger/obstruction are prohibited.\n\n"
            " 6) Tyres, mud guards, or wheel covers must NOT protrude.\n"
            "--------------------------------------------"
        )
    else:
        violation_message = "✅ No illegal modification detected."

    return {
        "illegal_detected": illegal_detected,
        "detected_classes": detected_classes,
        "violation_message": violation_message,
        "output_image": out_name
    }
