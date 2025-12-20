import os
import cv2
import uuid
from ultralytics import YOLO

# -----------------------------
# MODEL CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights", "best1.pt")

model = YOLO(MODEL_PATH)

WINDSHIELD_CLASS = 0
STICKER_CLASS = 1
MIN_VISIBILITY = 70  # %

# -----------------------------
# DETECTION FUNCTION
# -----------------------------
def detect_windscreen(image_path, upload_folder):
    img = cv2.imread(image_path)
    results = model(img)[0]

    windshield_area = 0
    sticker_area = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if cls == WINDSHIELD_CLASS:
            windshield_area += area
        elif cls == STICKER_CLASS:
            sticker_area += area

    annotated_img = results.plot()
    out_name = f"wind_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(upload_folder, out_name)
    cv2.imwrite(out_path, annotated_img)

    if windshield_area == 0:
        return {
            "legal": False,
            "visibility": 0,
            "message": "❌ No windshield detected",
            "output_image": out_name
        }

    visibility = ((windshield_area - sticker_area) / windshield_area) * 100
    legal = visibility >= MIN_VISIBILITY

    return {
        "legal": legal,
        "visibility": round(visibility, 2),
        "message": "✅ LEGAL" if legal else "❌ ILLEGAL – Visibility below 70%",
        "output_image": out_name
    }
