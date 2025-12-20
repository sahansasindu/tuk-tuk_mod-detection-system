# -----------------------------
# LEGAL OBJECT DETECTION IMPORTS
# -----------------------------
import os
import cv2
from ultralytics import YOLO

# -----------------------------
# MODEL CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "weights", "legal_charge_cal_model", "yolov8s_trained.pt"
)

CLASS_NAMES = ["LED_HeadLight", "Wind_Deflector"]

# Charges (Rs.)
LED_HEADLIGHT_CHARGE = 1000
WIND_DEFLECTOR_CHARGE = 1000

# -----------------------------
# Load YOLO model
# -----------------------------
object_model = YOLO(MODEL_PATH)

# -----------------------------
# LEGAL OBJECT PREDICTION + LAW
# -----------------------------
def predict_legal_objects_with_law(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image file")

    results = object_model(img)
    boxes = results[0].boxes

    led_detected = False
    deflector_detected = False

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id]

        if cls_name == "LED_HeadLight":
            led_detected = True

        elif cls_name == "Wind_Deflector":
            # Even one deflector is sufficient
            deflector_detected = True

    # -----------------------------
    # Charge calculation
    # -----------------------------
    total_charge = 0

    if led_detected:
        total_charge += LED_HEADLIGHT_CHARGE

    if deflector_detected:
        total_charge += WIND_DEFLECTOR_CHARGE

    # -----------------------------
    # Legal response formatting
    # -----------------------------
    return {
        "LED_HeadLights_detected": "YES" if led_detected else "NO",
        "Wind_Deflectors_detected": "YES" if deflector_detected else "NO",
        "charges": {
            "LED_HeadLight": LED_HEADLIGHT_CHARGE if led_detected else 0,
            "Wind_Deflector": WIND_DEFLECTOR_CHARGE if deflector_detected else 0
        },
        "total_charge_rs": total_charge,
        "law_reference": {
            "authority": "Department of Motor Traffic (Sri Lanka)",
            "approval_year": "2022–2023",
            "regulations": [
                "Replacing main lights with LED lights – Rs. 1000",
                "Installing wind deflectors – Rs. 1000"
            ]
        }
    }
