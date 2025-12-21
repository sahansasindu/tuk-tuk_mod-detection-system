# -----------------------------
# LEGAL OBJECT DETECTION IMPORTS
# -----------------------------
import os
import cv2
import base64
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
LED_CHARGE = 1000
DEFLECTOR_CHARGE = 1000

# Legal Reference
LAW_REFERENCE = {
    "authority": "Department of Motor Traffic (Sri Lanka)",
    "regulation": "Motor Tricycle Modernization Program (2023)",
    "details": [
        "Replacing main lights with LED lights – Rs. 1000",
        "Installing wind deflectors – Rs. 1000"
    ],
    "source": "https://dmt.gov.lk/index.php?option=com_content&view=article&id=98:2023-07-13-11-08-43&catid=8&Itemid=140&lang=en"
}

# -----------------------------
# Load YOLO model (ONCE)
# -----------------------------
object_model = YOLO(MODEL_PATH)

# -----------------------------
# IMAGE → BASE64 CONVERTER
# -----------------------------
def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# -----------------------------
# OBJECT PREDICTION + CHARGES
# -----------------------------
def predict_objects_with_charges(image_path):
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
            # One deflector is sufficient according to visual limitations
            deflector_detected = True

    # -----------------------------
    # Charge Calculation
    # -----------------------------
    total_charge = 0
    if led_detected:
        total_charge += LED_CHARGE
    if deflector_detected:
        total_charge += DEFLECTOR_CHARGE

    # -----------------------------
    # Annotated Image
    # -----------------------------
    annotated_img = results[0].plot()
    annotated_img_base64 = encode_image_to_base64(annotated_img)

    # -----------------------------
    # Structured Response
    # -----------------------------
    return {
        "LED_HeadLights_detected": "YES" if led_detected else "NO",
        "Wind_Deflectors_detected": "YES" if deflector_detected else "NO",
        "total_charge_rs": total_charge,
        "law_reference": LAW_REFERENCE,
        "annotated_image_base64": annotated_img_base64
    }
