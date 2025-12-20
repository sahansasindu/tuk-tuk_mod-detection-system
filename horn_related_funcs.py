# -----------------------------
# HORN DETECTION IMPORTS
# -----------------------------
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# ------------------------------
# HORN MODEL CONFIG (MUST MATCH TRAINING)
# ------------------------------
SR = 44100
DURATION = 3
SAMPLES = SR * DURATION
N_MELS = 64
HOP = 512
CONFIDENCE_THRESHOLD = 70.0  # percentage

# ------------------------------
# Load horn model + labels
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

horn_model = load_model(
    os.path.join(BASE_DIR, "weights", "sound_model", "cnn_horn_classifier.h5")
)

HORN_CLASS_NAMES = np.load(
    os.path.join(BASE_DIR, "weights", "sound_model", "label_encoder_classes.npy")
)


# ------------------------------
# HORN AUDIO PREPROCESSING
# ------------------------------
def load_audio_fixed(path):
    audio, sr = librosa.load(path, sr=SR)

    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    return audio

def extract_melspec(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=N_MELS,
        hop_length=HOP
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# ------------------------------
# HORN PREDICTION + LEGAL MAPPING
# ------------------------------
def predict_audio_with_law(audio_path):
    audio = load_audio_fixed(audio_path)
    mel = extract_melspec(audio)

    mel = np.expand_dims(mel, axis=-1)
    mel = np.expand_dims(mel, axis=0)

    preds = horn_model.predict(mel, verbose=0)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100
    label = HORN_CLASS_NAMES[class_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "horn_type": "Uncertain / Non-horn",
            "confidence": round(confidence, 2),
            "legal": False,
            "message": "Prediction suppressed due to low confidence"
        }

    # ------------------------------
    # Legal interpretation
    # ------------------------------
    if label.lower() == "multi_tone":
        return {
            "horn_type": "Multi-tone horn",
            "confidence": round(confidence, 2),
            "legal": False,
            "law": "Motor Traffic (Amendment) Act No. 8 of 2009, Section 155(3)",
            "violation_message": (
                "No person shall use a motor vehicle that has been equipped with a multi-tone horn "
                "sounding a succession of different notes, or with any other sound-producing device "
                "giving a harsh, shrill, loud or alarming noise, except for emergency vehicles."
            )
        }

    elif label.lower() == "single_tone":
        return {
            "horn_type": "Single-tone horn",
            "confidence": round(confidence, 2),
            "legal": True,
            "law": "Motor Traffic (Amendment) Act No. 8 of 2009, Section 155(3)",
            "violation_message": (
                "Single-tone horns are permitted provided they do not produce harsh, shrill, "
                "or alarming noise."
            )
        }
