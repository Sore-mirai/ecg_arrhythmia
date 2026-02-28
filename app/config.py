import os

# ──────────────────────────────────────────────
# Application Configuration
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ECG Signal Parameters
SAMPLING_RATE = 360          # MIT-BIH sampling rate (Hz)
WINDOW_SIZE = 720            # 2-second windows (360 * 2)
WINDOW_STRIDE = 180          # 0.5-second stride for sliding window
LEAD_INDEX = 0               # Use MLII lead (channel 0)

# AAMI Beat Classification Mapping
# Maps MIT-BIH annotation symbols to 5 AAMI classes
AAMI_CLASSES = {
    "N": 0,   # Normal
    "S": 1,   # Supraventricular ectopic
    "V": 2,   # Ventricular ectopic
    "F": 3,   # Fusion
    "Q": 4,   # Unknown / Paced
}

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unknown/Paced (Q)"]
CLASS_SHORT = ["N", "S", "V", "F", "Q"]

# MIT-BIH symbol → AAMI class mapping
SYMBOL_TO_AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# Model Hyperparameters
NUM_CLASSES = 5
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

# Confidence Threshold for alerts
CONFIDENCE_THRESHOLD = 0.60

# Model file path
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
