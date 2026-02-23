import os
import torch
import joblib
import numpy as np
from .model import TrustScoreModel

# === Base directory ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# === Correct paths ===
MODEL_PATH  = os.path.join(BASE_DIR, "data", "models", "trust_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "features.pkl")

# === Threshold ===
THRESHOLD = 0.1

# === Load scaler once ===
scaler = joblib.load(SCALER_PATH)

# === Global model variable ===
model = None


def load_model():
    """
    Loads trained PyTorch model into memory.
    """
    global model

    input_dim = scaler.mean_.shape[0]

    model = TrustScoreModel(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()


# Load model at startup
load_model()


def predict_risk(features):
    """
    Returns only risk probability (float)
    """

    features = np.array(features, dtype=np.float32)
    expected_features = scaler.mean_.shape[0]

    if len(features) < expected_features:
        features = np.pad(features, (0, expected_features - len(features)))

    if len(features) > expected_features:
        features = features[:expected_features]

    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    return float(probability)   # ✅ ONLY float