import os
import torch
import joblib
import numpy as np
from .model import TrustScoreModel


# === Base directory ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# === Correct paths (matching your training script) ===
MODEL_PATH  = os.path.join(BASE_DIR, "data", "models", "trust_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "features.pkl")


# === Load scaler once ===
scaler = joblib.load(SCALER_PATH)

# === Global model variable ===
model = None


def load_model():
    """
    Loads trained PyTorch model into memory.
    """
    global model

    # Determine input dimension from scaler
    input_dim = scaler.mean_.shape[0]

    model = TrustScoreModel(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()


# Load model at startup
load_model()


def predict_risk(features):
    """
    Takes a list of numerical features
    Returns fraud risk score between 0 and 1
    """

    # Convert to numpy
    features = np.array(features, dtype=np.float32)

    # Expected number of features from training
    expected_features = scaler.mean_.shape[0]

    # Pad with zeros if frontend sends fewer features
    if len(features) < expected_features:
        features = np.pad(features, (0, expected_features - len(features)))

    # Trim if too many (safety)
    if len(features) > expected_features:
        features = features[:expected_features]

    # Reshape for scaler
    features = features.reshape(1, -1)

    # Scale
    features_scaled = scaler.transform(features)

    # Convert to tensor
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        risk_score = output.item()

    return float(risk_score)