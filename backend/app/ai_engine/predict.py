import torch
import os
import joblib
import numpy as np
from .model import TrustScoreModel


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trust_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


# Load once at startup
scaler = joblib.load(SCALER_PATH)

model = None


def load_model():
    global model
    dummy_input = np.zeros((1, scaler.mean_.shape[0]))
    input_dim = dummy_input.shape[1]

    model = TrustScoreModel(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()


load_model()


def predict_risk(features):

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        risk_score = output.item()

    return float(risk_score)