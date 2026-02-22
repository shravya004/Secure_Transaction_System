# Placeholder file for inference.py
import torch
import numpy as np
from .model import TrustScoreModel

MODEL_PATH = "data/models/trust_model.pt"

def load_model(input_dim):
    model = TrustScoreModel(input_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def predict_trust_score(model, features):
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        score = model(features)
    return float(score.item())
