import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# === SAME MODEL ARCHITECTURE AS TRAINED ===
class TrustScoreModel(nn.Module):
    def __init__(self, input_dim):
        super(TrustScoreModel, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# === Load Model + Scaler ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "trust_model.pt")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Load scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Load model
input_dim = 30  # credit card dataset feature count
model = TrustScoreModel(input_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def predict_transaction(features: list):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    tensor_input = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor_input).item()

    return float(output)