import os
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TrustScoreModel(nn.Module):
    def __init__(self, input_dim):
        super(TrustScoreModel, self).__init__()
        self.fc1     = nn.Linear(input_dim, 64)
        self.fc2     = nn.Linear(64, 32)
        self.fc3     = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_PATH   = os.path.join(PROJECT_ROOT, "data", "models", "trust_model.pt")
SCALER_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "features.pkl")

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Read input_dim from scaler (don't hardcode 30)
input_dim = scaler.mean_.shape[0]

# Load model
model = TrustScoreModel(input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def predict_transaction(features: list):
    arr = np.array(features, dtype=np.float32)

    # If frontend sends fewer features than model expects,
    # pad with zeros on the right
    if arr.shape[0] < input_dim:
        arr = np.pad(arr, (0, input_dim - arr.shape[0]))
    # If too many, truncate
    elif arr.shape[0] > input_dim:
        arr = arr[:input_dim]

    arr = arr.reshape(1, -1)
    scaled = scaler.transform(arr)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor).item()

    return float(output)