import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from .model import TrustScoreModel


# --------------------------------------------------
# Path Handling (Robust & Clean)
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "transactions.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "trust_model.pt")
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# --------------------------------------------------
# Training Function
# --------------------------------------------------

def train_model():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    # -----------------------------------------
    # Feature Scaling
    # -----------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)

    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # -----------------------------------------
    # Model Setup
    # -----------------------------------------
    model = TrustScoreModel(input_dim=X_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------------------
    # Training Loop
    # -----------------------------------------
    print("Training started...\n")

    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/20 | Loss: {loss.item():.6f}")

    # -----------------------------------------
    # Save Model
    # -----------------------------------------
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\nModel and scaler saved successfully.")
    print(f"Model path: {MODEL_SAVE_PATH}")
    print(f"Scaler path: {SCALER_SAVE_PATH}")


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    train_model()