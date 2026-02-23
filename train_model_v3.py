"""
FRAUD-X Model Trainer v4
- Proper class imbalance handling
- Uses BCEWithLogitsLoss (CORRECT way)
- Threshold = 0.1 for fraud flag
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# ---------------- PATHS ----------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "backend", "data", "raw", "creditcard.csv")
MODEL_OUT  = os.path.join(BASE_DIR, "backend", "data", "models", "trust_model.pt")
SCALER_OUT = os.path.join(BASE_DIR, "backend", "data", "processed", "scaler.pkl")

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_OUT), exist_ok=True)

THRESHOLD = 0.1

# ---------------- MODEL ----------------
class TrustScoreModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # NO sigmoid here

# ---------------- LOAD DATA ----------------
print("\n[1/6] Loading dataset...")
df = pd.read_csv(CSV_PATH)

print(f"Rows: {len(df):,}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

feature_cols = [c for c in df.columns if c != "Class"]
X = df[feature_cols].values.astype(np.float32)
y = df["Class"].values.astype(np.float32)

input_dim = X.shape[1]

# ---------------- SCALE ----------------
print("\n[2/6] Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_OUT)
print("Scaler saved ->", SCALER_OUT)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n[3/6] Train/Test split done")

# ---------------- CLASS IMBALANCE FIX ----------------
pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()])
print("pos_weight:", pos_weight.item())

# ---------------- DATALOADER ----------------
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=512,
    shuffle=True
)

# ---------------- TRAIN ----------------
print("\n[4/6] Training...")
model = TrustScoreModel(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

EPOCHS = 20
best_auc = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb).squeeze()
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Evaluate every epoch
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test)).squeeze()
        probs = torch.sigmoid(logits).numpy()

    auc = roc_auc_score(y_test, probs)

    print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_state = model.state_dict()

# ---------------- SAVE BEST ----------------
model.load_state_dict(best_state)
torch.save(model.state_dict(), MODEL_OUT)

print("\n[5/6] Best AUC:", best_auc)
print("Model saved ->", MODEL_OUT)

# ---------------- FINAL REPORT ----------------
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_test)).squeeze()
    probs = torch.sigmoid(logits).numpy()

preds = (probs > THRESHOLD).astype(int)

print("\n[6/6] Classification Report (Threshold = 0.1)\n")
print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))

# ---------------- SANITY CHECK ----------------
fraud_sample = df[df["Class"] == 1].iloc[0][feature_cols].values.astype(np.float32)
legit_sample = df[df["Class"] == 0].iloc[0][feature_cols].values.astype(np.float32)

fraud_scaled = scaler.transform(fraud_sample.reshape(1, -1))
legit_scaled = scaler.transform(legit_sample.reshape(1, -1))

with torch.no_grad():
    fraud_score = torch.sigmoid(model(torch.tensor(fraud_scaled))).item()
    legit_score = torch.sigmoid(model(torch.tensor(legit_scaled))).item()

print("\nSanity Check:")
print("Fraud sample score:", round(fraud_score, 4))
print("Legit sample score:", round(legit_score, 4))

print("\nDONE ")