"""
FRAUD-X Model Trainer v3
Uses actual fraud samples from dataset for sanity check
Fixes threshold to match real data distribution
"""

import os, sys, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "backend", "data", "raw", "creditcard.csv")
MODELS_DIR = os.path.join(BASE_DIR, "backend", "data", "models")
SCALER_OUT = os.path.join(BASE_DIR, "backend", "data", "processed", "features.pkl")
MODEL_OUT  = os.path.join(BASE_DIR, "backend", "data", "models", "trust_model.pt")

class TrustScoreModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, 64)
        self.fc2     = nn.Linear(64, 32)
        self.fc3     = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

os.makedirs(MODELS_DIR, exist_ok=True)

print("[1/6] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"      {len(df):,} rows | Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

feature_cols = [c for c in df.columns if c != "Class"]
X = df[feature_cols].values.astype(np.float32)
y = df["Class"].values.astype(np.float32)
input_dim = X.shape[1]
print(f"[2/6] Input dim: {input_dim}")

# Save a real fraud sample and a real legit sample for sanity check BEFORE scaling
fraud_sample = df[df['Class'] == 1].iloc[0][feature_cols].values.astype(np.float32)
legit_sample = df[df['Class'] == 0].iloc[0][feature_cols].values.astype(np.float32)
print(f"      Real fraud sample amount: ${fraud_sample[-1]:.2f}")
print(f"      Real legit sample amount: ${legit_sample[-1]:.2f}")

print("[3/6] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_OUT)
print(f"      Scaler saved -> {SCALER_OUT}")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[4/6] Split: {len(X_train):,} train / {len(X_test):,} test")

# Use moderate pos_weight — 50x is enough to learn fraud without saturation
pos_weight = torch.tensor([50.0], dtype=torch.float32)
print(f"      pos_weight: {pos_weight.item():.1f}x")

loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=256, shuffle=True
)

print("[5/6] Training...")
model     = TrustScoreModel(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn   = nn.BCELoss(weight=pos_weight)

best_auc   = 0
best_state = None
EPOCHS     = 25

for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb).squeeze(), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if ep % 5 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            scores = model(torch.tensor(X_test)).squeeze().numpy()
        auc = roc_auc_score(y_test, scores)
        print(f"      Epoch {ep:02d}/{EPOCHS}  loss={avg_loss:.4f}  AUC={auc:.4f}")
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        print(f"      Epoch {ep:02d}/{EPOCHS}  loss={avg_loss:.4f}")

# Save best
model.load_state_dict(best_state)
torch.save(model.state_dict(), MODEL_OUT)
print(f"\n[6/6] Best AUC: {best_auc:.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    scores = model(torch.tensor(X_test)).squeeze().numpy()

preds = (scores > 0.5).astype(int)
print("\n      Classification Report:")
print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))

# Sanity check with REAL samples from dataset
print("      Sanity check (using real dataset samples):")
fraud_scaled = scaler.transform(fraud_sample.reshape(1, -1))
legit_scaled = scaler.transform(legit_sample.reshape(1, -1))

with torch.no_grad():
    fraud_score = model(torch.tensor(fraud_scaled, dtype=torch.float32)).item()
    legit_score = model(torch.tensor(legit_scaled, dtype=torch.float32)).item()

print(f"      Real fraud sample score : {fraud_score:.4f} (should be > 0.5)")
print(f"      Real legit sample score : {legit_score:.4f} (should be < 0.5)")

# Also show score distribution on test set
fraud_test_scores = scores[y_test == 1]
legit_test_scores = scores[y_test == 0]
print(f"\n      Fraud scores  — mean: {fraud_test_scores.mean():.4f}  max: {fraud_test_scores.max():.4f}")
print(f"      Legit scores  — mean: {legit_test_scores.mean():.4f}  max: {legit_test_scores.max():.4f}")

# Print a few real fraud V-feature values so frontend sliders make sense
print(f"\n      Example real fraud V1-V6 values (for testing in UI):")
fraud_examples = df[df['Class'] == 1][['V1','V2','V3','V4','V5','V6','Amount']].head(3)
print(fraud_examples.to_string())

print(f"\n      Model saved -> {MODEL_OUT}")
print("\nDONE! Now run: cd backend && uvicorn app.main:app --reload\n")
