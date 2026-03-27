from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from preprocess import load_and_preprocess


# ===============================
# MODEL DEFINITION
# ===============================
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
        x = self.fc3(x)
        return x


# ===============================
# TRAIN FUNCTION
# ===============================
def train():
    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess(return_df=True)

    # ===============================
    # SHUFFLE BEFORE VAL SPLIT
    # ===============================
    perm    = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(42))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # ===============================
    # VALIDATION SPLIT
    # ===============================
    val_size = int(0.1 * len(X_train))
    X_val,   y_val   = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]

    print(f"\nTrain size : {len(X_train)}")
    print(f"Val size   : {len(X_val)}")
    print(f"Test size  : {len(X_test)}")
    print(f"Val class balance — Legit: {(y_val==0).sum().item()}, Fraud: {(y_val==1).sum().item()}\n")

    # ===============================
    # MODEL + OPTIMIZER
    # ===============================
    model     = TrustScoreModel(X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # ===============================
    # EARLY STOPPING SETUP
    # ===============================
    epochs        = 500
    patience      = 30
    best_val_loss = float('inf')
    counter       = 0

    BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(BASE_DIR, "best_model.pt")

    train_losses = []
    val_losses   = []

    # ===============================
    # TRAINING LOOP WITH TQDM
    # ===============================
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch", colour="green")

    for epoch in epoch_bar:
        # --- Train ---
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train).squeeze()
        loss    = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val).squeeze()
            val_loss    = criterion(val_outputs, y_val).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        # --- Update tqdm bar ---
        epoch_bar.set_postfix({
            "Train Loss": f"{loss.item():.4f}",
            "Val Loss":   f"{val_loss:.4f}",
            "Patience":   f"{counter}/{patience}"
        })

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter       = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1

        if counter >= patience:
            tqdm.write(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
            break

    tqdm.write("✅ Best model loaded.")
    model.load_state_dict(torch.load(best_model_path))

    # ===============================
    # THRESHOLD TUNING ON VAL SET
    # ===============================
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val).squeeze()).numpy()

    y_val_np = y_val.numpy()

    best_threshold, best_f1 = 0.5, 0
    print("\n🔍 Tuning threshold on validation set...")

    for t in tqdm(np.arange(0.1, 0.9, 0.05), desc="Threshold Tuning", colour="blue"):
        y_pred_temp = (val_probs >= t).astype(int)
        f1          = f1_score(y_val_np, y_pred_temp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    print(f"\n✅ Best Threshold: {best_threshold:.2f}")

    # ===============================
    # FINAL EVALUATION ON TEST SET
    # ===============================
    with torch.no_grad():
        logits       = model(X_test).squeeze()
        test_outputs = torch.sigmoid(logits).numpy()

    y_true = y_test.numpy()
    y_pred = (test_outputs >= best_threshold).astype(int)

    # ===============================
    # METRICS — FINAL CAKE 🎂
    # ===============================
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, test_outputs)

    print("\n" + "="*45)
    print("🎂           MODEL PERFORMANCE           🎂")
    print("="*45)
    print(f"  🎯 Accuracy  : {acc:.4f}")
    print(f"  🎯 Precision : {prec:.4f}")
    print(f"  🎯 Recall    : {rec:.4f}")
    print(f"  🎯 F1 Score  : {f1:.4f}")
    print(f"  🎯 ROC AUC   : {auc:.4f}")
    print("="*45)

    # ===============================
    # LOSS CURVE
    # ===============================
    plt.figure()
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses,   label="Val Loss",   color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

    # ===============================
    # ROC CURVE
    # ===============================
    fpr, tpr, _ = roc_curve(y_true, test_outputs)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()

    # ===============================
    # CORRELATION HEATMAP
    # ===============================
    plt.figure(figsize=(12, 10))
    corr = df.iloc[:, :10].corr()
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()

    # ===============================
    # SAVE FINAL MODEL + SCALER
    # ===============================
    model_path  = os.path.join(BASE_DIR, "trust_model.pt")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    torch.save(model.state_dict(), model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n💾 Model saved at : {model_path}")
    print(f"💾 Scaler saved at: {scaler_path}")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    train()