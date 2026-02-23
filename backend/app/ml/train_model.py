from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

from preprocess import load_and_preprocess


# === EXACT SAME MODEL AS SYSTEM EXPECTS ===
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


def train():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()

    model = TrustScoreModel(X_train.shape[1])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    # ===============================
    # TRAINING LOOP
    # ===============================
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # ===============================
    # MODEL EVALUATION
    # ===============================
    model.eval()

    with torch.no_grad():
        test_outputs = model(X_test).squeeze().numpy()

    y_pred = (test_outputs >= 0.5).astype(int)
    y_true = y_test.numpy()

    print("\n=== MODEL PERFORMANCE ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, test_outputs))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # ===============================
    # ROC CURVE
    # ===============================
    fpr, tpr, thresholds = roc_curve(y_true, test_outputs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    # ===============================
    # SAVE MODEL + SCALER
    # ===============================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "trust_model.pt")
    torch.save(model.state_dict(), model_path)

    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\nModel saved at:", model_path)
    print("Scaler saved at:", scaler_path)


if __name__ == "__main__":
    train()