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

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train).squeeze()
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Save model
    model_path = os.path.join(BASE_DIR, "trust_model.pt")
    torch.save(model.state_dict(), model_path)

    # Save scaler
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\nModel saved at:", model_path)
    print("Scaler saved at:", scaler_path)


if __name__ == "__main__":
    train()