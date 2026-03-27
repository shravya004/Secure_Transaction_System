import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch


def load_and_preprocess(return_df=False):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "creditcard.csv")
    print("Loading dataset from:", file_path)

    df = pd.read_csv(file_path)

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    df['Amount'] = np.log1p(df['Amount'])
    df['Time'] = df['Time'] / df['Time'].max()

    # ===============================
    # SPLIT FEATURES & TARGET
    # ===============================
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # ===============================
    # STEP 1: SPLIT FIRST (no leakage)
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ===============================
    # STEP 2: SCALE (fit only on train)
    # ===============================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ===============================
    # STEP 3: SMOTE (only on train)
    # ===============================
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # ===============================
    # STEP 4: SHUFFLE (critical after SMOTE)
    # ===============================
    shuffle_idx = np.random.RandomState(42).permutation(len(X_train_resampled))
    X_train_resampled = X_train_resampled[shuffle_idx]
    y_train_resampled = np.array(y_train_resampled)[shuffle_idx]

    # ===============================
    # CONVERT TO TENSORS
    # ===============================
    X_train_t = torch.tensor(X_train_resampled, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_scaled,     dtype=torch.float32)
    y_train_t = torch.tensor(y_train_resampled, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values,     dtype=torch.float32)

    if return_df:
        return X_train_t, X_test_t, y_train_t, y_test_t, scaler, df
    else:
        return X_train_t, X_test_t, y_train_t, y_test_t, scaler


# ===============================
# TEST PREPROCESS STANDALONE
# ===============================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)
    print("Train class balance — Legit:", (y_train==0).sum().item(), "Fraud:", (y_train==1).sum().item())
    print("Test class balance  — Legit:", (y_test==0).sum().item(),  "Fraud:", (y_test==1).sum().item())