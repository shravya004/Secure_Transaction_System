import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch


def load_and_preprocess(return_df=False, random_seed=42):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
    file_path = os.path.join(BACKEND_DIR, "data", "creditcard.csv")

    print("Loading dataset from:", file_path)

    df = pd.read_csv(file_path)

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df["Amount"] = np.log1p(df["Amount"])
    df["Time"] = df["Time"] / df["Time"].max()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # ==================================================
    # Train/Test Split
    # ==================================================
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_seed,
    )

    # ==================================================
    # Train/Validation Split
    # ==================================================
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.10,
        stratify=y_trainval,
        random_state=random_seed,
    )

    # ==================================================
    # Scaling (fit ONLY on training)
    # ==================================================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = scaler.transform(X_test)

    # ==================================================
    # SMOTE ONLY on TRAIN
    # ==================================================
    smote = SMOTE(random_state=random_seed)

    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_scaled,
        y_train,
    )

    # Shuffle after SMOTE

    idx = np.random.RandomState(random_seed).permutation(
        len(X_train_balanced)
    )

    X_train_balanced = X_train_balanced[idx]
    y_train_balanced = np.array(y_train_balanced)[idx]

    # ==================================================
    # Convert to tensors
    # ==================================================
    X_train_t = torch.tensor(X_train_balanced, dtype=torch.float32)

    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_t = torch.tensor(y_train_balanced, dtype=torch.float32)

    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    if return_df:
        return (
            X_train_t,
            X_val_t,
            X_test_t,
            y_train_t,
            y_val_t,
            y_test_t,
            scaler,
            df,
        )

    return (
        X_train_t,
        X_val_t,
        X_test_t,
        y_train_t,
        y_val_t,
        y_test_t,
        scaler,
    )


if __name__ == "__main__":

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = load_and_preprocess()

    print("\nTraining")
    print((y_train == 0).sum().item(), (y_train == 1).sum().item())

    print("\nValidation")
    print((y_val == 0).sum().item(), (y_val == 1).sum().item())

    print("\nTest")
    print((y_test == 0).sum().item(), (y_test == 1).sum().item())