import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch


def load_and_preprocess():
    # Absolute path to this file's directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Correct dataset path
    file_path = os.path.join(BASE_DIR, "data", "creditcard.csv")

    print("Loading dataset from:", file_path)

    df = pd.read_csv(file_path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=42
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
        scaler
    )