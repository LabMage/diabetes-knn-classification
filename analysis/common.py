"""
Shared data loading and train/test split for ALL models (kNN 1-4, DT, DT SMOTE, RF).

Every model script imports from here, so all models are evaluated on the same
untouched test set. Matches the shared preprocessing block in index.qmd:
duplicates removed, stratified 70/30 split, random_state=100.

Run directly to check the split:  python analysis/common.py
"""
import os
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Diabetes_binary"
RANDOM_STATE = 100
TEST_SIZE = 0.3
CONTINUOUS = ["BMI", "MentHlth", "PhysHlth"]


def load_data():
    """Raw dataset, local copy if present, otherwise downloaded from UCI."""
    if os.path.exists("cdc_data.csv"):
        return pd.read_csv("cdc_data.csv")
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=891)
    return pd.concat([d.data.features, d.data.targets], axis=1)


def get_split():
    """Deduplicated data -> one stratified split. Returns X_train, X_test, y_train, y_test (unscaled)."""
    cdc_df = load_data().drop_duplicates()
    X = cdc_df.drop(columns=[TARGET])
    y = cdc_df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


if __name__ == "__main__":
    raw = load_data()
    X_train, X_test, y_train, y_test = get_split()
    print(f"Raw rows: {len(raw)}   after removing duplicates: {len(X_train) + len(X_test)}")
    print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")
    print(f"Class 1 share  train: {y_train.mean():.4f}   test: {y_test.mean():.4f}")
    # Fingerprint of the test set: must be identical every run and for every model
    fp = hashlib.sha256(",".join(map(str, sorted(X_test.index))).encode()).hexdigest()[:12]
    print(f"Test set fingerprint: {fp}")
