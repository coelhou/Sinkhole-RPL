"""
Loading the dataset and producing the train/test split.
Used by train_models.py (and indirectly by anything that needs X_test/y_test,
though those are cached to disk by train_models.py so this module usually
only needs to run once).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, DROP_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE


def load_data(path: str = DATA_PATH):
    """Load the CSV and split it into numeric features X and binary target y."""
    df = pd.read_csv(path)
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[TARGET_COL].astype(int)

    print("Data shape:", X.shape, "Positive rate:", y.mean().round(4))
    return X, y


def split_data(X, y):
    """Stratified train/test split, plus the scale_pos_weight diagnostic for
    imbalanced classes (used by XGBoost)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Imbalance (train): positives={n_pos}, negatives={n_neg}, "
          f"scale_pos_weight≈{scale_pos_weight:.2f}")

    return X_train, X_test, y_train, y_test, scale_pos_weight
