"""
Shared configuration for the classification pipeline.
Centralizing these values keeps train_models.py, evaluate_models.py, and
feature_importance.py consistent with each other.
"""

# Data
DATA_PATH = "final_dataset_enriched.csv"
DROP_COLS = ["ID", "label", "attack"]   # non-feature columns dropped from X
TARGET_COL = "attack"

# Train/test split
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Cross-validation
N_SPLITS = 5
SCORING = {"roc_auc": "roc_auc", "f1": "f1", "precision": "precision", "recall": "recall"}
REFIT_METRIC = "roc_auc"

# Where trained models / intermediate results are cached between scripts
ARTIFACTS_DIR = "artifacts"
