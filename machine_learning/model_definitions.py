"""
Model + hyperparameter grid definitions.
Kept separate from train_models.py so a model can be edited or reused
(e.g. for a quick experiment) without touching the training/CV logic.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_models_and_grids(scale_pos_weight: float, random_state: int = 42):
    """Return a list of (name, estimator, param_grid) tuples, one per model."""
    models_and_grids = []

    # Logistic Regression (needs scaling)
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs"),
    )
    lr_grid = {
        "logisticregression__C": [0.1, 1, 10],
        "logisticregression__penalty": ["l2"],
    }
    models_and_grids.append(("LogisticRegression", lr, lr_grid))

    # KNN (needs scaling)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
    knn_grid = {
        "kneighborsclassifier__n_neighbors": [3, 5, 7, 11],
        "kneighborsclassifier__weights": ["uniform", "distance"],
        "kneighborsclassifier__p": [1, 2],  # Manhattan vs Euclidean
    }
    models_and_grids.append(("KNN", knn, knn_grid))

    # Random Forest (tree-based)
    rf = RandomForestClassifier(
        n_estimators=300, random_state=random_state, class_weight="balanced", n_jobs=-1
    )
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"],
    }
    models_and_grids.append(("RandomForest", rf, rf_grid))

    # XGBoost
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    xgb_grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "reg_lambda": [1.0, 10.0],
    }
    models_and_grids.append(("XGBoost", xgb, xgb_grid))

    # SVM (needs scaling)
    svm = make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True))
    svm_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["linear", "rbf"],
        "svc__gamma": ["scale", 0.01, 0.1, 1, 10],
    }
    models_and_grids.append(("SVM", svm, svm_grid))

    return models_and_grids
