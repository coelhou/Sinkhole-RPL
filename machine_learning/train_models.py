"""
Step 1: train + tune every model with GridSearchCV and cache the results.

Run this first:
    python train_models.py

It writes to ARTIFACTS_DIR (default "artifacts/"):
    - best_estimators.joblib   -> dict[name -> fitted best estimator]
    - cv_results.joblib        -> dict[name -> {"best_score", "best_params"}]
    - test_data.joblib         -> (X_test, y_test), held out for evaluate_models.py
    - feature_names.joblib     -> list of feature column names

evaluate_models.py and feature_importance.py both read these artifacts, so
this script must be run at least once before them.
"""
import os
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from config import ARTIFACTS_DIR, N_SPLITS, RANDOM_STATE, SCORING, REFIT_METRIC
from data_utils import load_data, split_data
from model_definitions import get_models_and_grids


def train_all_models(X_train, y_train, scale_pos_weight):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models_and_grids = get_models_and_grids(scale_pos_weight, random_state=RANDOM_STATE)

    best_estimators = {}
    cv_results = {}

    for name, est, grid in models_and_grids:
        print(f"Tuning {name}...")
        gs = GridSearchCV(
            estimator=est, param_grid=grid, scoring=SCORING, refit=REFIT_METRIC,
            cv=cv, n_jobs=-1, verbose=0, return_train_score=False,
        )
        gs.fit(X_train, y_train)

        best_estimators[name] = gs.best_estimator_
        cv_results[name] = {"best_score": gs.best_score_, "best_params": gs.best_params_}
        print(f"  best CV ROC-AUC: {gs.best_score_:.4f} | params: {gs.best_params_}")

    return best_estimators, cv_results


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test, scale_pos_weight = split_data(X, y)

    best_estimators, cv_results = train_all_models(X_train, y_train, scale_pos_weight)

    joblib.dump(best_estimators, os.path.join(ARTIFACTS_DIR, "best_estimators.joblib"))
    joblib.dump(cv_results, os.path.join(ARTIFACTS_DIR, "cv_results.joblib"))
    joblib.dump((X_test, y_test), os.path.join(ARTIFACTS_DIR, "test_data.joblib"))
    joblib.dump(list(X.columns), os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))

    print(f"\nSaved trained models and test data to '{ARTIFACTS_DIR}/'.")


if __name__ == "__main__":
    main()
