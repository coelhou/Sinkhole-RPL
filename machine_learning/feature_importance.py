"""
Step 3: inspect which features drive each model's predictions.

Run after train_models.py:
    python feature_importance.py

Reads best_estimators.joblib / feature_names.joblib from ARTIFACTS_DIR and
writes feature_importance_tables.joblib (dict[name -> DataFrame]).

Notes on two small fixes made vs. the original notebook code:
  - top_features_linear() now takes the pipeline step name as an argument
    instead of hardcoding "logisticregression", so it also works correctly
    for the SVM pipeline (step name "svc"). Previously the SVM branch always
    hit the except-block and returned an empty DataFrame.
  - KNN has no coefficients or feature_importances_, so it no longer calls
    top_features_tree() on it (which would just fail silently); it prints an
    explicit "not applicable" note instead.
"""
import os
import joblib
import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR


def top_features_linear(pipe, feature_names, linear_step_name, top_k=15):
    try:
        lin = pipe.named_steps[linear_step_name]
        coefs = lin.coef_.ravel()
        idx = np.argsort(np.abs(coefs))[::-1][:top_k]
        return pd.DataFrame({
            "feature": np.array(feature_names)[idx],
            "coef": coefs[idx],
            "abs_coef": np.abs(coefs[idx]),
        })
    except Exception:
        return pd.DataFrame()


def top_features_tree(model, feature_names, top_k=15):
    try:
        imps = model.feature_importances_
        idx = np.argsort(imps)[::-1][:top_k]
        return pd.DataFrame({
            "feature": np.array(feature_names)[idx],
            "importance": imps[idx],
        })
    except Exception:
        return pd.DataFrame()


def main():
    best_estimators = joblib.load(os.path.join(ARTIFACTS_DIR, "best_estimators.joblib"))
    feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.joblib"))

    tables = {}

    if "LogisticRegression" in best_estimators:
        tf_lr = top_features_linear(
            best_estimators["LogisticRegression"], feature_names, "logisticregression"
        )
        print("\nTop features — Logistic Regression (|coef|):")
        print(tf_lr.to_string(index=False))
        tables["LogisticRegression"] = tf_lr

    if "KNN" in best_estimators:
        print("\nTop features — KNN: not applicable "
              "(KNN has no feature importances or coefficients).")

    if "RandomForest" in best_estimators:
        tf_rf = top_features_tree(best_estimators["RandomForest"], feature_names)
        print("\nTop features — Random Forest (importance):")
        print(tf_rf.to_string(index=False))
        tables["RandomForest"] = tf_rf

    if "XGBoost" in best_estimators:
        tf_xgb = top_features_tree(best_estimators["XGBoost"], feature_names)
        print("\nTop features — XGBoost (importance):")
        print(tf_xgb.to_string(index=False))
        tables["XGBoost"] = tf_xgb

    if "SVM" in best_estimators:
        svm_model = best_estimators["SVM"].named_steps.get("svc")
        if svm_model is not None and svm_model.kernel == "linear":
            tf_svm = top_features_linear(best_estimators["SVM"], feature_names, "svc")
            print("\nTop features — SVM (|coef|):")
            print(tf_svm.to_string(index=False))
            tables["SVM"] = tf_svm
        else:
            print("\nTop features — SVM: not applicable "
                  "(best kernel is not linear, no direct coefficients).")

    joblib.dump(tables, os.path.join(ARTIFACTS_DIR, "feature_importance_tables.joblib"))
    return tables


if __name__ == "__main__":
    main()
