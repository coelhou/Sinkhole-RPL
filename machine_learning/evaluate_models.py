"""
Step 2: evaluate every tuned model on the held-out test set.

Run after train_models.py:
    python evaluate_models.py

Reads best_estimators.joblib / cv_results.joblib / test_data.joblib from
ARTIFACTS_DIR, and writes:
    - model_comparison.csv        -> sortable results table
    - roc_curves.png              -> combined ROC plot for all models
    - evaluation_results.joblib   -> full per-model metrics (incl. confusion
                                      matrices and classification reports)
"""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report,
)

from config import ARTIFACTS_DIR
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (7, 5)  # one chart per figure; no specific colors/styles set


def evaluate_model(name, estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        y_prob = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X_test)
        y_prob = (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        y_prob = y_pred.astype(float)

    return {
        "model": name,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4),
        "roc_curve": roc_curve(y_test, y_prob),
    }


def plot_roc_curves(fpr_dict, tpr_dict, auc_dict, save_path=None):
    plt.figure()
    for name in fpr_dict:
        plt.plot(fpr_dict[name], tpr_dict[name], label=f"{name} (AUC={auc_dict[name]:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def main():
    best_estimators = joblib.load(os.path.join(ARTIFACTS_DIR, "best_estimators.joblib"))
    cv_results = joblib.load(os.path.join(ARTIFACTS_DIR, "cv_results.joblib"))
    X_test, y_test = joblib.load(os.path.join(ARTIFACTS_DIR, "test_data.joblib"))

    results = []
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for name, est in best_estimators.items():
        m = evaluate_model(name, est, X_test, y_test)
        m["cv_best_roc_auc"] = cv_results[name]["best_score"]
        m["best_params"] = cv_results[name]["best_params"]
        results.append(m)

        fpr, tpr, _ = m["roc_curve"]
        fpr_dict[name], tpr_dict[name], auc_dict[name] = fpr, tpr, m["test_roc_auc"]

    res_df = pd.DataFrame([{
        "model": r["model"],
        "cv_best_roc_auc": r["cv_best_roc_auc"],
        "test_accuracy": r["test_accuracy"],
        "test_precision": r["test_precision"],
        "test_recall": r["test_recall"],
        "test_f1": r["test_f1"],
        "test_roc_auc": r["test_roc_auc"],
        "best_params": r["best_params"],
    } for r in results]).sort_values("test_roc_auc", ascending=False)

    print("\n=== Model comparison ===")
    print(res_df.to_string(index=False))

    for r in results:
        print("\n== Classification report:", r["model"], "==")
        print(r["classification_report"])

    for r in results:
        print(f"\n== Confusion matrix: {r['model']} ==")
        print(r["confusion_matrix"])

    plot_roc_curves(
        fpr_dict, tpr_dict, auc_dict,
        save_path=os.path.join(ARTIFACTS_DIR, "roc_curves.png"),
    )

    res_df.to_csv(os.path.join(ARTIFACTS_DIR, "model_comparison.csv"), index=False)
    joblib.dump(results, os.path.join(ARTIFACTS_DIR, "evaluation_results.joblib"))
    print(f"\nSaved evaluation results to '{ARTIFACTS_DIR}/'.")

    return results, res_df


if __name__ == "__main__":
    main()
