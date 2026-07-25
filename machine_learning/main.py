"""
Convenience entry point: runs the whole pipeline end-to-end.

    python main.py

Equivalent to running train_models.py, then evaluate_models.py, then
feature_importance.py in order. Each step also works standalone as long as
train_models.py has been run at least once (its artifacts are cached to
disk under ARTIFACTS_DIR).
"""
import train_models
import evaluate_models
import feature_importance


def main():
    print("=== [1/3] Training models ===")
    train_models.main()

    print("\n=== [2/3] Evaluating models ===")
    evaluate_models.main()

    print("\n=== [3/3] Feature importance ===")
    feature_importance.main()


if __name__ == "__main__":
    main()
