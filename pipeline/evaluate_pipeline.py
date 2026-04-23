# ========================================
# EVALUATION MODULE
# ========================================
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, mean_squared_error, r2_score)


def evaluate_classification(X_test, y_test, run_ids):
    """Evaluate all classification models, log metrics to MLflow."""
    print("=" * 50)
    print("STEP 5: EVALUATE CLASSIFICATION MODELS")
    print("=" * 50)

    results = {}
    mlflow.set_experiment("Student Placement Prediction")

    for model_name, run_id in run_ids.items():
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_precision", prec)
            mlflow.log_metric("test_recall", rec)
            mlflow.log_metric("test_f1", f1)

        print(f"  {model_name} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        results[model_name] = acc

    best_name = max(results, key=results.get)
    best_score = results[best_name]
    print(f"\n  Best: {best_name} (Accuracy: {best_score:.4f})")
    print("")
    return best_score


def evaluate_regression(X_test, y_test, run_ids):
    """Evaluate all regression models, log metrics to MLflow."""
    print("=" * 50)
    print("STEP 4: EVALUATE REGRESSION MODELS")
    print("=" * 50)

    results_mse = {}
    mlflow.set_experiment("Student Salary Prediction")

    for model_name, run_id in run_ids.items():
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_r2", r2)

        print(f"  {model_name} | MSE: {mse:.4f} | R2: {r2:.4f}")
        results_mse[model_name] = r2

    best_name = max(results_mse, key=results_mse.get)
    best_score = results_mse[best_name]
    print(f"\n  Best: {best_name} (R2: {best_score:.4f})")
    print("")
    return best_score


if __name__ == "__main__":
    pass # For testing purposes only