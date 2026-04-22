# ========================================
# PIPELINE ORCHESTRATOR
# ========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from data_ingestion import ingest_data
from train_pipeline import train_regression, train_classification
from evaluate_pipeline import evaluate_regression, evaluate_classification
from pathlib import Path

ACCURACY_THRESHOLD = 0.3
BASE_DIR = Path(__file__).parent.parent


def run_pipeline():
    """Execute full ML pipeline: ingest -> train -> evaluate."""

    print("\n" + "=" * 50)
    print("RUNNING FULL ML PIPELINE")
    print("=" * 50 + "\n")

    # Step 1: Data Ingestion
    df = ingest_data()

    # Preprocessing
    df = df.drop("Student_ID", axis=1)
    df['extracurricular_involvement'] = df['extracurricular_involvement'].fillna("None")

    X = df.drop(['placement_status', 'salary_lpa'], axis=1)
    y_reg = df['salary_lpa']
    y_cls = df['placement_status']

    # Encode classification target
    le = LabelEncoder()
    y_cls_encoded = le.fit_transform(y_cls)
    joblib.dump(le, BASE_DIR / "models" / "label_encoder.pkl")

    # Train-test split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, y_cls_encoded, test_size=0.2, random_state=42, stratify=y_cls_encoded
    )

    # Step 2: Train Regression
    reg_run_ids, reg_pipelines = train_regression(X_train_reg, y_train_reg)

    # Step 3: Train Classification
    cls_run_ids, cls_pipelines = train_classification(X_train_cls, y_train_cls)

    # Step 4: Evaluate Regression
    reg_score = evaluate_regression(X_test_reg, y_test_reg, reg_run_ids)

    # Step 5: Evaluate Classification
    cls_score = evaluate_classification(X_test_cls, y_test_cls, cls_run_ids)

    # Step 6: Approval Check
    print("=" * 50)
    print("PIPELINE RESULT")
    print("=" * 50)

    if (reg_score > ACCURACY_THRESHOLD) and (cls_score > ACCURACY_THRESHOLD):
        print("Models APPROVED for deployment")
    else:
        print("Models REJECTED - below threshold")

    print("")


if __name__ == "__main__":
    run_pipeline()
