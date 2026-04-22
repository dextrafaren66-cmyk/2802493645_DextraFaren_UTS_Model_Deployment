# ========================================
# TRAINING MODULE (sklearn.pipeline + MLflow)
# ========================================
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)

BASE_DIR = Path(__file__).parent.parent
ARTIFACT_DIR = BASE_DIR / "models"

# Column definitions
NUMCOLS = ['gender', 'branch', 'part_time_job', 'internet_access']
CATCOLS = ['family_income_level', 'city_tier', 'extracurricular_involvement']
ORDER = [
    ['Low', 'Medium', 'High'],
    ['Tier 3', 'Tier 2', 'Tier 1'],     # Ordinal encoding
    ['None', 'Low', 'Medium', 'High']
]


def build_preprocessor():
    """Build reusable ColumnTransformer for preprocessing."""
    return ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False), NUMCOLS),
            ('ordinal', OrdinalEncoder(categories=ORDER), CATCOLS)
        ],
        remainder='passthrough'
    )


def train_regression(X_train, y_train):
    """Train 3 regression models with MLflow tracking."""
    print("=" * 50)
    print("STEP 2: TRAINING REGRESSION MODELS")
    print("=" * 50)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest_Reg": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42),
        "GradientBoosting_Reg": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    }

    trained_pipelines = {}
    run_ids = {}

    mlflow.set_experiment("Student Salary Prediction")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            pipeline = Pipeline([
                ('preprocessing', build_preprocessor()),
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            trained_pipelines[model_name] = pipeline

            mlflow.log_params(model.get_params())

            artifact_path = ARTIFACT_DIR / f"regression_{model_name}.pkl"
            joblib.dump(pipeline, artifact_path)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            run_ids[model_name] = run.info.run_id
            print(f"  {model_name} trained and logged.")

    print("")
    return run_ids, trained_pipelines


def train_classification(X_train, y_train):
    """Train 3 classification models with MLflow tracking."""
    print("=" * 50)
    print("STEP 3: TRAINING CLASSIFICATION MODELS")
    print("=" * 50)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "LogisticRegression_Cls": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest_Cls": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "GradientBoosting_Cls": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    }

    trained_pipelines = {}
    run_ids = {}

    mlflow.set_experiment("Student Placement Prediction")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            pipeline = Pipeline([
                ('preprocessing', build_preprocessor()),
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            trained_pipelines[model_name] = pipeline

            mlflow.log_params(model.get_params())

            artifact_path = ARTIFACT_DIR / f"classification_{model_name}.pkl"
            joblib.dump(pipeline, artifact_path)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            run_ids[model_name] = run.info.run_id
            print(f"  {model_name} trained and logged.")

    print("")
    return run_ids, trained_pipelines


if __name__ == "__main__":
    print("Use run_pipeline.py to execute the full pipeline.")
