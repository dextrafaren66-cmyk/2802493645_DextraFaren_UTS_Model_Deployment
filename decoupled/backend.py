# ========================================
# DECOUPLED BACKEND (FastAPI)
# ========================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ========================================
# LOAD MODELS
# ========================================
MODEL_DIR = Path(__file__).parent.parent / "models"

cls_pipeline = joblib.load(MODEL_DIR / "classification_pipeline.pkl")
reg_pipeline = joblib.load(MODEL_DIR / "regression_pipeline.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

# ========================================
# FASTAPI APP
# ========================================
app = FastAPI(
    title="Student Placement & Salary Prediction API",
    description="API Server for serving classification and regression predictions.",
    version="1.0.0"
)


# ========================================
# SCHEMA
# ========================================
class StudentInput(BaseModel):
    gender: str = Field(..., example="Male")
    branch: str = Field(..., example="CSE")
    cgpa: float = Field(..., ge=0, le=10, example=8.5)
    tenth_percentage: float = Field(..., ge=0, le=100, example=75.0)
    twelfth_percentage: float = Field(..., ge=0, le=100, example=72.0)
    backlogs: int = Field(..., ge=0, example=0)
    study_hours_per_day: float = Field(..., ge=0, example=5.0)
    attendance_percentage: float = Field(..., ge=0, le=100, example=80.0)
    projects_completed: int = Field(..., ge=0, example=4)
    internships_completed: int = Field(..., ge=0, example=2)
    coding_skill_rating: int = Field(..., ge=1, le=5, example=4)
    communication_skill_rating: int = Field(..., ge=1, le=5, example=3)
    aptitude_skill_rating: int = Field(..., ge=1, le=5, example=4)
    hackathons_participated: int = Field(..., ge=0, example=3)
    certifications_count: int = Field(..., ge=0, example=3)
    sleep_hours: float = Field(..., ge=0, example=7.0)
    stress_level: int = Field(..., ge=1, le=10, example=5)
    part_time_job: str = Field(..., example="No")
    family_income_level: str = Field(..., example="Medium")
    city_tier: str = Field(..., example="Tier 1")
    internet_access: str = Field(..., example="Yes")
    extracurricular_involvement: str = Field(..., example="Medium")


class ClassificationResponse(BaseModel):
    placement_status: str
    confidence: dict


class RegressionResponse(BaseModel):
    predicted_salary_lpa: float


class FullPredictionResponse(BaseModel):
    placement_status: str
    confidence: dict
    predicted_salary_lpa: float


# ========================================
# HELPER
# ========================================
def input_to_dataframe(data: StudentInput) -> pd.DataFrame:
    return pd.DataFrame([data.model_dump()])


# ========================================
# ENDPOINTS
# ========================================
@app.get("/")
def root():
    return {"message": "Student Prediction API is running.", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict/classification", response_model=ClassificationResponse)
def predict_classification(data: StudentInput):
    """Predict placement status (Placed / Not Placed)."""
    try:
        df = input_to_dataframe(data)
        pred = cls_pipeline.predict(df)
        label = label_encoder.inverse_transform(pred)[0]

        confidence = {}
        if hasattr(cls_pipeline.named_steps['model'], 'predict_proba'):
            proba = cls_pipeline.predict_proba(df)[0]
            for i, cls_name in enumerate(label_encoder.classes_):
                confidence[cls_name] = round(float(proba[i]), 4)

        return ClassificationResponse(placement_status=label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(data: StudentInput):
    """Predict expected salary in LPA."""
    try:
        df = input_to_dataframe(data)
        pred = reg_pipeline.predict(df)
        salary = max(0.0, round(float(pred[0]), 2))

        return RegressionResponse(predicted_salary_lpa=salary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full", response_model=FullPredictionResponse)
def predict_full(data: StudentInput):
    """Predict both placement status and salary."""
    try:
        df = input_to_dataframe(data)

        cls_pred = cls_pipeline.predict(df)
        label = label_encoder.inverse_transform(cls_pred)[0]

        confidence = {}
        if hasattr(cls_pipeline.named_steps['model'], 'predict_proba'):
            proba = cls_pipeline.predict_proba(df)[0]
            for i, cls_name in enumerate(label_encoder.classes_):
                confidence[cls_name] = round(float(proba[i]), 4)

        reg_pred = reg_pipeline.predict(df)
        salary = max(0.0, round(float(reg_pred[0]), 2))

        return FullPredictionResponse(
            placement_status=label,
            confidence=confidence,
            predicted_salary_lpa=salary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
