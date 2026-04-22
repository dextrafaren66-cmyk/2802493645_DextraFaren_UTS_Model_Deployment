# ========================================
# MONOLITHIC DEPLOYMENT (Streamlit)
# ========================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Student Placement & Salary Predictor",
    page_icon="🎓",
    layout="wide"
)

# ========================================
# LOAD MODELS
# ========================================
MODEL_DIR = Path(__file__).parent.parent / "models"


@st.cache_resource
def load_models():
    cls_pipeline = joblib.load(MODEL_DIR / "classification_pipeline.pkl")
    reg_pipeline = joblib.load(MODEL_DIR / "regression_pipeline.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return cls_pipeline, reg_pipeline, label_encoder


cls_pipeline, reg_pipeline, label_encoder = load_models()

# ========================================
# SIDEBAR
# ========================================
st.sidebar.title("🎓 Student Predictor")
st.sidebar.markdown("---")
task = st.sidebar.selectbox("Select Task", ["Classification", "Regression", "Both"])
st.sidebar.markdown("---")
st.sidebar.info("Model: sklearn Pipeline (.pkl)\nDeployment: Streamlit Monolithic")

# ========================================
# MAIN HEADER
# ========================================
st.title("🎓 Student Placement & Salary Prediction")
st.markdown("Predict placement status and expected salary based on student profile.")
st.markdown("---")

# ========================================
# INPUT FORM
# ========================================
st.subheader("📋 Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "ME", "CE"])
    cgpa = st.slider("CGPA", 4.0, 10.0, 7.5, 0.01)
    tenth_pct = st.slider("10th Percentage", 40.0, 100.0, 70.0, 0.1)
    twelfth_pct = st.slider("12th Percentage", 40.0, 100.0, 70.0, 0.1)
    backlogs = st.number_input("Backlogs", 0, 10, 0)
    study_hours = st.slider("Study Hours/Day", 0.0, 12.0, 4.0, 0.1)

with col2:
    attendance = st.slider("Attendance %", 40.0, 100.0, 75.0, 0.1)
    projects = st.number_input("Projects Completed", 0, 20, 3)
    internships = st.number_input("Internships Completed", 0, 10, 1)
    coding_skill = st.slider("Coding Skill (1-5)", 1, 5, 3)
    comm_skill = st.slider("Communication Skill (1-5)", 1, 5, 3)
    aptitude_skill = st.slider("Aptitude Skill (1-5)", 1, 5, 3)
    hackathons = st.number_input("Hackathons Participated", 0, 20, 2)

with col3:
    certifications = st.number_input("Certifications", 0, 20, 2)
    sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.1)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    part_time = st.selectbox("Part-Time Job", ["Yes", "No"])
    family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet = st.selectbox("Internet Access", ["Yes", "No"])
    extracurricular = st.selectbox("Extracurricular Involvement", ["None", "Low", "Medium", "High"])

# ========================================
# BUILD INPUT DATAFRAME
# ========================================
input_data = pd.DataFrame([{
    'gender': gender,
    'branch': branch,
    'cgpa': cgpa,
    'tenth_percentage': tenth_pct,
    'twelfth_percentage': twelfth_pct,
    'backlogs': backlogs,
    'study_hours_per_day': study_hours,
    'attendance_percentage': attendance,
    'projects_completed': projects,
    'internships_completed': internships,
    'coding_skill_rating': coding_skill,
    'communication_skill_rating': comm_skill,
    'aptitude_skill_rating': aptitude_skill,
    'hackathons_participated': hackathons,
    'certifications_count': certifications,
    'sleep_hours': sleep_hours,
    'stress_level': stress_level,
    'part_time_job': part_time,
    'family_income_level': family_income,
    'city_tier': city_tier,
    'internet_access': internet,
    'extracurricular_involvement': extracurricular
}])

# ========================================
# PREDICTION
# ========================================
st.markdown("---")

if st.button("🔮 Predict", type="primary", use_container_width=True):
    col_r1, col_r2 = st.columns(2)

    if task in ["Classification", "Both"]:
        with col_r1:
            st.subheader("📊 Classification Result")
            cls_pred = cls_pipeline.predict(input_data)
            cls_label = label_encoder.inverse_transform(cls_pred)[0]

            if cls_label == "Placed":
                st.success(f"**Prediction: {cls_label}** ✅")
            else:
                st.error(f"**Prediction: {cls_label}** ❌")

            if hasattr(cls_pipeline.named_steps['model'], 'predict_proba'):
                proba = cls_pipeline.predict_proba(input_data)
                st.write("**Confidence:**")
                for i, cls_name in enumerate(label_encoder.classes_):
                    st.progress(float(proba[0][i]), text=f"{cls_name}: {proba[0][i]:.2%}")

    if task in ["Regression", "Both"]:
        with col_r2:
            st.subheader("💰 Regression Result")
            reg_pred = reg_pipeline.predict(input_data)
            salary = max(0, reg_pred[0])
            st.metric("Predicted Salary", f"₹{salary:.2f} LPA")

    st.markdown("---")
    st.subheader("📄 Input Data Summary")
    st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)
