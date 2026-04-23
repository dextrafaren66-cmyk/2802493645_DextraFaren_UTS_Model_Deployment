# ========================================
# DECOUPLED FRONTEND (Streamlit -> FastAPI)
# ========================================
import streamlit as st
import requests
import pandas as pd

# ========================================
# CONFIG
# ========================================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Student Predictor (Decoupled)",
    page_icon="🎓",
    layout="wide"
)

# ========================================
# SIDEBAR
# ========================================
st.sidebar.title("🎓 Student Predictor")
st.sidebar.markdown("**Decoupled (FastAPI + Streamlit)**")
st.sidebar.markdown("---")
task = st.sidebar.selectbox("Select Task", ["Classification", "Regression", "Both (Full)"])
st.sidebar.markdown("---")

# API health check
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("API Status: Connected ✅")
    else:
        st.sidebar.error("API Status: Error ❌")
except requests.ConnectionError:
    st.sidebar.error("API Status: Disconnected ❌")
    st.sidebar.info("Start backend: `uvicorn backend:app --port 8000`")

# ========================================
# MAIN HEADER
# ========================================
st.title("🎓 Student Placement & Salary Prediction")
st.markdown("**Decoupled Architecture:** Streamlit (Frontend) → FastAPI (Backend)")
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
# BUILD PAYLOAD
# ========================================
payload = {
    "gender": gender,
    "branch": branch,
    "cgpa": cgpa,
    "tenth_percentage": tenth_pct,
    "twelfth_percentage": twelfth_pct,
    "backlogs": backlogs,
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "projects_completed": projects,
    "internships_completed": internships,
    "coding_skill_rating": coding_skill,
    "communication_skill_rating": comm_skill,
    "aptitude_skill_rating": aptitude_skill,
    "hackathons_participated": hackathons,
    "certifications_count": certifications,
    "sleep_hours": sleep_hours,
    "stress_level": stress_level,
    "part_time_job": part_time,
    "family_income_level": family_income,
    "city_tier": city_tier,
    "internet_access": internet,
    "extracurricular_involvement": extracurricular
}

# ========================================
# PREDICTION VIA API
# ========================================
st.markdown("---")

if st.button("🔮 Predict via API", type="primary", use_container_width=True):

    try:
        col_r1, col_r2 = st.columns(2)

        if task == "Classification":
            response = requests.post(f"{API_URL}/predict/classification", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                with col_r1:
                    st.subheader("📊 Classification Result")
                    if result["placement_status"] == "Placed":
                        st.success(f"**Prediction: {result['placement_status']}** ✅")
                    else:
                        st.error(f"**Prediction: {result['placement_status']}** ❌")

                    if result["confidence"]:
                        st.write("**Confidence:**")
                        for cls_name, prob in result["confidence"].items():
                            st.progress(prob, text=f"{cls_name}: {prob:.2%}")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        elif task == "Regression":
            response = requests.post(f"{API_URL}/predict/regression", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                with col_r2:
                    st.subheader("💰 Regression Result")
                    st.metric("Predicted Salary", f"₹{result['predicted_salary_lpa']:.2f} LPA")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

        else:  # Both (Full)
            response = requests.post(f"{API_URL}/predict/full", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                with col_r1:
                    st.subheader("📊 Classification Result")
                    if result["placement_status"] == "Placed":
                        st.success(f"**Prediction: {result['placement_status']}** ✅")
                    else:
                        st.error(f"**Prediction: {result['placement_status']}** ❌")

                    if result["confidence"]:
                        st.write("**Confidence:**")
                        for cls_name, prob in result["confidence"].items():
                            st.progress(prob, text=f"{cls_name}: {prob:.2%}")

                with col_r2:
                    st.subheader("💰 Regression Result")
                    st.metric("Predicted Salary", f"₹{result['predicted_salary_lpa']:.2f} LPA")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.ConnectionError:
        st.error("Failed to connect to API. Make sure backend is running on port 8000.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

    st.markdown("---")
    st.subheader("📤 Request Payload (JSON)")
    st.json(payload)
