from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_mental_health_model.joblib"
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
NUMERIC_FEATURES = (
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
)
CATEGORICAL_FEATURES = (
    "Diet_Quality",
    "Physical_Activity",
    "Part_Time_Job",
    "Previous_Mental_Health_History",
)


NUMERIC_TRANSFORMS = {
    "Year_of_Study": (-3.5, 8.5, 2.62, 1.3145721737508367),
    "Sleep_Hours": (3.75, 9.75, 6.74884375, 1.100805205559066),
    "Study_Hours_Per_Week": (4.0, 36.0, 19.9763125, 5.752581607621377),
    "GPA": (1.64, 3.48, 2.5643375, 0.3414660467656338),
    "Social_Media_Hours": (3.2, 8.8, 6.0315625, 1.1034933432484992),
    "Screen_Time_Hours": (3.95, 13.15, 8.57734375, 1.6532290772412446),
    "Financial_Stress": (0.4, 11.6, 5.970125, 1.8632283768703717),
    "Family_Support": (1.0, 12.2, 6.528, 1.8226844213960902),
    "Attendance_Rate": (65.9, 104.3, 85.073625, 7.109740983986337),
}
CATEGORY_ENCODINGS = {
    "Diet_Quality": {"Poor": 0.0, "Average": 1.0, "Good": 2.0},
    "Physical_Activity": {"Low": 0.0, "Moderate": 1.0, "High": 2.0},
    "Part_Time_Job": {"No": 0.0, "Yes": 1.0},
    "Previous_Mental_Health_History": {"No": 0.0, "Yes": 1.0},
}


def transform_input(values):
    encoded = []
    for feature in NUMERIC_FEATURES:
        lower, upper, mean, scale = NUMERIC_TRANSFORMS[feature]
        capped = np.clip(float(values[feature]), lower, upper)
        encoded.append((capped - mean) / scale)

    for feature in CATEGORICAL_FEATURES:
        encoded.append(CATEGORY_ENCODINGS[feature][values[feature]])

    return pd.DataFrame([encoded], columns=[str(index) for index in range(13)])


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    if getattr(model, "n_features_in_", None) != 13:
        raise ValueError("The saved model does not accept the expected 13 input features.")
    if set(int(label) for label in model.classes_) != set(RISK_LABELS):
        raise ValueError("The saved model classes do not match the expected risk labels.")
    return model


st.set_page_config(page_title="Student Mental Health Risk", layout="centered")
st.title("Student Mental Health Risk")
st.write(
    "Enter survey responses to see the model's predicted risk category. "
    "Age and gender are not used by this model."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.stop()

with st.form("student_survey"):
    st.subheader("Academic and daily habits")
    academic_col, sleep_col, study_col = st.columns(3)

    with academic_col:
        year_of_study = st.selectbox("Year of study", [1, 2, 3, 4, 5])
        gpa = st.number_input("GPA", 1.5, 4.0, 3.0, step=0.1)
    with sleep_col:
        sleep_hours = st.number_input("Sleep per night (hours)", 0.0, 24.0, 7.0, step=0.5)
        screen_time = st.number_input("Screen time per day (hours)", 0.0, 24.0, 7.0, step=0.5)
    with study_col:
        study_hours = st.number_input(
            "Study per week (hours)", 0.0, 168.0, 20.0, step=1.0
        )
        social_media_hours = st.number_input(
            "Social media per day (hours)", 0.0, 24.0, 4.0, step=0.5
        )

    st.subheader("Wellbeing and support")
    activity_col, stress_col, support_col = st.columns(3)

    with activity_col:
        physical_activity = st.selectbox(
            "Physical activity", ["Low", "Moderate", "High"]
        )
        diet_quality = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
    with stress_col:
        financial_stress = st.slider("Financial stress (1–10)", 1.0, 10.0, 5.0, step=0.5)
        attendance_rate = st.slider("Class attendance (%)", 0.0, 100.0, 85.0, step=1.0)
    with support_col:
        family_support = st.slider("Family support (1–10)", 1.0, 10.0, 6.0, step=0.5)
        part_time_job = st.selectbox("Part-time job", ["No", "Yes"])

    previous_history = st.selectbox(
        "Previous mental health history", ["No", "Yes"]
    )
    submitted = st.form_submit_button("Predict risk level", type="primary")

if submitted:
    responses = {
        "Year_of_Study": year_of_study,
        "Sleep_Hours": sleep_hours,
        "Study_Hours_Per_Week": study_hours,
        "GPA": gpa,
        "Social_Media_Hours": social_media_hours,
        "Screen_Time_Hours": screen_time,
        "Physical_Activity": physical_activity,
        "Financial_Stress": financial_stress,
        "Family_Support": family_support,
        "Attendance_Rate": attendance_rate,
        "Part_Time_Job": part_time_job,
        "Diet_Quality": diet_quality,
        "Previous_Mental_Health_History": previous_history,
    }

    features = transform_input(responses)
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    probability_by_label = {
        RISK_LABELS[int(label)]: float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    st.subheader("Prediction")
    st.metric("Predicted risk category", RISK_LABELS[prediction])
    st.caption("Model scores by category (not clinical probabilities)")
    for label in ("Low", "Medium", "High"):
        st.write(f"**{label}:** {probability_by_label[label]:.1%}")
        st.progress(probability_by_label[label])

    st.caption(
        "If you are concerned about your wellbeing, consider speaking with a qualified "
        "mental health professional or someone you trust."
    )