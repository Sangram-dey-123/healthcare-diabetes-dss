
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Healthcare Diabetes DSS",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("healthcare_diabetes_catboost.pkl")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏥 Healthcare Diabetes Decision Support System")

st.markdown(
    """
    **Machine Learning-Based Diabetes Risk Prediction and Analytics**

    This prototype uses a CatBoost machine-learning model to estimate
    diabetes risk from selected healthcare indicators.
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR — PATIENT INPUTS
# --------------------------------------------------

st.sidebar.header("Patient Healthcare Inputs")

pregnancies = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.sidebar.number_input(
    "Glucose",
    min_value=0.0,
    max_value=250.0,
    value=120.0
)

blood_pressure = st.sidebar.number_input(
    "Blood Pressure",
    min_value=0.0,
    max_value=150.0,
    value=70.0
)

skin_thickness = st.sidebar.number_input(
    "Skin Thickness",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

insulin = st.sidebar.number_input(
    "Insulin",
    min_value=0.0,
    max_value=900.0,
    value=80.0
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

predict_button = st.sidebar.button(
    "🔍 Predict Diabetes Risk"
)

# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.subheader("📊 Healthcare DSS Prediction")

if predict_button:

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Prediction probability
    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= 0.50)

    # DSS risk category
    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Diabetes Probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        if prediction == 1:
            st.metric(
                "Model Prediction",
                "Diabetes Risk"
            )
        else:
            st.metric(
                "Model Prediction",
                "No Diabetes"
            )

    with col3:
        st.metric(
            "DSS Risk Category",
            risk
        )

    st.divider()

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

    st.subheader("🧠 Decision Support Interpretation")

    if risk == "Low Risk":
        st.info(
            "The model estimates a relatively low diabetes probability "
            "for the provided input values."
        )

    elif risk == "Moderate Risk":
        st.warning(
            "The model estimates a moderate diabetes probability. "
            "The result can be considered for further assessment."
        )

    else:
        st.error(
            "The model estimates a high diabetes probability. "
            "Further clinical assessment may be appropriate."
        )

    st.subheader("📋 Input Summary")

    display_data = input_data.T
    display_data.columns = ["Value"]

    st.dataframe(
        display_data,
        use_container_width=True
    )

else:

    st.info(
        "Enter the healthcare indicators in the sidebar "
        "and click **Predict Diabetes Risk**."
    )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("📈 Model Performance")

performance = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        "88.96%",
        "83.64%",
        "85.19%",
        "84.40%",
        "0.9524"
    ]
})

st.dataframe(
    performance,
    hide_index=True,
    use_container_width=True
)

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.caption(
    "Academic prototype only. This system is intended for "
    "decision-support and educational purposes and is not a "
    "medical diagnosis or a substitute for professional clinical assessment."
)
