import streamlit as st
import pandas as pd

from fuzzy_logic import (
    predict_mamdani,
    predict_sugeno,
    classify_risk,
    class_label,
    risk_category
)


st.set_page_config(
    page_title="Fuzzy Logic Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)


st.title("Fuzzy Logic-Based Diabetes Risk Prediction")

st.write(
    """
    This application uses **Mamdani** and **Sugeno fuzzy inference**
    to estimate diabetes risk from selected health indicators.
    The fuzzy inference pipeline is implemented from scratch without a fuzzy-logic library.
    """
)

st.warning(
    "Academic demonstration only. This application is not a medical diagnosis or clinical decision-support tool."
)


st.subheader("Health Indicators")

bmi = st.slider(
    "BMI",
    min_value=12.0,
    max_value=98.0,
    value=26.0,
    step=1.0
)

age_mapping = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80 or older"
}

age = st.slider(
    "Age Category",
    min_value=1,
    max_value=13,
    value=5,
    step=1,
    help="This follows the BRFSS age-category encoding and is not a person's exact age."
)

st.caption(f"Selected age category: {age} = {age_mapping[age]} years")

genhlth = st.slider(
    "General Health",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    help="1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor"
)

highbp = st.selectbox(
    "High Blood Pressure",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

highchol = st.selectbox(
    "High Cholesterol",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

physactivity = st.selectbox(
    "Physical Activity",
    options=[0, 1],
    format_func=lambda x: "Inactive" if x == 0 else "Active"
)


input_data = pd.Series({
    "BMI": bmi,
    "Age": age,
    "GenHlth": genhlth,
    "HighBP": highbp,
    "HighChol": highchol,
    "PhysActivity": physactivity
})


st.subheader("Input Summary")

input_df = pd.DataFrame([input_data])
st.dataframe(input_df, use_container_width=True)


if st.button("Predict Diabetes Risk"):
    mamdani_score = predict_mamdani(input_data)
    sugeno_score = predict_sugeno(input_data)

    mamdani_pred = classify_risk(mamdani_score)
    sugeno_pred = classify_risk(sugeno_score)

    result_df = pd.DataFrame({
        "Method": ["Mamdani", "Sugeno"],
        "Risk Score": [round(mamdani_score, 2), round(sugeno_score, 2)],
        "Risk Category": [
            risk_category(mamdani_score),
            risk_category(sugeno_score)
        ],
        "Predicted Class": [mamdani_pred, sugeno_pred],
        "Predicted Label": [
            class_label(mamdani_pred),
            class_label(sugeno_pred)
        ]
    })

    st.subheader("Prediction Results")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("Interpretation")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Mamdani Risk Score",
            value=f"{mamdani_score:.2f}"
        )
        st.write(f"Kategori: **{risk_category(mamdani_score)}**")
        st.write(f"Prediksi: **{class_label(mamdani_pred)}**")

    with col2:
        st.metric(
            label="Sugeno Risk Score",
            value=f"{sugeno_score:.2f}"
        )
        st.write(f"Kategori: **{risk_category(sugeno_score)}**")
        st.write(f"Prediksi: **{class_label(sugeno_pred)}**")

    st.info(
        """
        The classification threshold is 50.
        A risk score greater than or equal to 50 is classified as Diabetes;
        a lower score is classified as Non-Diabetes.
        """
    )
