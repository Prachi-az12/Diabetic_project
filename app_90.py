import streamlit as st
import pickle
import numpy as np

# Load Model and Scaler
#model = pickle.load(open("model.pkl", "rb"))
#sc = pickle.load(open("sc.pkl", "rb"))
from joblib import load

model = load("model.joblib")


# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)



# Title
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's details to predict whether they are diabetic or not.")

st.sidebar.header("About")
st.sidebar.info("""
This application predicts diabetes using a Machine Learning model.
Fill in all patient details and click Predict.
""")

# Input Layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    smoking_history = st.selectbox(
        "Smoking History",
        ["No Info", "Current", "Former", "Never", "Ever", "Not Current"]
    )
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100)

# Encoding
gender = 1 if gender == "Male" else 0
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

smoking_dict = {
    "No Info": 0,
    "Current": 1,
    "Former": 2,
    "Never": 3,
    "Ever": 4,
    "Not Current": 5
}

smoking_history = smoking_dict[smoking_history]

# Prediction
if st.button("Predict Diabetes"):

    data = np.array([[gender,
                      age,
                      hypertension,
                      heart_disease,
                      smoking_history,
                      bmi,
                      hba1c,
                      blood_glucose]])

    #data = sc.transform(data)

    prediction = model.predict(data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is not likely to have Diabetes.")

    st.markdown("---")
    st.info("This prediction is based on a Machine Learning model and should not replace professional medical advice.")
