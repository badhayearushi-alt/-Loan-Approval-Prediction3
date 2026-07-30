import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and scaler
model_LR = joblib.load("logistic_model.pkl")
model_knn = joblib.load("knn_model.pkl")
model_NB = joblib.load("nb_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Loan Prediction App", layout="centered")

st.title("🏦 Loan Prediction System")

model = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ("Logistic Regression", "KNN", "Naive Bayes")
)

st.header("Enter Applicant Details")

Gender = st.number_input("Gender (0=Female, 1=Male)", min_value=0)
Married = st.number_input("Married (0=No, 1=Yes)", min_value=0)
Dependents = st.number_input("Dependents", min_value=0)
Education = st.number_input("Education (0=Not Graduate, 1=Graduate)", min_value=0)
Self_Employed = st.number_input("Self Employed (0=No, 1=Yes)", min_value=0)
ApplicantIncome = st.number_input("Applicant Income", min_value=0.0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
LoanAmount = st.number_input("Loan Amount", min_value=0.0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0.0)
Credit_History = st.number_input("Credit History (0 or 1)", min_value=0)
Property_Area = st.number_input("Property Area (0=Rural,1=Semiurban,2=Urban)", min_value=0)

input_data = np.array([[Gender, Married, Dependents, Education, Self_Employed,
                        ApplicantIncome, CoapplicantIncome, LoanAmount,
                        Loan_Amount_Term, Credit_History, Property_Area]])

input_df = pd.DataFrame(input_data, columns=columns)

input_scaled = scaler.transform(input_df)

if st.button("Predict"):

    if model == "Logistic Regression":
        prediction = model_LR.predict(input_scaled)

    elif model == "KNN":
        prediction = model_knn.predict(input_scaled)

    elif model == "Naive Bayes":
        prediction = model_NB.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
