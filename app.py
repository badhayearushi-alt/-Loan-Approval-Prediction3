import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Loan Approval Prediction",
        page_icon="🏦",
            layout="centered"
            )

            st.title("🏦 Loan Approval Prediction")
            st.write("Fill in the details below to predict loan approval.")

            # Load Model
            if not os.path.exists("loan_model.pkl"):
                st.error("❌ loan_model.pkl not found! Run train_model.py first.")
                    st.stop()

                    model = joblib.load("loan_model.pkl")

                    # Input Fields
                    gender = st.selectbox("Gender", ["Male", "Female"])

                    married = st.selectbox("Married", ["Yes", "No"])

                    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

                    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

                    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

                    applicant_income = st.number_input(
                        "Applicant Income",
                            min_value=0,
                                value=5000
                                )

                                coapplicant_income = st.number_input(
                                    "Coapplicant Income",
                                        min_value=0,
                                            value=0
                                            )

                                            loan_amount = st.number_input(
                                                "Loan Amount",
                                                    min_value=0,
                                                        value=120
                                                        )

                                                        loan_amount_term = st.number_input(
                                                            "Loan Amount Term",
                                                                min_value=12,
                                                                    value=360
                                                                    )

                                                                    credit_history = st.selectbox(
                                                                        "Credit History",
                                                                            [1, 0]
                                                                            )

                                                                            property_area = st.selectbox(
                                                                                "Property Area",
                                                                                    ["Urban", "Semiurban", "Rural"]
                                                                                    )

                                                                                    # Encode Values
                                                                                    gender = 1 if gender == "Male" else 0
                                                                                    married = 1 if married == "Yes" else 0
                                                                                    education = 1 if education == "Graduate" else 0
                                                                                    self_employed = 1 if self_employed == "Yes" else 0

                                                                                    if dependents == "3+":
                                                                                        dependents = 3
                                                                                        else:
                                                                                            dependents = int(dependents)

                                                                                            if property_area == "Urban":
                                                                                                property_area = 2
                                                                                                elif property_area == "Semiurban":
                                                                                                    property_area = 1
                                                                                                    else:
                                                                                                        property_area = 0

                                                                                                        # Create Input DataFrame
                                                                                                        input_data = pd.DataFrame({
                                                                                                            "Gender": [gender],
                                                                                                                "Married": [married],
                                                                                                                    "Dependents": [dependents],
                                                                                                                        "Education": [education],
                                                                                                                            "Self_Employed": [self_employed],
                                                                                                                                "ApplicantIncome": [applicant_income],
                                                                                                                                    "CoapplicantIncome": [coapplicant_income],
                                                                                                                                        "LoanAmount": [loan_amount],
                                                                                                                                            "Loan_Amount_Term": [loan_amount_term],
                                                                                                                                                "Credit_History": [credit_history],
                                                                                                                                                    "Property_Area": [property_area]
                                                                                                                                                    })

                                                                                                                                                    # Prediction
                                                                                                                                                    if st.button("Predict Loan Approval"):
                                                                                                                                                        prediction = model.predict(input_data)

                                                                                                                                                            if prediction[0] == 1:
                                                                                                                                                                    st.success("✅ Loan Approved")
                                                                                                                                                                            st.balloons()
                                                                                                                                                                                else:
                                                                                                                                                                                        st.error("❌ Loan Not Approved")