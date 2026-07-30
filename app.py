import streamlit as st
import numpy as np

# ML Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Title
st.title("🏦 Loan Prediction System")

# Sidebar
st.sidebar.title("Model Selection")

model_option = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ["Logistic Regression", "KNN", "Naive Bayes", "Linear Regression"]
)

st.subheader("Enter Applicant Details")

# Input Fields
gender = st.number_input("Gender (0=Female, 1=Male)", min_value=0, max_value=1)
married = st.number_input("Married (0=No, 1=Yes)", min_value=0, max_value=1)
dependents = st.number_input("Dependents", min_value=0, max_value=5)
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.number_input("Credit History (0 or 1)", min_value=0, max_value=1)

# Dummy Dataset
X = np.array([
    [1,1,0,5000,200,1],
    [0,0,1,3000,100,1],
    [1,1,2,4000,150,0],
    [0,1,0,6000,250,1]
])

y = np.array([1,0,0,1])

# Model Selection
if model_option == "Logistic Regression":
    model = LogisticRegression()

elif model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=3)

elif model_option == "Naive Bayes":
    model = GaussianNB()

elif model_option == "Linear Regression":
    model = LinearRegression()

# Train Model
model.fit(X, y)

# Predict Button
if st.button("Predict Loan Status"):

    input_data = np.array([[gender, married, dependents, income, loan_amount, credit_history]])

    prediction = model.predict(input_data)

    # Convert Linear Regression output
    if model_option == "Linear Regression":
        prediction = [1 if p > 0.5 else 0 for p in prediction]

    # Output
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
