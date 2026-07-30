import streamlit as st
import numpy as np

# ML Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Title
st.title("🏦 Loan Prediction System")

# Sidebar
model_option = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ["Logistic Regression", "KNN", "Naive Bayes", "Linear Regression"]
)

st.subheader("Enter Applicant Details")

# Inputs
gender = st.number_input("Gender (0=Female, 1=Male)", 0, 1)
married = st.number_input("Married (0=No, 1=Yes)", 0, 1)
dependents = st.number_input("Dependents", 0, 5)
education = st.number_input("Education (0=Not Graduate, 1=Graduate)", 0, 1)
self_employed = st.number_input("Self Employed (0=No, 1=Yes)", 0, 1)
app_income = st.number_input("Applicant Income", 0.00)
co_income = st.number_input("Coapplicant Income", 0.00)
loan_amount = st.number_input("Loan Amount", 0.00)
loan_term = st.number_input("Loan Amount Term", 0.00)
credit_history = st.number_input("Credit History (0 or 1)", 0, 1)
property_area = st.number_input("Property Area (0=Rural,1=Semiurban,2=Urban)", 0, 2)

# Dataset
X = np.array([
    [1,1,0,1,0,5000,0,200,360,1,2],
    [0,0,1,1,1,3000,1500,100,360,1,1],
    [1,1,2,0,0,4000,1000,150,360,0,0],
    [0,1,0,1,1,6000,0,250,360,1,2]
])

y = np.array([1,0,0,1])

# Model selection
if model_option == "Logistic Regression":
    model = LogisticRegression()
elif model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=3)
elif model_option == "Naive Bayes":
    model = GaussianNB()
elif model_option == "Linear Regression":
    model = LinearRegression()

# Train
model.fit(X, y)

# Prediction
if st.button("Predict"):

    input_data = np.array([[gender, married, dependents, education, self_employed,
                            app_income, co_income, loan_amount, loan_term,
                            credit_history, property_area]])

    prediction = model.predict(input_data)

    if model_option == "Linear Regression":
        prediction = [1 if p > 0.5 else 0 for p in prediction]

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

# Footer
st.markdown("Made with ❤️ using Streamlit")