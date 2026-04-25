import streamlit as st
import pandas as pd
import joblib

from src.preprocess import preprocess_data

# load model, scaler, and features
model = joblib.load("model/loan_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/features.pkl")

st.title("Loan Approval Prediction")

# user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

app_income = st.number_input("Applicant Income")
coapp_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
credit_history = st.selectbox("Credit History", [1.0, 0.0])

property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# prediction
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': app_income,
        'CoapplicantIncome': coapp_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])

    # preprocess
    processed = preprocess_data(input_data)

    # ✅ align columns with training
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    # scale
    processed = scaler.transform(processed)

    # predict
    prediction = model.predict(processed)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")