import streamlit as st
import pandas as pd
import joblib

df=pd.read_csv("credit_risk_dataset.csv")
model=joblib.load("svm_CreditRisk_model.pkl")
scaler=joblib.load("scaler_CreditRisk.pkl")
expected_columns=joblib.load("feature_columns_CreditRisk.pkl")

st.title("Credit Risk Prediction 📈")
st.markdown("Provide the following details to predict the credit risk.")

#collect user input
person_age=st.slider("Person's Age", min_value=18, max_value=150, value=30,step=1)
person_income=st.number_input("Person's Income", min_value=0, value=50000,step=1000)
person_home_ownership=st.selectbox("Person's Home Ownership", options=df["person_home_ownership"].unique())
person_emp_length=st.number_input("Person's Employment Length", min_value=0, value=5,step=1)
person_loan_intent=st.selectbox("Person's Loan Intent", options=df["loan_intent"].unique())
person_loan_grade=st.selectbox("Person's Loan Grade", options=df["loan_grade"].unique())
person_loan_amnt=st.number_input("Person's Loan Amount", min_value=0, value=35000,step=500)
person_loan_int_rate=st.number_input("Person's Loan Interest Rate", min_value=0.0, value=10.0,step=0.1)
person_loan_percent_income=st.number_input("Person's Loan Percent Income", min_value=0.0, value=5.0,step=0.1)
person_default_on_file=st.selectbox("Person's Default on File", options=df["cb_person_default_on_file"].unique())
person_credit_history=st.number_input("Person's Credit History", min_value=0, value=5,step=1)

#when predict is clicked
if st.button("Predict Credit Risk"):
    raw_input={
        "person_age":person_age,
        "person_income":person_income,
        "person_home_ownership":person_home_ownership,
        "person_emp_length":person_emp_length,
        "loan_intent":person_loan_intent,
        "loan_grade":person_loan_grade,
        "loan_amnt":person_loan_amnt,
        "loan_int_rate":person_loan_int_rate,
        "loan_percent_income":person_loan_percent_income,
        "cb_person_default_on_file":person_default_on_file,
        "cb_person_cred_hist_length":person_credit_history
    }

    input_df=pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]

    probabilities = model.predict_proba(scaled_input)[0]
    dynamic_confidence = probabilities[prediction] * 100 
    
    if prediction == 1:
        st.error("⚠️ The applicant is likely to default on the loan.")
        st.info(f"Model Confidence: {dynamic_confidence:.2f}% sure they will default.")
    else:
        st.success("✅ The applicant is unlikely to default on the loan.")  
        st.info(f"Model Confidence: {dynamic_confidence:.2f}% sure they will not default.")  