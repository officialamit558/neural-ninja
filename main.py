import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL, SCALER, AND COLUMNS

loaded_scaler = joblib.load("scaler.joblib")
loaded_columns = joblib.load("columns.joblib")
loaded_model = joblib.load("best_model.joblib")


# PREPROCESSING FUNCTION (MUST MATCH TRAINING EXACTLY)
def preprocess_input(df):

    # always use loaded_scaler — never use variable named "scaler"
    # 1. Fill NA values
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Gender_Male"] = df["Gender_Male"].fillna(df["Gender_Male"].mode()[0])
    df["MonthlyUsageHours"] = df["MonthlyUsageHours"].fillna(df["MonthlyUsageHours"].mean())
    df["NumTransactions"] = df["NumTransactions"].fillna(df["NumTransactions"].mean())
    df["SubscriptionType"] = df["SubscriptionType"].fillna(df["SubscriptionType"].mode()[0])
    df["Complaints"] = df["Complaints"].fillna(df["Complaints"].median())

    # 2. One-hot encode
    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)

    if "Gender_Male" not in df.columns:
        df["Gender_Male"] = 0

    # 3. Encode subscription
    map_dict = {"Basic": 1, "Premium": 2, "Gold": 3}
    df["SubscriptionType"] = df["SubscriptionType"].map(map_dict)

    # 4. Reindex columns
    df = df.reindex(columns=loaded_columns, fill_value=0)

    print("Scaler type:", type(loaded_scaler))   # diagnostic

    # 5. Scale data
    df_scaled = loaded_scaler.transform(df)

    return df_scaled


# STREAMLIT UI
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict probability of churn.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
muh = st.number_input("Monthly Usage Hours", min_value=0.0, value=50.0)
num_trans = st.number_input("Number of Transactions", min_value=0.0, value=10.0)
sub_type = st.selectbox("Subscription Type", ["Basic", "Premium", "Gold"])
complaints = st.number_input("Number of Complaints", min_value=0.0, value=2.0)

# Button
if st.button("Predict Churn"):

    # Create DataFrame
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender_Male": gender,
        "MonthlyUsageHours": muh,
        "NumTransactions": num_trans,
        "SubscriptionType": sub_type,
        "Complaints": complaints
    }])

    # Preprocess
    processed = preprocess_input(input_df)

    # Predict
    pred = loaded_model.predict(processed)[0]
    prob = loaded_model.predict_proba(processed)[0][1]

    # Output
    if pred == 1:
        st.error(f" Customer Likely to Churn (Probability = {prob:.2f})")
    else:
        st.success(f" Customer Not Likely to Churn (Probability = {prob:.2f})")
