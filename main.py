import streamlit as st
import pandas as pd
import joblib

# 1. LOAD ARTIFACTS
@st.cache_resource
def load_artifacts():
    try:
        scaler = joblib.load("scaler.joblib")
        columns = joblib.load("columns.joblib")
        model = joblib.load("best_model.joblib")
        return scaler, columns, model
    except FileNotFoundError:
        st.error("Artifacts not found. Please run the training script first to generate .joblib files.")
        return None, None, None

loaded_scaler, loaded_columns, loaded_model = load_artifacts()

# 2. PREPROCESSING FUNCTION
def preprocess_input(age, gender, usage, transactions, sub_type, complaints):
    # 1. Initialize Dictionary with raw inputs
    data = {
        "Age": [age],
        "MonthlyUsageHours": [usage],
        "NumTransactions": [transactions],
        "Complaints": [complaints],
        # We handle Categorical manually to avoid get_dummies shape errors
        "Gender_Male": [1 if gender == "Male" else 0], 
        "SubscriptionType": [0] # Placeholder
    }
    
    df = pd.DataFrame(data)

    # Map: Basic: 1, Premium: 2, Gold: 3
    sub_map = {"Basic": 1, "Premium": 2, "Gold": 3}
    df["SubscriptionType"] = sub_map[sub_type]

    # 3. Ensure Column Order Matches Training Exactly
    df = df.reindex(columns=loaded_columns, fill_value=0)

    # Identify numeric columns (based on your training logic)
    numeric_cols = ["Age", "MonthlyUsageHours", "NumTransactions", "SubscriptionType", "Complaints"]
    
    try:
        df[numeric_cols] = loaded_scaler.transform(df[numeric_cols])
    except ValueError:
        # If scaler expects the whole dataframe including Gender_Male
        df = pd.DataFrame(loaded_scaler.transform(df), columns=loaded_columns)

    return df

# Streamlit app
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict probability of churn.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    muh = st.number_input("Monthly Usage Hours", min_value=0.0, value=50.0)

with col2:
    num_trans = st.number_input("Number of Transactions", min_value=0.0, value=10.0)
    sub_type = st.selectbox("Subscription Type", ["Basic", "Premium", "Gold"])
    complaints = st.number_input("Number of Complaints", min_value=0.0, value=2.0)

# Prediction Logic
if st.button("Predict Churn"):
    if loaded_model is not None:
        # Preprocess
        processed_df = preprocess_input(age, gender, muh, num_trans, sub_type, complaints)

        # Predict
        try:
            prediction = loaded_model.predict(processed_df)[0]
            probability = loaded_model.predict_proba(processed_df)[0][1]

            # Display Results
            st.divider()
            if prediction == 1:
                st.error(f"Customer is Likely to Churn (Probability: {probability:.2%})")
                st.write("Suggested Action: Offer a discount or reach out to customer support.")
            else:
                st.success(f"Customer is NOT Likely to Churn (Probability: {probability:.2%})")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Tip: Ensure your 'scaler.joblib' contains a StandardScaler object, not a DataFrame.")