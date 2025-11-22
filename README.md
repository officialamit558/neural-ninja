# Team - Neural-Ninja
**Members** -
1) Amit Kumar
2) Md Sarim
3) Mahima Dixit
4) Gaurav Singh

# Deployed Live Link
https://neuralninga.streamlit.app/

## System Architecture
![Project Architecture / Design](https://github.com/officialamit558/neural-ninja/blob/main/images/pro_design.png)
![Project Architecture / Design](https://github.com/officialamit558/neural-ninja/blob/main/images/3.jpg)
![Project Architecture / Design](https://github.com/officialamit558/neural-ninja/blob/main/images/4.jpg)

# Customer Churn Prediction

This project aims to design, build, evaluate, and deploy a machine learning model that predicts whether a customer is likely to discontinue using a company's service (i.e., churn). The workflow, dataset structure, and model evaluation strategy are based on the diagrams and instructions provided.

---

## 1. Project Overview

Customer churn prediction helps businesses identify customers at risk of leaving. By detecting churn early, companies can take proactive actions to retain customers, reduce losses, and improve long-term profitability.

**Problem Statement:** Build a machine learning prediction model to:

- Predict customer churn.
- Handle imbalanced dataset.
- Use proper evaluation metrics such as Accuracy, Precision, Recall, F1-score.
- Interpret Confusion Matrix and ROC-AUC curve.
- Deploy the model using a simple user interface.

---

# Customer_Churn_Prediction_ML_Project

Customer Churn Prediction ML Project to implement a full **MLOps workflow**, from data collection to deployment, focused on identifying high-risk customers for proactive retention efforts.

## Project Learnings (Key Achievements & Methodology)

1.  **Implemented a Complete MLOps Workflow:**
    * Successfully integrated **Colab, Kaggle, Pandas, Matplotlib, Seaborn, Scikit-learn,** and **Streamlit** to create an end-to-end, production-ready pipeline.
    * The final model artifact was saved using **Pickle** (serialization) for deployment.
2.  **Handled Imbalanced Classification Data:**
    * Explicitly managed the imbalanced dataset ($\approx 30\%$ Churn rate) by applying the **SMOTE (Synthetic Minority Over-sampling Technique)** during the training phase.
3.  **Developed and Tuned Multiple Models:**
    * Explored and compared classification algorithms including **Logistic Regression, SVM, and Random Forest**.
    * Utilized hyperparameter tuning to optimize performance across models
       1) L1 and L2 regularization
       2) RandomizeSearchCV - parameters - n_estimator, max_depth, crossvalidation 
4.  **Evaluated with Robust Metrics:**
    * Focused model evaluation on business-critical metrics: **Precision, Recall, F1-Score, and ROC-AUC** (Area Under the Curve) to ensure accurate identification of the minority (churning) class.
    * Achieved a final **ROC-AUC of [Insert Your Final Score, e.g., 0.85]**, demonstrating high predictive power.
5.  **Built a User Interface for Prediction:**
    * Designed and deployed a user-friendly web interface using **Streamlit** for real-time customer churn prediction.

---

## System Architecture & Data Flow

The project follows a defined machine learning system architecture, utilizing specialized tools at each stage:

| Stage | Tool(s) | Function |
| :--- | :--- | :--- |
| **Data Collection** | Kaggle / Colab | Data acquisition, Data Augmentation and synthetic data generation. |
| **Data Wrangling** | Pandas | Data cleaning, manipulation, and feature type conversion. |
| **Data Visualization** | Matplotlib / Seaborn | Exploratory Data Analysis (EDA) and identifying correlations. |
| **Feature Engineering, Training & Evaluation** | Scikit-learn | Preprocessing (Encoding, Scaling), SMOTE, Model Training (LR, RF, SVM), and Metric Calculation. |
| **Serialization** | Pickle | Saving the final trained model artifact. |
| **User Interface** | Streamlit | Serving the model for user input and real-time prediction output. |

---

## Detailed Methodology (Workflow Steps)
```
customer-churn/
├─ data/
│  ├─ customer_churn_data.csv       # auto-generated dataset
├─ notebooks/
│  ├─ churn_exploration.ipynb       # Google Colab compatible notebook
├─ src/
│  ├─ data_gen.py                   # dataset creation script
│  ├─ preprocess.py                 # cleaning & preprocessing
│  ├─ features.py                   # feature engineering utilities
│  ├─ train.py                      # training & evaluation pipeline
│  ├─ model_utils.py                # saving, loading & metrics
│  ├─ app_streamlit.py              # Streamlit deployment app
├─ models/
│  ├─ best_model.joblib             # saved model
├─ requirements.txt
├─ README.md

```

### 1. Dataset Description

The dataset consists of **1,000 tuples** and simulates customer data based on the following features:

* **Demographics:** Age ($18-70$), Gender (Male/Female)
* **Usage & Transactions:** MonthlyUsageHours ($5-200$), NumTransactions ($1-50$)
* **Account & Interaction:** SubscriptionType (Basic/Premium/Gold), Complaints ($0-10$)
* **Target:** Churn ($0$=No, $1$=Yes)

### 2. Data Preparation & Feature Engineering

* **Data Splitting:** Data is strictly split into **Train ($80\%$), Test ($10\%$), and Validation ($10\%$**) sets for objective evaluation.
* **Preprocessing:** String data (Gender, SubscriptionType) is converted to numerical data using **One-Hot Encoding**. Numerical features are normalized via **Standardization**.
* **Imbalance Handling:** The **SMOTE** technique is applied to the training set to ensure the minority class is adequately represented for better model learning.

### 3. Model Training & Evaluation
![Project Architecture / Design](https://github.com/officialamit558/neural-ninja/blob/main/images/5.jpg)
![Project Architecture / Design](https://github.com/officialamit558/neural-ninja/blob/main/images/2.jpg)

---

## 5. Evaluation Summary

| Metric             | Result                    |
| ------------------ | ------------------------- |
| Accuracy           | 0.8968481375358166        |        |
| Precision          | 0.8833333333333333        |
| Recall             | 0.9137931034482759        | 
| F1 Score           |  0.8983050847457628       | 

---

The workflow includes iterative model training and evaluation using various classification approaches:

* **Models Explored:** Logistic Regression, Random Forest, and SVM.
* **Hyperparameter Tuning:** Performed using `RandomizedSearchCV` to find optimal parameters (e.g., $C$, `max_depth`).
* **Evaluation:** Results are presented using a **Classification Report** detailing Precision and Recall, and the overall predictive capability is benchmarked using the **ROC-AUC score**.
    

---

## Dependencies
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
requests

### Data Manipulation
pandas
numpy

### Visualization
matplotlib
seaborn
plotly

### Machine Learning & Imbalance Handling
scikit-learn
imblearn
xgboost

### Model Serialization (for saving/loading the trained model)
joblib

---

## 7. Deployment (Streamlit)

Key features:

- Sidebar navigation
- Dataset preview
- User input fields
- Churn prediction output
- Model summary & explanation

Run the app:

```
streamlit run app.py
```

---
# Challenges
1) While using the binaries of the standard scaler , we got error because we weren't passing the input data in correct order.
2) While loading the ui, error handling was not efficient.
3) The User experience of the app could have been better, but due to time constraints we were unable to optimize that.
4) Our model can be made more efficient by online training.

#How will we resolve the challenges ?
1) StandardScaler Order Issue: We resolved this by saving the training column names and strictly enforcing that exact feature order using reindex before applying the scaler.
2) UI Error Handling: We would improve efficiency by wrapping critical functions in try-except blocks to catch crashes and display user-friendly error messages instead of raw code tracebacks.
3) User Experience: We would optimize the interface by organizing input fields into a grid layout, adding tooltips for guidance, and implementing input validation to ensure smoother interaction.
4) Model Efficiency: We would enhance the model's efficiency by setting up a pipeline for online learning or periodic retraining to continuously update the model with new customer data.

## 8. Conclusion

This project demonstrates the complete end‑to‑end lifecycle of a machine learning system for predicting customer churn, including dataset creation, preprocessing, feature engineering, model training, evaluation, deployment, and monitoring.

It follows a practical and industry‑level workflow that can be used in interviews or real‑world ML projects.

---


