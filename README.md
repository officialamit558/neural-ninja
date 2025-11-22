# Team - Neural-Ninja
**Members** -
1) Amit Kumar
2) Md Sarim
3) Mahima Dixit
4) Gaurav Singh

![Project Architecture / Design](https://github.com/officialamit558/neural-ninja/blob/main/images/pro_design.png)


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

The workflow includes iterative model training and evaluation using various classification approaches:

* **Models Explored:** Logistic Regression, Random Forest, and SVM.
* **Hyperparameter Tuning:** Performed using `RandomizedSearchCV` to find optimal parameters (e.g., $C$, `max_depth`).
* **Evaluation:** Results are presented using a **Classification Report** detailing Precision and Recall, and the overall predictive capability is benchmarked using the **ROC-AUC score**.
    

---

## Team & Setup

| Name | Contribution Highlights |
| :--- | :--- |
| [Amit Kumar] | [Role] - Full MLOps pipeline construction, SMOTE implementation, Model Tuning. |
| [Md Sarim] | [Role] - EDA, Data Wrangling, Streamlit UI development. |
| [Mahima Dixit] | [Role] - Full MLOps pipeline construction, SMOTE implementation, Model Tuning. |
| [Gaurav Singh] | [Role] - EDA, Data Wrangling, Streamlit UI development. |

## Dependencies


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

### Deployment 
streamlit

