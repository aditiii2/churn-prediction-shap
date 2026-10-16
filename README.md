# 📊 Telco Customer Churn Prediction & Explainability Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-prediction-shap.streamlit.app)

An end-to-end Machine Learning web application built to predict customer churn risk using Support Vector Machines (SVM) and interpret predictions using SHAP (Shapley Additive exPlanations).

---

## 🚀 Live Demo
Access the live interactive application here: **[Customer Churn Predictor](https://churn-pred-shap.streamlit.app/)** 

---

## 📌 Features
* **Interactive Prediction Interface**: Customize customer parameters (tenure, monthly charges, contract type, payment method, etc.) to evaluate real-time churn risk.
* **Robust Preprocessing Pipeline**: Automatically applies one-hot encoding, aligns feature dimensions, and scales numeric fields using `StandardScaler`.
* **Explainable AI (XAI)**: Integrated SHAP (`KernelExplainer`) feature attributions highlight top key factors driving individual prediction scores.
* **Executive Dark-Mode UI**: Built with Streamlit and Matplotlib with a streamlined visual design.

---

## 🛠️ Project Structure
```text
├── app.py                      # Main Streamlit web application
├── 1.ipynb                     # Jupyter notebook for EDA, model training & export
├── churn_model.pkl             # Pre-trained SVM model (joblib)
├── scaler.pkl                  # Fitted StandardScaler object (joblib)
├── model_columns.pkl           # Trained feature column order (joblib)
├── Telco-Customer-Churn.csv    # Source dataset
├── requirements.txt            # Python dependencies for deployment
└── README.md                   # Project documentation
