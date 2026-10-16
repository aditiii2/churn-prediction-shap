import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction & Explainability System")
st.markdown("Predict customer churn probability and analyze underlying feature impact using SHAP.")

# 1. Load saved model and preprocessing artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# 2. Sidebar Inputs
st.sidebar.header("User Inputs")

def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=780.0)
    
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.sidebar.radio("Paperless Billing", ["Yes", "No"])
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method,
        'PaperlessBilling': paperless_billing
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Selected Customer Information")
st.dataframe(input_df)

# 3. Preprocess and Scale Data
def preprocess_and_scale(df, model_cols, scaler_obj):
    df_copy = df.copy()
    
    # Check numeric columns that the scaler expects
    if hasattr(scaler_obj, "feature_names_in_"):
        numeric_cols = list(scaler_obj.feature_names_in_)
    else:
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Scale numeric features
    df_copy[numeric_cols] = scaler_obj.transform(df_copy[numeric_cols])
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_copy)
    
    # Align with model expected columns
    for col in model_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[model_cols]
    return df_encoded

processed_input = preprocess_and_scale(input_df, model_columns, scaler)

# 4. Dynamic Prediction
if st.button("Predict Churn Risk"):
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(processed_input)[0][1]
        churn_risk = "High Churn Risk 🚨" if prob > 0.5 else "Low Churn Risk ✅"
        score_display = f"{prob * 100:.1f}%"
    else:
        margin = model.decision_function(processed_input)[0]
        churn_risk = "High Churn Risk 🚨" if margin > 0 else "Low Churn Risk ✅"
        score_display = f"{margin:.2f}"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Status", value=churn_risk)
    with col2:
        st.metric(label="Churn Decision Margin / Score", value=score_display)

# Add a divider and extra padding before SHAP section
    st.divider()

# 5. SHAP Explainability Plot (Custom Dark-Mode Bar Chart)
    st.subheader("💡 Model Explainability (SHAP Feature Impact)")
    
    try:
        # Define prediction function for explainer
        pred_func = model.predict_proba if hasattr(model, "predict_proba") else model.decision_function
        
        # Pass synthetic zero background for KernelExplainer baseline calculation
        background = np.zeros((1, processed_input.shape[1]))
        explainer = shap.KernelExplainer(pred_func, background)
        shap_vals = explainer.shap_values(processed_input)
        
        # Extract 1D array of feature importance scores
        if isinstance(shap_vals, list):
            importance_scores = np.abs(shap_vals[1][0])
        else:
            importance_scores = np.abs(shap_vals[0])
            
        feature_importance = pd.DataFrame({
            'col_name': processed_input.columns,
            'importance': importance_scores
        }).sort_values(by='importance', ascending=True).tail(10)

        # Matplotlib plot configured specifically for Streamlit dark theme
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh(feature_importance['col_name'], feature_importance['importance'], color='#00C49F')
        
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white', labelsize=10)
        ax.set_xlabel('Mean |SHAP Value| (Feature Impact)', color='white', fontsize=10)
        
        for spine in ax.spines.values():
            spine.set_color('#333333')
            
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as err:
        st.warning(f"Could not render SHAP plot: {err}")
