import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .churn-yes {
        background-color: #ffcccc;
        color: #cc0000;
        border: 2px solid #cc0000;
    }
    .churn-no {
        background-color: #ccffcc;
        color: #006600;
        border: 2px solid #006600;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model package
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model_package = load_model()
    model = model_package['model']
    le_dict = model_package['label_encoders']
    feature_names = model_package['feature_names']
    accuracy = model_package['accuracy']
    roc_auc = model_package['roc_auc']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown('<div class="main-header">✈️ Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict customer churn using Random Forest Machine Learning Model</div>', unsafe_allow_html=True)

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    st.info("""
    **Algorithm:** Random Forest Classifier

    **Model Performance:**
    - Accuracy: {:.1f}%
    - ROC AUC Score: {:.4f}

    **Features Used:**
    - Age
    - Frequent Flyer Status
    - Annual Income Class
    - Services Opted
    - Social Media Sync
    - Hotel Booking Status
    """.format(accuracy*100, roc_auc))

    st.markdown("---")
    st.header("📁 About")
    st.write("""
    This application predicts whether a customer is likely to churn 
    based on their demographic and service usage data.

    **Developed for:** B.Tech Gen AI (2nd Semester)
    **Project:** Customer Churn Prediction using Random Forest
    """)

# Main content - Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Customer Details")

    with st.form("prediction_form"):
        age = st.slider("Age", min_value=18, max_value=80, value=30, step=1)

        frequent_flyer = st.selectbox(
            "Frequent Flyer Status",
            options=["No", "Yes", "No Record"]
        )

        annual_income = st.selectbox(
            "Annual Income Class",
            options=["Low Income", "Middle Income", "High Income"]
        )

        services_opted = st.slider(
            "Services Opted",
            min_value=1, max_value=10, value=3, step=1
        )

        social_media = st.selectbox(
            "Account Synced to Social Media",
            options=["No", "Yes"]
        )

        hotel_booked = st.selectbox(
            "Booked Hotel or Not",
            options=["No", "Yes"]
        )

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

with col2:
    st.subheader("📈 Prediction Result")

    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'FrequentFlyer': [frequent_flyer],
            'AnnualIncomeClass': [annual_income],
            'ServicesOpted': [services_opted],
            'AccountSyncedToSocialMedia': [social_media],
            'BookedHotelOrNot': [hotel_booked]
        })

        # Encode categorical features
        input_encoded = input_data.copy()
        for col in ['FrequentFlyer', 'AnnualIncomeClass', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']:
            input_encoded[col] = le_dict[col].transform(input_encoded[col])

        # Ensure column order matches training
        input_encoded = input_encoded[feature_names]

        # Make prediction
        prediction = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]

        churn_probability = prediction_proba[1] * 100
        no_churn_probability = prediction_proba[0] * 100

        # Display result
        if prediction == 1:
            st.markdown(
                f'<div class="prediction-box churn-yes">⚠️ CHURN PREDICTED<br>Probability: {churn_probability:.1f}%</div>',
                unsafe_allow_html=True
            )
            st.warning("This customer is likely to churn. Consider retention strategies.")
        else:
            st.markdown(
                f'<div class="prediction-box churn-no">✅ NO CHURN<br>Probability: {no_churn_probability:.1f}%</div>',
                unsafe_allow_html=True
            )
            st.success("This customer is likely to stay. Maintain good service.")

        # Probability bars
        st.markdown("---")
        st.subheader("📊 Probability Breakdown")

        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.metric("No Churn Probability", f"{no_churn_probability:.1f}%")
        with col_prob2:
            st.metric("Churn Probability", f"{churn_probability:.1f}%")

        st.progress(int(no_churn_probability), text="No Churn Confidence")
        st.progress(int(churn_probability), text="Churn Confidence")

        # Show input summary
        st.markdown("---")
        st.subheader("📝 Input Summary")
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)
    else:
        st.info("👈 Fill in the customer details on the left and click 'Predict Churn' to see the result.")

        # Placeholder visualization
        st.markdown("---")
        st.subheader("📊 Sample Prediction")
        sample_data = pd.DataFrame({
            'Scenario': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Age': [35, 30, 28],
            'FrequentFlyer': ['No', 'Yes', 'Yes'],
            'Income': ['High', 'Low', 'Low'],
            'Services': [2, 4, 1],
            'SocialMedia': ['No', 'Yes', 'No'],
            'Hotel': ['Yes', 'No', 'No']
        })
        st.dataframe(sample_data, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 14px;">
    <p>🎓 B.Tech – Gen AI (2nd Semester) | Final Project: Customer Churn Prediction</p>
    <p>Built with ❤️ using Streamlit & Random Forest</p>
</div>
""", unsafe_allow_html=True)
