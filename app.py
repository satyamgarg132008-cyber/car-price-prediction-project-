import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Models & Scaler ---
@st.cache_resource
def load_models():
    try:
        with open("car_price_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or Scaler files not found! Please run the notebook and generate them first.")
        return None, None

model, scaler = load_models()

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: 700;
        color: #1F2937;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 40px;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        color: white;
    }
    .result-box {
        background-color: #DEF7EC;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #059669;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown('<div class="main-header">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Get an estimated selling price for your used car!</div>', unsafe_allow_html=True)

# --- Input Form ---
st.markdown("### 📊 Enter Car Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        Present_Price = st.number_input("Present Price (in Lakhs ₹)", min_value=0.0, max_value=200.0, value=5.5)
        Kms_Driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=25000, step=1000)
        Owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
        
    with col2:
        Car_Age = st.number_input("Age of Car (in Years)", min_value=0, max_value=50, value=5)
        Fuel_Type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        Seller_Type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        Transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

    submit_button = st.form_submit_button("Predict Selling Price")

# --- Prediction Logic ---
if submit_button:
    if model is not None and scaler is not None:
        try:
            # Map categorical to numerical based on One-Hot Encoding during training
            fuel_diesel = 1 if Fuel_Type == "Diesel" else 0
            fuel_petrol = 1 if Fuel_Type == "Petrol" else 0
            seller_individual = 1 if Seller_Type == "Individual" else 0
            transmission_manual = 1 if Transmission == "Manual" else 0
            
            # Create feature array matching the training columns:
            # ['Present_Price', 'Kms_Driven', 'Owner', 'Car_Age', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual', 'Transmission_Manual']
            features = pd.DataFrame([{
                'Present_Price': Present_Price,
                'Kms_Driven': Kms_Driven,
                'Owner': Owner,
                'Car_Age': Car_Age,
                'Fuel_Type_Diesel': fuel_diesel,
                'Fuel_Type_Petrol': fuel_petrol,
                'Seller_Type_Individual': seller_individual,
                'Transmission_Manual': transmission_manual
            }])
            
            # Scale features
            scaled_features = scaler.transform(features)
            
            # Predict
            prediction = model.predict(scaled_features)[0]
            
            # Display Result
            if prediction < 0:
                st.warning("⚠️ The predicted price is negative. Please check your inputs.")
            else:
                st.markdown(f"""
                    <div class="result-box">
                        <h3 style="color: #065F46; margin: 0;">Predicted Selling Price:</h3>
                        <h1 style="color: #047857; margin: 0;">₹ {prediction:.2f} Lakhs</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                st.balloons()
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
