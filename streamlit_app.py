import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title
st.title("🚗 Car Price Prediction App")
st.write("Welcome! This is my ML project")

# Load dataset
df = pd.read_csv("car data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Feature engineering
current_year = 2024
df['no_year'] = current_year - df['year']

# Drop unnecessary columns
df.drop(['car_name', 'year'], axis=1, inplace=True)

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
try:
    model = pickle.load(open("car_model.pkl", "rb"))
except:
    st.warning("Model file not found. Please upload car_model.pkl")
    st.stop()

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("Enter Car Details")

present_price = st.number_input("Present Price", min_value=0.0)
kms_driven = st.number_input("Kms Driven", min_value=0)
owner = st.number_input("Owner", min_value=0, max_value=3)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

no_year = st.number_input("Car Age (years)", min_value=0)

# Convert categorical to numeric
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
seller_type_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Price"):
    input_data = np.array([[present_price, kms_driven, owner,
                            no_year, fuel_type_diesel,
                            seller_type_individual,
                            transmission_manual]])

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Selling Price: {prediction[0]:.2f} Lakhs")
