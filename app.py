
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# Load the pre-trained model
pipeline = joblib.load("C:/Users/vipin/OneDrive/Documents/Desktop/Project6/model_03-11-2024-01-52-11-258835.pkl")

# Streamlit UI setup
st.title("Store Sales Prediction Dashboard")

# Input Section: Store ID and File Upload
st.sidebar.header("Input Parameters")
store_id = st.sidebar.text_input("Store ID", "Enter store ID")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    input_data = pd.read_csv(uploaded_file)

    # Additional parameters for user input (if any)
    is_promo = st.sidebar.selectbox("Is Promo", [0, 1])
    
    # Preprocess input data
    input_data['Store_id'] = store_id  # Add store ID
    input_data['IsPromo'] = is_promo  # Add promo field
    
    # Display input data
    st.write("Uploaded Data:")
    st.write(input_data.head())

    # Preprocess and Predict
    def preprocess_input(data):
        # Include any preprocessing steps used during model training
        data['Dates'] = pd.to_datetime(data['Date'])
        data['DayOfWeek'] = data['Dates'].dt.dayofweek
        data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        # Add other transformations as needed
        return data.drop(columns=['Date', 'Dates'])  # Drop non-feature columns

    processed_data = preprocess_input(input_data)
    predictions = pipeline.predict(processed_data)
    
    # Output Prediction Results
    input_data["Predicted Sales"] = predictions
    st.write("Prediction Results:")
    st.write(input_data[['Date', 'Predicted Sales']])

    # Plot Predictions
    st.subheader("Predicted Sales Trend")
    plt.figure(figsize=(10, 5))
    plt.plot(input_data['Date'], input_data['Predicted Sales'], label="Predicted Sales")
    plt.xlabel("Date")
    plt.ylabel("Predicted Sales")
    plt.title("Sales Prediction Over Time")
    plt.legend()
    st.pyplot(plt.gcf())

    # Download Prediction as CSV
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(input_data)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_data,
        file_name="predicted_sales.csv",
        mime="text/csv"
    )
else:
    st.write("Please upload a CSV file to get predictions.")