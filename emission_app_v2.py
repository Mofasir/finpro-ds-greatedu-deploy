import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="CO2 Emission Predict App",
    layout="wide",
    page_icon="🌍"
)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Predict Emissions", "About"],
        icons=["house", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Helper function for sine-cosine encoding
def sin_cos_encode(latitude, longitude, week_no, month, max_week_val):
    latitude_sin = np.sin(np.radians(latitude))
    latitude_cos = np.cos(np.radians(latitude))
    longitude_sin = np.sin(np.radians(longitude))
    longitude_cos = np.cos(np.radians(longitude))
    
    month_sin = np.sin(2 * np.pi * (month / 12))
    month_cos = np.cos(2 * np.pi * (month / 12))
    week_no_sin = np.sin(2 * np.pi * (week_no / max_week_val))
    week_no_cos = np.cos(2 * np.pi * (week_no / max_week_val))
    
    return np.array([latitude_sin, longitude_sin, latitude_cos, longitude_cos, week_no_sin, week_no_cos, month_sin, month_cos])

# Home Page
if selected == "Home":
    st.title('Welcome to the CO2 Emission Prediction App')
    st.markdown("""
    This application predicts CO2 emissions using machine learning models trained on open-source emissions data.
    Explore the app to predict emissions for different locations and time periods.
    """)

# Predict Emissions Page
elif selected == "Predict Emissions":
    st.title('CO2 Emission Prediction using ML (XGBoost Regressor)')

    # Load the model
    try:
        model_file = 'emission_model.sav'
        with open(model_file, 'rb') as f:
            emission_model = pickle.load(f)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

    # Maximum week value (should be derived from your training data)
    max_week_val = 53

    # Preprocessing functions
    def preprocess_input(latitude, longitude, year, week_no, max_week_val):
        # Calculate the date from year and week_no
        date = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(int(week_no) * 7, unit="d")
        month = date.month

        # Transform input data using the sine-cosine encoding
        encoded_input = sin_cos_encode(float(latitude), float(longitude), int(week_no), month, max_week_val)
        year_int = int(year) - 2019
        input_data = np.concatenate(([year_int], encoded_input)).reshape(1, -1)
        return input_data

    # Getting the input data from the user
    col1, col2 = st.columns(2)

    with col1:
        latitude = st.text_input('Coordinate of Latitude')
    with col2:
        longitude = st.text_input('Coordinate of Longitude')
    with col1:
        year = st.text_input('Year')
    with col2:
        week_no = st.text_input('Number of week')

    # Code for prediction
    emission_predict = ''

    # Creating a button for Prediction
    if st.button('Predict'):
        try:
            user_input = preprocess_input(latitude, longitude, year, week_no, max_week_val)
            emission_predict = emission_model.predict(user_input)
            emission_predict = np.power(emission_predict, 3)  # Transform back the prediction
            st.success(f'Predicted CO2 Emission: {emission_predict[0]}')
        except ValueError as e:
            st.error(f"Invalid input: {e}")

# About Page
elif selected == "About":
    st.title("About This App")
    st.markdown("""
    This application uses machine learning models to predict CO2 emissions based on satellite data.
    The model was trained using data from Sentinel-5P satellite observations.
    
    **Features:**
    - Predict CO2 emissions based on geographical coordinates and time.
    - Easy to use interface with clear input fields.
    
    **Developed by:**
    - Rama Ngurah Putera Pinatih
    - Rahuldi
    - Amin Yazid Achmad
    - Muhammad Asri Alfajri
    - Mohammad Faikar Natsir
    """)
