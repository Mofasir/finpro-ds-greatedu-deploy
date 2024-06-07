import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

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

    # Getting the working directory of the emission_app.py
    working_dir = os.path.dirname(os.path.abspath(__file__))

    # Loading the saved models
    model_file = 'emission_model.sav'
    model_path = os.path.join(working_dir, model_file)
    with open(model_path, 'rb') as f:
        emission_model = pickle.load(f)

    # Loading the saved scaler
    scaler_file = 'scaler.sav'
    scaler_path = os.path.join(working_dir, scaler_file)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Preprocessing functions
    def preprocess_input(latitude, longitude, year, week_no, scaler):
        # Transform input data using the loaded scaler
        input_data = np.array([[float(latitude), float(longitude), int(year), int(week_no)]])
        scaled_data = scaler.transform(input_data)
        return scaled_data

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
            user_input = preprocess_input(latitude, longitude, year, week_no, scaler)
            emission_predict = emission_model.predict(user_input)
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