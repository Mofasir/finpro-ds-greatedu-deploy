import os
import io
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import base64
import plotly.express as px
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="CO2 Emission Predict App",
    layout="wide",
    page_icon="🌍"
)

# Getting the working directory of the emission_app.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Description", "Analytics","Predict Emissions", "About Us"],
        icons=["house", "clipboard", "graph-up", "cloud", "people"],
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

# Load the image background
image_path = os.path.join(working_dir, "image/background.png")
image = Image.open(image_path)
buffered = io.BytesIO()
image.save(buffered, format="PNG")
img_bg = base64.b64encode(buffered.getvalue()).decode()

# Load the image logo
image_path = os.path.join(working_dir, "image/logo.png")
image = Image.open(image_path)
buffered = io.BytesIO()
image.save(buffered, format="PNG")
img_logo = base64.b64encode(buffered.getvalue()).decode()

# Home Page
if selected == "Home":
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_bg}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .header {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .logo {{
            width: 360px;
            height: 60px;
            background-image: url("data:image/png;base64,{img_logo}");
            background-size: contain;
            background-repeat: no-repeat;
        }}
        .title {{
            text-align: left;
        }}
        .main-content {{
            display: flex;
            justify-content: space-between;
            border-radius: 10px;
            margin-top: 20px;
        }}
        .fotodas{{
            display: flex;
            justify-content: space-between;
        }}
        .des{{
            flex: 1;
            margin-left: 10px;
            text-align: justify;
        }}
        .description {{
            border-radius: 10px;
            margin-top: 20 px;
        }}
        .objective {{
            flex: 1;
            padding: 20px;
            border-radius: 10px;
            background-color: #D9FAEE;
            margin-right: 10px;
            width: 50%;
        }}
        .benefit {{
            flex: 1;
            padding: 20px;
            border-radius: 10px;
            background-color: #D9FAEE;
            width: 50%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
    """

        <div class="header">
            <div class="logo"></div>
            <div class="title">
                <h1>Predict CO2 Emission in Rwanda</h1>
            </div>
        </div>
        <hr style="margin: 5px 0; border-top: 1px solid #ddd;">
        <div class="main-content">
            <div class="description">
                <p style="text-align: justify;">
                Kemampuan untuk memantau emisi karbon secara akurat merupakan langkah penting dalam perjuangan melawan perubahan iklim. 
                Pembacaan karbon yang tepat memungkinkan para peneliti dan pemerintah memahami sumber dan pola keluaran massa karbon. 
                Meskipun Eropa dan Amerika Utara memiliki sistem ekstensif untuk memantau emisi karbon di lapangan, hanya sedikit sistem yang tersedia di Afrika.
                </p>
                <p style="text-align: justify;">
                    <span><b>- Emisi CO2 yang Rendah di Rwanda </b></span><br>
                    Negara dengan emisi CO2 yang rendah, dengan emisi per kapita 0,4 ton CO2 pada tahun 2020.<br>
                    <span><b>- Target Emisi Nol Bersih 2050</b></span><br>
                    Negara Rwanda telah berkomitmen untuk mencapai target emisi nol bersih pada tahun 2050.<br>
                    <span><b>- Tantangan di Rwanda</b></span><br>
                    Pelepasan gas karbon dioksida ke atmosfer sebagai hasil dari berbagai aktivitas manusia dan alam.<br>
                    <span><b>- Penanggulangan Emisi CO2</b></span><br>
                    Melacak dan memahami sumber emisi CO2 serta mencari solusi untuk mengurangi emisi CO2.
                </p>
                <div style='display: flex; justify-content: space-between; margin-bottom: 20px;'>
                  <div class="objective">
                    <h6 style="text-align: center; font-weight:bold">Tujuan</h6>
                    <p style="text-align: justify;">
                        <span><b>1. Model Prediksi</b></span><br>
                        Mengembangkan model machine learning yang mampu memprediksi emisi CO2 untuk setiap minggu dan setiap lokasi di Rwanda.<br>
                        <span><b>2. Identifikasi Skenario Emisi</b></span><br>
                        Menjelajahi berbagai skenario berdasarkan berbagai faktor polutan, sehingga memungkinkan para pembuat kebijakan untuk memahami konsekuensi lingkungan.<br>
                        <span><b>3. Benefit</b></span><br>
                        Rekomendasi kebijakan konkret yang dirumuskan untuk memandu Rwanda dalam mengurangi emisi CO2 secara efektif.<br>
                    </p>
                  </div>
                </div>
            </div>
        </div>
    """, 
    unsafe_allow_html=True
    )
    
    # Footer
    st.markdown("""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    <div style="display: flex; flex-direction: column;">
        <hr style="border-top: 1px solid #ddd; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
            <div style="display: flex; align-items: center; font-size: 15px; color: #4C4D50;">
                <i class="far fa-copyright" style="font-size: 15px; margin-right: 5px;"></i>
                2024 <span style="margin-left: 2px">PyBoys Group | Data Scientist in GreatEdu | SIB Cycle 6</span>. All Rights Reserved
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Data Description Page
elif selected == "Data Description": 
    # Set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_bg}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title('Data Description')

# Analytics Page
elif selected == "Analytics":
    # Set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_bg}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title('Analytics')

# Predict Emissions Page
elif selected == "Predict Emissions":
    # Set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_bg}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title('CO2 Emission Prediction using ML (XGBoost Regressor)')

    # Load the model
    try:
        model_file = 'saved_model/emission_model.sav'
        with open(model_file, 'rb') as f:
            emission_model = pickle.load(f)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

    # Maximum week value (should be derived from your training data)
    max_week_val = 52

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
        year = st.text_input('Year')
    with col1:
        longitude = st.text_input('Coordinate of Longitude')
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

# About Us Page
elif selected == "About Us":
    # Set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_bg}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("About Us")
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
