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

# Load Dataset
@st.cache_data
# Dataset Train (Versi lite)
def load_data():
  dataset_path = os.path.join(working_dir, "datasets/train_lite.csv")
  data = pd.read_csv(dataset_path)
  return data

# Dataset Describe
def load_data1():
  dataset_path1 = os.path.join(working_dir, "datasets/train_desc.csv")
  data = pd.read_csv(dataset_path1)
  return data

# Load the model
model_file = 'saved_model/emission_model.sav'
with open(model_file, 'rb') as f:
    emission_model = pickle.load(f)


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
        align-items: center;
        justify-content: center;
    }}
    .logo {{
        margin-top: -40px;
        width: 360px;
        height: 80px;
        background-image: url("data:image/png;base64,{img_logo}");
        background-size: contain;
        background-repeat: no-repeat;
    }}
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="header">
        <div class="logo"></div>
    </div>
    """,
    unsafe_allow_html=True
)

# Home Page
if selected == "Home":
    st.title('Predict CO2 Emission in Rwanda')
    st.markdown(
        f"""
        <style>
        .main-content {{
            display: flex;
            justify-content: space-between;
            border-radius: 10px;
        }}
        .description {{
            border-radius: 10px;
            margin-top: 20 px;
        }}
        .objective {{
            flex: 1;
            padding: 20px;
            border-radius: 10px;
            background-color: #E6E0CD;
            margin-right: 10px;
            width: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
    """
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
                    <h3 style="font-weight:bold">Tujuan</h3>
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

# Data Description Page
elif selected == "Data Description": 
    st.title('Data Description')
    st.markdown(
        f"""
        <style>
        .stApp {{
            text-align: justify;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
    """
    Dataset yang digunakan bersumber dari https://www.kaggle.com/competitions/playground-series-s3e20/data.
    Dataset ini berisi data emisi sumber terbuka (dari pengamatan satelit Sentinel-5P) untuk memprediksi emisi karbon.
    Sekitar 497 lokasi unik dipilih dari berbagai area di Rwanda, dengan distribusi di sekitar lahan pertanian, kota, dan pembangkit listrik. 
    Data untuk kompetisi ini dibagi berdasarkan waktu tahun 2019 - 2021 termasuk dalam data train, dan tugas kita memprediksi data emisi CO2 untuk tahun 2022 hingga November.

    Tujuh fitur utama diekstraksi setiap minggu dari Sentinel-5P dari Januari 2019 hingga November 2022. 
    Setiap fitur (Sulfur Dioksida, Karbon Monoksida, dll) mengandung sub fitur seperti column_number_density yang merupakan kerapatan kolom vertikal di permukaan tanah, yang dihitung dengan menggunakan teknik DOAS. 
    Kita dapat membaca lebih lanjut mengenai setiap fitur pada tautan di bawah ini, termasuk bagaimana fitur tersebut diukur dan definisi variabel. 
    Kita akan diberikan nilai fitur-fitur ini dalam set test dan tujuan kita untuk memprediksi emisi CO2 dengan menggunakan informasi waktu serta fitur-fitur ini.
    
    **Fitur Utama:**
    - Sulphur Dioxide : https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_SO2
    - Carbon Monoxide : https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_CO
    - Nitrogen Dioxide : https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_NO2
    - Formaldehyde : https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_HCHO
    - UV Aerosol Index : https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_AER_AI
    - Ozone : https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_O3
    - Cloud : https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_CLOUD

    **Berkas-berkas:**
    - train.csv
    - test.csv
    - sample_submission.csv - file contoh pengiriman dalam format yang benar
    """
    )

    st.markdown(
    """
    Berikut dataset (versi lite, tanpa fitur utama) yang digunakan sebagai data pelatihan dalam prediksi :
    """
    )

    df = load_data()

    st.dataframe(df, width=1050)
    total_rows = len(df)
    total_columns = len(df.columns)
    total_rows_formatted = "{:,}".format(total_rows).replace(",", ".")
    st.write(f"Total data terdiri dari {total_rows_formatted} baris dan 76 kolom (Data Train Asli)")

# Analytics Page
elif selected == "Analytics":
    st.title('Analytics')

# Predict Emissions Page
elif selected == "Predict Emissions":
    st.title('CO2 Emission Prediction using ML (XGBoost Regressor)')

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
""", unsafe_allow_html=True
)
