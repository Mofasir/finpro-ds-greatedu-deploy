import os
import io
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
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

# Home Page
if selected == "Home":
    # Load the image background
    image_path = os.path.join(working_dir, "image/background.png")
    image = Image.open(image_path)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Load the image logo
    image_path = os.path.join(working_dir, "image/logo.png")
    image = Image.open(image_path)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str1 = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
        }}  
        .logo {{
            display: flex;
            background-position: center;
            margin-top: 20px;
            background-image: url("data:image/jpeg;base64,{img_str1}");
            background-size: contain;
            background-repeat: no-repeat;
            width: auto;
            height: auto;
        }}
        .main-content {{
            display: flex;
            justify-content: space-between;
            border-radius: 10px;
            margin-top: 60px;
        }}
        .header .title h1 {{
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
            <div class="title">
                <h1>Predict CO2 Emissions in Rwanda</h1>
            </div>
            <div class="logo"></div>
        </div>
        <hr style="margin: 20px 0; border-top: 1px solid #ddd;">
        <div class="main-content">
            <div class="description">
              <div class = "fotodas">
                <p class = "des">Pentingnya kualitas air dalam menjaga keberlanjutan lingkungan dan kesehatan masyarakat telah menjadi perhatian utama di seluruh dunia. 
                Air adalah aset berharga yang tidak dapat digantikan, namun sering kali terpapar oleh berbagai faktor, mulai dari polusi industri hingga limbah domestik. 
                Dalam menghadapi tantangan ini, perlu adanya upaya untuk mengembangkan sistem prediktif yang mampu memantau dan mengukur kualitas air secara akurat. 
                Melalui analisis data dan teknologi yang inovatif, kita dapat mengidentifikasi pola perilaku air dan memprediksi potensi risiko yang terkait dengan perubahan lingkungan.</p>
              </div>
                <p style="text-align: justify;">Pada Studi Kasus Water Quality Prediction bertujuan untuk mengatasi tantangan ini dengan mengembangkan model prediksi yang efektif dan dapat diandalkan. 
                Dengan memanfaatkan data terkini dan teknik analisis yang canggih, kita dapat memperkirakan kualitas air di lokasi tertentu dan mengidentifikasi faktor-faktor yang berpotensi mempengaruhi. 
                Dengan demikian, upaya ini tidak hanya akan membantu dalam menjaga keberlanjutan sumber daya air, tetapi juga dapat memberikan informasi yang berharga bagi pengambil keputusan dalam menangani masalah lingkungan di masa depan.</p>
                <div style='display: flex; justify-content: space-between; margin-bottom: 20px;'>
                  <div class="objective">
                    <h6 style="text-align: center;">Objective</h6>
                    <p style="font-size: 0.8em; text-align: justify;">
                        <span><b>1. Meningkatkan pemahaman tentang kualitas air</b></span><br>
                        Melalui analisis prediktif, tujuan utama adalah meningkatkan pemahaman tentang faktor-faktor yang memengaruhi kualitas air di berbagai lokasi. Hal ini akan membantu dalam mengidentifikasi sumber polusi dan potensi risiko terhadap kesehatan manusia dan lingkungan.<br>
                        <span><b>2. Peningkatan responsibilitas lingkungan</b></span><br>
                        Dengan memprediksi kualitas air secara akurat, tujuan ini adalah untuk memberikan solusi yang dapat meningkatkan tanggung jawab lingkungan dalam pengelolaan sumber daya air. Hal ini dapat mencakup upaya untuk mengurangi polusi air, mengoptimalkan penggunaan air, dan meminimalkan dampak negatif terhadap ekosistem air.<br>
                        <span><b>3. Mendukung keberlanjutan lingkungan</b></span><br>
                        Melalui pemahaman yang lebih baik tentang kualitas air, tujuan ini adalah untuk mendukung upaya-upaya dalam menjaga keberlanjutan lingkungan. Hal ini dapat mencakup pengelolaan air yang lebih efisien, perlindungan habitat air, dan pemulihan ekosistem air yang terganggu.<br>
                        <span><b>4. Peningkatan kesehatan masyarakat</b></span><br>
                        Dengan memantau kualitas air secara berkala dan melakukan prediksi yang akurat, tujuan ini adalah untuk melindungi kesehatan masyarakat dari risiko yang berkaitan dengan konsumsi air yang tercemar. Hal ini akan membantu dalam mengurangi risiko penyakit terkait air dan meningkatkan kualitas hidup masyarakat secara keseluruhan.
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
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 40px; border-top: 1px solid #ddd; padding-top: 10px;">
        <div style="display: flex; align-items: flex-start; font-size: 15px;">
            <i class="material-icons" style="font-size: 25px; margin-right: 5px; color: #4C4D50 ;">location_on</i>
            <div style="font-size: 15px; color: #4C4D50;">
                Jl. Duren Tiga Raya No.09, RT.12/RW.1, Duren<br>
                Tiga, Kec. Pancoran, Kota Jakarta Selatan,<br>
                Daerah Khusus Ibukota Jakarta 12760
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    <div style="display: flex; flex-direction: column;">
        <hr style="border-top: 1px solid #ddd; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
            <div style="display: flex; align-items: center; font-size: 15px; color: #4C4D50;">
                <i class="far fa-copyright" style="font-size: 20px; margin-right: 5px;"></i>
                2024 <span style="margin-left: 2px"><b>Fun-tastic Four</b></span>. All Rights Reserved
            </div>
            <div style="font-size: 14px; margin: 0; color: #4C4D50;">
                SIB Cycle 6 | 2024
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts CO2 emissions using machine learning models trained on open-source emissions data.
    Explore the app to predict emissions for different locations and time periods.
    """)

# Data Description Page
elif selected == "Data Description":
    st.title('Data Description')

# Analytics Page
elif selected == "Analytics":
    st.title('Analytics')

# Predict Emissions Page
elif selected == "Predict Emissions":
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

# About Page
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
