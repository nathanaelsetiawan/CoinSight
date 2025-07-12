import streamlit as st
from PIL import Image

st.set_page_config(
    page_title = "CoinSight | Cryptocurrency Price Forecasting",
    page_icon = "coinsight-logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.logo(
    image="coinsight-logo.png", 
    icon_image="coinsight-logo.png", 
    size="large"
)

# ---------- Multipages ----------
dashboard = st.Page(
    page="pages/dashboard.py",
    title="Dashboard",
)

predict = st.Page(
    page="pages/predict.py",
    title="Price History & Prediction",
)

about = st.Page(
    page="pages/about.py",
    title="About CoinSight",
)

pages = st.navigation([dashboard, predict ,about])

pages.run()

# ---------- Custom seluruh tampilan ----------
st.markdown("""
    <style>
    /* Ubah latar belakang utama ke hitam */
    .stApp {
        background-color: #000000 !important;
    }

    /* Ubah latar belakang sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
    }

    /* Ubah warna teks di sidebar */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Sembunyikan header dan footer bawaan Streamlit */
    header, footer {
        visibility: hidden;
    }
            

    </style>
""", unsafe_allow_html=True)