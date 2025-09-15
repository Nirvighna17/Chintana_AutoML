import streamlit as st
from PIL import Image
import os
import streamlit.components.v1 as components

components.html("""
    <script>
        history.pushState(null, '', location.href);
        window.onpopstate = function () {
            history.go(1);
        };
    </script>
""", height=0)

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="CHINTANA AI",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse sidebar by default
)

# ---- HIDE SIDEBAR ----
hide_sidebar = """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---- LOAD LOGO SAFELY ----
logo_path = "assets/logo.png"
if not os.path.exists(logo_path):
    st.error("❌ Logo not found! Please make sure 'assets/logo.jpg' exists.")
else:
    logo = Image.open(logo_path)

# ---- CUSTOM CSS ----
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(-45deg, #141E30, #243B55, #0F2027, #2C5364);
            background-size: 400% 400%;
            animation: gradientBG 12s ease infinite;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .title {
            font-size: 3rem;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-top: 1rem;
        }

        .subtitle {
            font-size: 1.2rem;
            color: #cccccc;
            text-align: center;
            margin-bottom: 2rem;
        }

        .info-box {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 1.5rem;
            border-radius: 15px;
            color: #eeeeee;
            text-align: left;
            max-width: 600px;
            margin: auto;
            box-shadow: 0 0 12px rgba(255,255,255,0.08);
        }

        .footer {
            text-align: center;
            font-size: 0.85rem;
            color: #aaaaaa;
            margin-top: 3rem;
        }

        button[kind="primary"] {
            background-color: #00c853 !important;
            color: white !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ---- CENTERED LOGO ----
if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(logo, use_container_width=True)

st.markdown("### Key Features of **Chintana**", unsafe_allow_html=True)
st.markdown("<p style='color: #cccccc; font-size: 0.95rem;'>Everything you need to run your ML projects with zero hassle.</p>", unsafe_allow_html=True)

features = [
    (" Smart Data Intake", "Easily upload CSVs or Excel files and preview data with automatic cleaning."),
    (" Task Detection", "Automatically identifies Classification, Regression, Clustering, or Association Mining tasks."),
    (" Preprocessing & Transformation", "Handles missing values, encoding, scaling, and transformations – both automatic & manual."),
    (" Model Training & Evaluation", "Choose models, tune parameters, and get real-time metrics with visual plots and tables."),
    (" Visual Insights", "Auto-generates visualizations with skewness warnings, correlation heatmaps, and histograms."),
    (" Export & Download", "Download reports, visualizations, models (PKL), and evaluation scores with a click.")
]

# Display in two-column layout
for i in range(0, len(features), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(features):
            with cols[j]:
                st.markdown(
                    f"""
                    <div style="background-color: rgba(255,255,255,0.05); padding: 1rem 1.2rem; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 0 8px rgba(255,255,255,0.06);">
                        <h4 style="color: #ffffff; margin-bottom: 0.3rem;">{features[i + j][0]}</h4>
                        <p style="color: #cccccc; font-size: 0.9rem;">{features[i + j][1]}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ---- GET STARTED BUTTON ----
if st.button(" Get Started", use_container_width=True):
    st.switch_page("pages/login_signup.py")  # Updated to redirect to login/signup page

st.markdown("""
    <div style="background-color: rgba(0, 230, 118, 0.1); border-left: 6px solid #00e676; padding: 1rem 1.5rem; border-radius: 12px; margin-top: 2rem;">
        <h4 style="color: #ffffff;">Need help using Chintana?</h4>
        <p style="color: #cccccc;">Watch this short tutorial where we explain exactly how Chintana works step-by-step.</p>
        <a href="https://www.youtube.com/watch?v=YOUR_VIDEO_ID" target="_blank" style="text-decoration: none;">
            <button style="background-color: #00e676; color: white; padding: 0.6rem 1.2rem; font-weight: bold; border: none; border-radius: 8px; cursor: pointer;">
                ▶️ Watch Tutorial on YouTube
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("""
    <hr style="margin-top: 3rem; border: 0.5px solid #444444;" />

    <div style='text-align: center; color: #cccccc; font-size: 0.9rem; padding-top: 1rem;'>
        © 2025 <b>Chintana</b> · All rights reserved.<br>
        Made by Nirvighna Shendurnikar · <a href='https://github.com/yourrepo' style='color:#00e676;'>GitHub</a> | <a href='mailto:contact@chintana.ai' style='color:#00e676;'>Contact</a>
    </div>
""", unsafe_allow_html=True)
