'''Parking Spot Detection App'''

import streamlit as st
import _pages.video as video
import _pages.upload as upload
import _pages.model_info as model_info

# App configuration
st.set_page_config(page_title="Parking Spot Detection", page_icon="🚗"
                   , initial_sidebar_state="collapsed")


# App background
def set_background_color(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background_color("white")

# Font style
st.markdown("""
    <style>
    .stApp {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Hide Streamlit UI footer/menu
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚗 Parking Spot Detection")
page = st.sidebar.radio("", ["📹 Real-Time Video", "🖼️ Upload Image", "📊 Model Info"])

# Page 1 - Real-Time Detection
if page == "📹 Real-Time Video":
    video.page()

# Page 2 - Upload Image
elif page == "🖼️ Upload Image":
    upload.page()

# Page 3 - Model Info
elif page == "📊 Model Info":
    model_info.page()
