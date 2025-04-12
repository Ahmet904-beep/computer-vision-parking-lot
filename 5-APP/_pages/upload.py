# _pages/upload.py

'''Parking Spot Detection App - Upload Image Page
This page allows users to upload an image of a parking spot and get a prediction of whether it's occupied or empty.
It uses a pre-trained model to make the prediction.
The model is based on MobileNetV2 and is trained to classify parking spots as occupied or empty.
The app uses TensorFlow for model inference and PIL for image processing.
'''

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

#MODEL_PATH = '../model/model.keras'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'model.keras')
IMG_SIZE = (224, 224)

# @st.cache_resource is used to cache the model loading function to avoid reloading it every time the page is refreshed.
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model
model = load_model()

# Preprocess the image to the required size and format
# and add a batch dimension
def load_and_prepare_image(spot_crop):
    img = tf.convert_to_tensor(spot_crop, dtype=tf.float32)
    processed_image  = tf.image.resize(img, IMG_SIZE)
    processed_image  = tf.expand_dims(processed_image , axis=0)
    return processed_image 

# Function to predict if the parking spot is empty or occupied
# using the loaded model
def empty_or_not(spot_crop):
    processed_image  = load_and_prepare_image(spot_crop)
    prediction  = model.predict(processed_image )
    return 'Occupied' if prediction  > 0.5 else 'Empty'

def page():
    st.title("🧾 Upload Parking Spot Image")
    st.markdown("""
    Upload an image of **an overhead single parking spot** to check whether it's **Occupied** or **Empty**.
    """)
    uploaded_file = st.file_uploader(label= "Upload a single parking spot image", 
                                     type=["jpg", "png", "jpeg"],
                                     label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # Display the image in the app
        st.image(image, channels="BGR", use_container_width=True)
        
        img_np = np.array(image)
        label = empty_or_not(img_np)

        st.markdown(f"""
            <div style="background-color: #f0f0f0; color: #333; padding: 10px; border-radius: 5px;">
                <strong>Prediction:</strong> {label}
            </div>
        """, unsafe_allow_html=True)