# _pages/video.py

'''This is a Streamlit app for parking spot detection using a pre-trained model.
   The app allows users to upload a video and a binary mask, or use a sample video and mask.
    It processes the video frame by frame, detects parking spots, and classifies them as occupied or free.
    The results are displayed in real-time with bounding boxes around the detected spots.
    The app also provides options to stop the detection and clear the results.
    The model is based on MobileNetV2 and is trained to classify parking spots as occupied or empty.
    The app uses OpenCV for video processing and TensorFlow for model inference.'''	

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import shutil
import os

#VIDEO_PATH = '../video/parking_1920_1080_loop.mp4'
VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'video', 'parking_1920_1080_loop.mp4')

#MODEL_PATH = '../model/model.keras'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'model', 'model.keras')

#MASK_PATH = '../video/mask_1920_1080.png'
MASK_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'video', 'mask_1920_1080.png')

IMG_SIZE = (224, 224)

# Load the model only once and cache it for performance
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# Load the model
model = load_model()

def get_parking_spots_bboxes(connected_components):
    """
    Get the parking spots bounding boxes from the connected components.


    Parameters:
        -connected_components : tuple
    Returns
        - parking_spots_bboxes : list of tuples
    """

    # connected_components
    (totalLabels, label_ids, values, centroid) = connected_components


    # Get the parking spots bounding boxes
    parking_spots_bboxes = []


    # Get the parking spots bounding boxes
    # The first label is the background, so we start from 1
    coef = 1
    for i in range(1, totalLabels):
        x = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        parking_spots_bboxes.append((x, y, w, h))
       
    return parking_spots_bboxes

# Load and prepare the image for prediction
def load_and_prepare_image(spot_crop):
    img = tf.convert_to_tensor(spot_crop, dtype=tf.float32)
    processed_image  = tf.image.resize(img, IMG_SIZE)
    processed_image  = tf.expand_dims(processed_image , axis=0)
    return processed_image

# Predict if the parking spot is empty or occupied
def empty_or_not(spot_crop):
    processed_image = load_and_prepare_image(spot_crop)
    prediction = model.predict(processed_image)
    return 'Occupied' if prediction > 0.5 else 'Free'

def page():
    st.title("🎥 Parking Spot Counter")

    st.markdown("""Upload a video or press start detection to use sample""")
    video_file = st.file_uploader(label="Upload a video or press start detection to use sample", type=["mp4"],
                                     label_visibility="collapsed")


    st.markdown("""Upload a mask or press start detection to use sample""")
    mask_file = st.file_uploader(label="Upload a mask or press start detection to use sample", type=["png"],
                                     label_visibility="collapsed")

    st.markdown("""
        <style>
            .stButton>button {
                color: red !important;
            }
            .stButton>button:hover {
                color: red !important;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([6.5,6.5,4])
    start_user_sim = col1.button("▶ Start counter")
    start_sample_sim = col2.button("🎥 Use Sample Video")
    stop_simulation = col3.button("🛑 Stop Detection")

    st.markdown("""
        <style>
            /* Change color of the warning message text to red */
            .stAlert {
                color: red !important;
            }
            .stAlert p {
                color: red !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Set paths depending on the button pressed
    if start_user_sim:

        if not video_file or not mask_file:
            if not video_file:
                st.warning("Please upload a video file.")
            if not mask_file:
                st.warning("Please upload a mask file.")
            return

        # Handle uploaded video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_file.read())
            video_path = temp_video_file.name
        
        # Handle uploaded mask file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_mask_file:
            temp_mask_file.write(mask_file.read())
            mask_path = temp_mask_file.name

    elif start_sample_sim:
      
        video_path = VIDEO_PATH
        mask_path = MASK_PATH

    elif stop_simulation:
        st.info("🚫 Detection stopped.")
        return

    if start_sample_sim or start_user_sim:
        cap = cv2.VideoCapture(video_path)
        if hasattr(mask_path, "read"):
            mask = cv2.imdecode(np.frombuffer(mask_path.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        connected_components = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        parking_spots_bboxes = get_parking_spots_bboxes(connected_components)

        frame_placeholder = st.empty()
        counter_placeholder = st.empty()

        step = 120
        spots_status = [None for _ in parking_spots_bboxes]
        frame_nmr = 0
        ret = True

        while ret:
            if stop_simulation:
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("No more frames to process.")
                break

            if frame_nmr % step == 0:
                for spot_id, bbox in enumerate(parking_spots_bboxes):
                    x, y, w, h = bbox
                    spot_crop = frame[y:y+h, x:x+w, :]
                    spot_status = empty_or_not(spot_crop)
                    spots_status[spot_id] = spot_status

            occupied_count = spots_status.count('Occupied')

            for spot_id, bbox in enumerate(parking_spots_bboxes):
                x, y, w, h = bbox
                color = (0, 0, 255) if spots_status[spot_id] == 'Occupied' else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.rectangle(frame, (50, 20), (500, 70), (50, 50, 50), -1)
            cv2.putText(
                frame,
                f"Available: {len(spots_status) - occupied_count} / {len(spots_status)}",
                (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            frame_nmr += 1

        cap.release()
