import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from ultralytics import YOLO

# Load YOLOv12 vehicle detection model
model = YOLO("Desktop/vehicle-detection/best.pt")  # Replace with your YOLO model path

# Page Configuration
st.set_page_config(page_title="🚗 Vehicle Detection", layout="centered")

# --- Custom CSS for Clean, Vibrant UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #0a0a0a 0%, #101e30 100%);
        color: #e5f6ff;
    }

    .main {
        background: rgba(255, 255, 255, 0.03);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(0, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #3eeaff;
        text-align: center;
        font-weight: 800;
        letter-spacing: 1px;
        text-shadow: 0 0 10px #00e6ff77;
    }

    .stButton > button {
        background: linear-gradient(to right, #3f5efb, #fc466b);
        color: #fff;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.6rem;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(252, 70, 107, 0.3);
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 18px rgba(63, 94, 251, 0.6);
    }

    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stFileUploader {
        background-color: rgba(0, 0, 0, 0.2) !important;
        border: 2px dashed #3eeaff !important;
        border-radius: 12px;
        padding: 1.5em;
        transition: border-color 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: #fc466b !important;
    }

    .stSpinner > div {
        color: #3eeaff !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Layout ---
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("## 🚗 YOLOv12 Vehicle Detection System")

# Select input type
option = st.radio("Choose input source:", ["📷 Upload Image", "🎥 Upload Video"])

# --- Image Upload & Detection ---
if option == "📷 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded Image", use_container_width=True)

        with st.spinner("🚦 Detecting vehicles..."):
            results = model.predict(np.array(image), imgsz=640)[0]
            annotated_img = results.plot()
            st.image(annotated_img, caption="✅ Detected Vehicles", use_container_width=True)

# --- Video Upload & Detection ---
elif option == "🎥 Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        with st.spinner("🎥 Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(frame_rgb, imgsz=640)[0]
                annotated_frame = results.plot()
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)
            cap.release()

st.markdown("</div>", unsafe_allow_html=True)
