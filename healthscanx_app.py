
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import subprocess
import os

# Dynamically install mediapipe if not available
try:
    import mediapipe as mp
except ImportError:
    subprocess.check_call(["pip", "install", "mediapipe==0.10.9"])
    import mediapipe as mp


st.set_page_config(page_title="HealthScanX", layout="centered")
st.title("🖐️ HealthScanX: AI-Powered Hand Health Analyzer")

# Upload section
uploaded_file = st.file_uploader("Upload a clear image of your hand 👇", type=["jpg", "jpeg", "png"])

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Analysis function
def analyze_hand(image: Image.Image):
    st.subheader("📊 Analysis Results")

    # Convert to RGB and prepare image
    img = np.array(image.convert('RGB'))
    result = hands.process(img)

    if not result.multi_hand_landmarks:
        st.warning("🙁 No hand detected. Please upload a clearer image.")
        return

    # Draw hand landmarks
    annotated_image = img.copy()
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    st.image(annotated_image, caption="🖼️ Detected Hand Landmarks", use_column_width=True)

    # Simulated checks
    st.markdown("### 🩸 Blood Flow & Skin Health Check")

    avg_red = np.mean(img[:, :, 0])
    if avg_red < 100:
        st.error("🔴 Low red tone detected: Possible anemia or poor circulation")
    elif avg_red > 180:
        st.warning("🟠 High red tone: Possible inflammation or skin condition")
    else:
        st.success("🟢 Normal skin tone detected")

    st.markdown("### 💅 Nail Health (Simulated)")
    st.info("Pale or bluish nails might suggest anemia or low oxygen levels. (Detailed analysis coming soon)")

    st.markdown("### 💬 AI Summary")
    st.write("✅ This is a prototype system. For medical confirmation, always consult a doctor.")

# App logic
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Hand Image", use_column_width=True)
    analyze_hand(image)
else:
    st.info("📸 Please upload a hand image to begin diagnosis.")
