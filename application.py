import os
import logging
import warnings

# 🤫 STEP 1: Silence all background noise before other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 🛠️ STEP 2: Main Imports
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ⚙️ Page Configuration
st.set_page_config(page_title="Mask Detection", layout="centered", page_icon="🎭")

# 🚀 Cache model (fast loading)
@st.cache_resource
def load_cnn_model():
    return load_model("models/model.h5")

model = load_cnn_model()

# 🧠 Session state management
if "page" not in st.session_state:
    st.session_state.page = "input"

if "img" not in st.session_state:
    st.session_state.img = None

# 🟢 INPUT PAGE
if st.session_state.page == "input":
    st.title("🎭 Mask Detection System")
    st.write("Ensuring safety through AI-powered vision.")
    st.markdown("---")

    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a clear photo of a face", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and display the image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Image Preview", width=400)
        
        # Store in session state for the next page
        st.session_state.img = img

        # 🎯 THE PREDICTION BUTTON (Only appears after upload)
        st.write("") 
        if st.button("🔍 Predict / Analyze Image", type="primary", use_container_width=True):
            st.session_state.page = "prediction"
            st.rerun()

# 🔵 PREDICTION PAGE
elif st.session_state.page == "prediction":
    st.title("🔍 Analysis Result")

    if st.session_state.img is None:
        st.warning("⚠️ No image found. Please upload one first.")
        if st.button("⬅️ Back to Upload"):
            st.session_state.page = "input"
            st.rerun()

    else:
        img = st.session_state.img
        st.image(img, caption="Analyzed Image", use_container_width=True)

        # ✅ Preprocessing
        processed_img = img.resize((64, 64))
        processed_img = np.array(processed_img) / 255.0
        processed_img = np.expand_dims(processed_img, axis=0)

        # 🚀 Model Inference
        with st.spinner("AI is analyzing the image..."):
            prediction = model.predict(processed_img, verbose=0)[0][0]

        # 📊 Classification Logic
        # (Assuming 0 = Mask, 1 = No Mask)
        if prediction > 0.5:
            label = "❌ No Mask Detected"
            confidence = prediction * 100
            st.error(f"### {label}")
        else:
            label = "✅ Mask Detected"
            confidence = (1 - prediction) * 100
            st.success(f"### {label}")

        st.write(f"**Confidence Level:** {confidence:.2f}%")
        st.progress(int(confidence))

        # 💡 Safety Guidelines Section
        st.markdown("---")
        st.subheader("💡 Safety Guidelines")

        if prediction > 0.5:
            st.info("""
            * **Wear a mask** in all public indoor settings.
            * **Keep your distance** (at least 6 feet).
            * **Avoid touching** your eyes, nose, and mouth.
            * **Sanitize** your hands frequently.
            """)
        else:
            st.info("""
            * **Ensure fit:** Make sure the mask covers both nose and mouth.
            * **Cleanliness:** Replace or wash your mask if it becomes damp.
            * **Hygiene:** Wash hands before and after handling your mask.
            """)

        st.write("") # Spacer
        if st.button("⬅️ Upload Another Image"):
            st.session_state.page = "input"
            st.session_state.img = None
            st.rerun()