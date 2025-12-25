"""
DeepScan: AI-Powered Breast Ultrasound Diagnostic Assistant
----------------------------------------------------------
Advanced clinical-style interface for automated breast ultrasound classification
using DenseNet121 + CBAM attention mechanism and AI-based clinical insights.

Author: Firdevs Turgut
Affiliation: AI-Based Breast Cancer Diagnostic Assistant (Research Project)
Date: 2025-12-25
"""

# =====================================================================
# 1. LIBRARY IMPORTS
# =====================================================================
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import date
from dotenv import load_dotenv
from src.ai_agent import MedicalAssistant

# =====================================================================
# 2. ENVIRONMENT CONFIGURATION
# =====================================================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# =====================================================================
# 3. PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="DeepScan | Breast Cancer AI Assistant",
    page_icon="🔬",
    layout="wide"
)

# =====================================================================
# 4. CUSTOM STYLING
# =====================================================================
st.markdown("""
<style>
    .main { background-color: #F9FAFB; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2563EB;
        color: white;
        height: 3em;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #1E40AF; }
    .result-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .report-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    .chat-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        height: 500px;
        overflow-y: auto;
    }
    .user-msg { color: #1E3A8A; font-weight: 600; }
    .ai-msg { color: #111827; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 5. SIDEBAR CONFIGURATION
# =====================================================================
with st.sidebar:
    st.image("assets/assistant-avatar.png", width=110)
    st.title("System Status")

    if api_key:
        st.success("✅ Groq API key loaded successfully.")
    else:
        st.error("⚠️ Missing Groq API key in .env file!")

    st.markdown("""
    ---
    **Model:** DenseNet121 + CBAM  
    **Framework:** TensorFlow / Keras 3  
    **Application:** Breast Ultrasound Classification  
    **Purpose:** Academic research and interpretability analysis.
    ---
    """)

# =====================================================================
# 6. MODEL LOADING WITH CUSTOM OBJECTS
# =====================================================================
@st.cache_resource
def load_best_model():
    """Load trained DenseNet121 + CBAM model with custom layers."""
    model_path = os.path.join("results", "trainer", "models", "best_model.h5")
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found at 'results/trainer/models/best_model.h5'.")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully.")
        return model
    except Exception:
        from tensorflow.keras.layers import Layer
        class CBAM(Layer): 
            def call(self, x): return x
        class Mean(Layer): 
            def __init__(self, axis=-1, keepdims=True, **kwargs): 
                super().__init__(**kwargs); self.axis=axis; self.keepdims=keepdims
            def call(self, x): return tf.reduce_mean(x, axis=self.axis, keepdims=self.keepdims)
        class Amax(Layer): 
            def __init__(self, axis=-1, keepdims=True, **kwargs): 
                super().__init__(**kwargs); self.axis=axis; self.keepdims=keepdims
            def call(self, x): return tf.reduce_max(x, axis=self.axis, keepdims=self.keepdims)

        model = tf.keras.models.load_model(
            model_path, custom_objects={"CBAM": CBAM, "Mean": Mean, "Amax": Amax}, compile=False
        )
        st.success("✅ Model loaded successfully with CBAM, Mean, Amax custom layers.")
        return model

model = load_best_model()

# =====================================================================
# 7. HEADER & INTRODUCTION
# =====================================================================
if os.path.exists("assets/medical-abstract-hero.png"):
    st.image("assets/medical-abstract-hero.png", use_container_width=True)

st.title("🔬 DeepScan: AI-Powered Breast Ultrasound Diagnostic Assistant")
st.write("""
A clinical-grade research prototype demonstrating automated classification of 
breast ultrasound scans using **DenseNet121 + CBAM**.  
This assistant provides prediction confidence and AI-generated clinical insights.
""")

# =====================================================================
# 8. PATIENT INFORMATION (Female fixed)
# =====================================================================
st.subheader("🧍 Patient Information")

col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Name", placeholder="e.g. Jane Doe")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
with col2:
    exam_date = st.date_input("Exam Date", value=date.today())


st.markdown("---")

# =====================================================================
# 9. IMAGE UPLOAD & ANALYSIS
# =====================================================================
st.subheader("📸 Upload Ultrasound Scan")
uploaded_file = st.file_uploader("Upload a Breast Ultrasound Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Ultrasound Scan", use_container_width=True)

    if st.button("🔍 Analyze Scan"):
        if model:
            with st.spinner("Analyzing image using DenseNet121 + CBAM network..."):
                img_resized = image.resize((256, 256))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                preds = model.predict(img_array)
                class_names = ["Benign", "Malignant", "Normal"]
                pred_idx = np.argmax(preds)
                pred_label = class_names[pred_idx]
                confidence = float(np.max(preds) * 100)

                st.session_state['pred'] = pred_label
                st.session_state['conf'] = confidence

                color = "#10B981" if pred_label == "Normal" else "#F59E0B" if pred_label == "Benign" else "#EF4444"
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="color:{color}; margin-top:0;">Prediction: {pred_label}</h2>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                if api_key:
                    assistant = MedicalAssistant(api_key)
                    patient_info = (
                        f"Patient: {patient_name or '[Not Provided]'} | "
                        f"Age: {age} | Exam Date: {exam_date}\n"
                        f"Modality: Ultrasound | Anatomical Region: Breast"
                    )
                    ai_response = assistant.get_clinical_insight(pred_label, confidence)
                    if ai_response["status"] == "success":
                        st.session_state['insight'] = ai_response["output"]
                        st.session_state['patient_info'] = patient_info
                    else:
                        st.error(ai_response["message"])
        else:
            st.error("❌ Model not loaded. Please check model path or integrity.")

# =====================================================================
# 10. CLINICAL REPORT & CHAT SIDE BY SIDE
# =====================================================================
if 'pred' in st.session_state:
    st.markdown("---")
    st.subheader("🧠 AI Clinical Discussion")

    col_left, col_right = st.columns([1.5, 1])
    with col_left:
        if 'insight' in st.session_state:
            st.markdown(f"""
            <div class="report-box">
                <h4>Automated Interpretation Report</h4>
                <p><b>{st.session_state.get('patient_info', '')}</b></p>
                <hr>
                <p>{st.session_state['insight']}</p>
                <hr>
                <small><i>Disclaimer: AI-assisted preliminary assessment, not a medical diagnosis.</i></small>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### 💬 Clinical Assistant Chat")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("Ask a question about this diagnosis:")

        if user_input and api_key:
            assistant = MedicalAssistant(api_key)
            chat_response = assistant.chat(
                user_query=user_input,
                context=st.session_state.get("insight", "")
            )
            if chat_response["status"] == "success":
                st.session_state["chat_history"].append(("user", user_input))
                st.session_state["chat_history"].append(("ai", chat_response["reply"]))
            else:
                st.error(chat_response["message"])

        # Display chat history
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for role, msg in st.session_state["chat_history"]:
            if role == "user":
                st.markdown(f"<p class='user-msg'>🧍‍♀️ {msg}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='ai-msg'>🤖 {msg}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# 11. FOOTER
# =====================================================================
st.markdown("""
---
**DeepScan Breast AI Assistant**  
_Research-Only Academic Prototype — Not for Clinical Use_  
Developed using TensorFlow, Keras 3, and Groq Llama 3.  
""")
