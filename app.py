"""
DeepScan: AI-Powered Breast Ultrasound Diagnostic Assistant
Professional Unified Version
"""

import sys
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import date
from dotenv import load_dotenv
from keras.utils import custom_object_scope

# Add 'src' directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from ai_agent import MedicalAssistant

# =====================================================================
# 1. CONFIGURATION & STYLE
# =====================================================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="DeepScan | AI Breast Assistant",
    page_icon="🔬",
    layout="wide"
)

def apply_custom_styles():
    st.markdown("""
        <style>
        .stApp { background-color: #F8FAFC; }
        
        /* Clinical Report Design */
        .report-container {
            background-color: #FFFFFF;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            height: 550px;
            overflow-y: auto;
            color: #1E293B;
        }
        
        /* Style of Chat */
        .chat-container {
            background-color: #e5ddd5;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
            min-height: 200px;
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #d1d1d1;
        }

        .message {
            padding: 8px 12px;
            margin-bottom: 8px;
            max-width: 85%;
            font-size: 14px;
            line-height: 1.5;
            box-shadow: 0 1px 0.5px rgba(0,0,0,0.1);
        }

        .user-bubble {
            background-color: #dcf8c6;
            align-self: flex-end;
            border-radius: 7.5px 0 7.5px 7.5px;
            margin-left: auto;
        }

        .ai-bubble {
            background-color: #ffffff;
            align-self: flex-start;
            border-radius: 0 7.5px 7.5px 7.5px;
            margin-right: auto;
        }

        .sender-label {
            font-size: 11px;
            font-weight: bold;
            color: #075e54;
            display: block;
            margin-bottom: 2px;
        }

        .result-card {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;
            font-weight: 700;
            font-size: 1.6rem;
            margin-bottom: 20px;
        }

        /* Form styling to remove borders and padding */
        div[data-testid="stForm"] {
            border: none;
            padding: 0;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# =====================================================================
# 2. MODEL LOADING
# =====================================================================
@st.cache_resource
def load_analysis_model():
    path = "results/trainer/models/best_model.h5"
    if not os.path.exists(path): return None
    try:
        def Mean(**kwargs): return tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=kwargs.get("axis", -1), keepdims=kwargs.get("keepdims", True)))
        def Amax(**kwargs): return tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=kwargs.get("axis", -1), keepdims=kwargs.get("keepdims", True)))
        custom_objects = {"Mean": Mean, "Amax": Amax, "Average": Mean}
        with custom_object_scope(custom_objects):
            return tf.keras.models.load_model(path, compile=False)
    except: return None

model = load_analysis_model()

# =====================================================================
# 3. SIDEBAR
# =====================================================================
with st.sidebar:
    avatar_path = "assets/assistant-avatar.png"
    if os.path.exists(avatar_path):
        st.image(avatar_path, width=120)
    
    st.title("DeepScan AI")
    st.markdown("---")
    st.subheader("Patient Records")
    p_name = st.text_input("Name", placeholder="Patient Full Name")
    p_age = st.number_input("Age", 1, 120, 45)
    e_date = st.date_input("Exam Date", date.today())

# =====================================================================
# 4. HERO SECTION
# =====================================================================
h_col1, h_col2 = st.columns([2, 1])
with h_col1:
    st.title("Breast Ultrasound Analysis")
    st.markdown("Automated diagnostic support and clinical reporting system.")
with h_col2:
    hero_path = "assets/medical-abstract-hero.png"
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)

st.divider()

# =====================================================================
# 5. ANALYSIS & REPORTING
# =====================================================================
up_col, res_col = st.columns([1, 1], gap="large")

with up_col:
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader("Upload scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ultrasound Scan", use_container_width=True)
        
        if st.button("RUN ANALYSIS", use_container_width=True):
            if model:
                with st.spinner("Processing..."):
                    img_array = np.expand_dims(np.array(image.resize((256, 256))) / 255.0, axis=0)
                    preds = model.predict(img_array)
                    classes = ["Benign", "Malignant", "Normal"]
                    idx = np.argmax(preds)
                    
                    st.session_state["p_class"] = classes[idx]
                    st.session_state["p_conf"] = float(np.max(preds) * 100)
                    
                    if api_key:
                        assistant = MedicalAssistant(api_key)
                        meta = {"name": p_name, "age": p_age, "exam_date": str(e_date)}
                        report_resp = assistant.generate_clinical_report(st.session_state["p_class"], st.session_state["p_conf"], meta)
                        st.session_state["p_report"] = report_resp.get("report", "")

with res_col:
    if "p_class" in st.session_state:
        st.subheader("Result")
        color = "#EF4444" if st.session_state["p_class"] == "Malignant" else "#10B981" if st.session_state["p_class"] == "Normal" else "#F59E0B"
        
        st.markdown(f"""
            <div class="result-card" style="background-color: {color};">
                {st.session_state["p_class"].upper()} <br>
                <span style="font-size: 0.9rem; font-weight: 300;">Confidence: {st.session_state["p_conf"]:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)
        
        if "p_report" in st.session_state:
            st.markdown("##### Clinical Findings")
            st.markdown(f'<div class="report-container">{st.session_state["p_report"]}</div>', unsafe_allow_html=True)

# =====================================================================
# 6. CHAT Module
# =====================================================================
if "p_report" in st.session_state:
    st.divider()
    st.subheader("Clinical Assistant Chat")
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Chat Display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if not st.session_state["chat_history"]:
        st.markdown('<div style="text-align:center; color:#888; font-size:13px; margin-top:20px;">No messages yet. Ask about BI-RADS or findings below.</div>', unsafe_allow_html=True)
    
    for sender, msg in st.session_state["chat_history"]:
        if sender == "You":
            st.markdown(f'<div class="message user-bubble">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message ai-bubble"><span class="sender-label">Assistant</span>{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat Input
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_q = st.text_input("Message", placeholder="Type your question here...", label_visibility="collapsed")
        with col_btn:
            if st.form_submit_button("Send") and user_q and api_key:
                assistant = MedicalAssistant(api_key)
                chat_resp = assistant.chat(user_q, st.session_state["p_report"])
                if chat_resp["status"] == "success":
                    ans = chat_resp.get("reply") or chat_resp.get("output", "")
                    st.session_state["chat_history"].append(("You", user_q))
                    st.session_state["chat_history"].append(("Assistant", ans))
                    st.rerun()

st.markdown("<br><center><small>DeepScan AI v2.2 | Academic Research | 2026</small></center>", unsafe_allow_html=True)
