import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import date
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from src.ai_agent import MedicalAssistant

# -----------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="DeepScan | Breast Cancer AI Assistant",
    page_icon="🔬",
    layout="wide"
)

# -----------------------------------------------------------
# CUSTOM STYLING
# -----------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #F9FAFB; }
    .stButton>button {
        width: 100%; border-radius: 8px;
        background-color: #2563EB; color: white;
        height: 3em; font-weight: 500;
    }
    .result-card {
        background-color: #FFFFFF;
        padding: 20px; border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .small-text { font-size: 0.9em; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------
with st.sidebar:
    st.image("assets/assistant-avatar.png", width=100)
    st.title("System Status")

    if api_key:
        st.success("✅ Groq API key loaded successfully.")
    else:
        st.error("⚠️ Missing Groq API key in .env file!")

    st.info("This prototype uses DenseNet121 for classification and Groq Llama 3.3-70B for bilingual clinical reporting.")

    # Visualization option
    show_metrics = st.checkbox("📈 Show training metrics visualization", value=False)
    report_lang = st.radio("🗣️ Report Language", ["English", "Türkçe"], index=0)

# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "final_model.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        st.warning("⚠️ Model file not found (models/final_model.h5)")
        return None

model = load_model()

# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
if os.path.exists("assets/medical-abstract-hero.png"):
    st.image("assets/medical-abstract-hero.png", use_container_width=True)

st.title("🔬 DeepScan: AI-Powered Breast Ultrasound Diagnostic Assistant")
st.write("Academic prototype demonstrating AI-assisted radiological interpretation and bilingual clinical insights.")

# -----------------------------------------------------------
# PATIENT INFORMATION
# -----------------------------------------------------------
st.subheader("🧍 Patient Information")
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("Patient Name", placeholder="e.g. Jane Doe")
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
with col2:
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    exam_date = st.date_input("Exam Date", value=date.today())

st.markdown("---")

# -----------------------------------------------------------
# IMAGE UPLOAD AND PREDICTION
# -----------------------------------------------------------
st.subheader("📸 Upload Ultrasound Scan")
uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Ultrasound Scan", use_container_width=True)

    if st.button("🔍 Analyze Scan"):
        if model:
            with st.spinner("Analyzing scan using DenseNet121..."):
                # Preprocess
                img_array = np.expand_dims(np.array(image.resize((256, 256))) / 255.0, axis=0)
                preds = model.predict(img_array)
                class_names = ["Benign", "Malignant", "Normal"]
                pred_idx = np.argmax(preds)
                pred_label = class_names[pred_idx]
                confidence = np.max(preds) * 100

                st.session_state['pred'] = pred_label
                st.session_state['conf'] = confidence

                # Result Card
                st.subheader("📊 Analysis Results")
                color = "#10B981" if pred_label == "Normal" else "#F59E0B" if pred_label == "Benign" else "#EF4444"
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="color:{color}; margin-top:0;">{pred_label}</h2>
                    <p>Confidence: <b>{confidence:.2f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                # -----------------------------------------------------------
                # AI CLINICAL INTERPRETATION
                # -----------------------------------------------------------
                if api_key:
                    assistant = MedicalAssistant(api_key)
                    patient_info = (
                        f"Patient: {patient_name or '[Not Provided]'}, "
                        f"Age: {age}, Gender: {gender}, Exam Date: {exam_date}"
                    )

                    lang = "tr" if report_lang == "Türkçe" else "en"
                    ai_response = assistant.get_clinical_insight(pred_label, confidence, language=lang)

                    if ai_response["status"] == "success":
                        st.subheader("🧠 AI Clinical Interpretation")
                        st.markdown(f"""
                        ### Breast Ultrasound Report  
                        **{patient_info}**  
                        _Generated on {ai_response["timestamp"]}_

                        {ai_response["analysis"]}
                        """)
                        st.session_state['insight'] = ai_response["analysis"]

                        # Exportable report
                        st.download_button(
                            label="📄 Download Report (TXT)",
                            data=ai_response["analysis"],
                            file_name=f"AI_Report_{patient_name or 'patient'}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error(ai_response["message"])
                else:
                    st.warning("Groq API key missing — cannot generate AI insights.")
        else:
            st.error("Model not loaded. Please ensure the model file exists.")

# -----------------------------------------------------------
# TRAINING METRICS VISUALIZATION
# -----------------------------------------------------------
if show_metrics:
    st.markdown("---")
    st.subheader("📈 Model Training Performance Visualization")

    # Example visualization (replace with real metrics CSV if available)
    try:
        import pandas as pd
        hist_path = os.path.join("results", "training_history.csv")
        if os.path.exists(hist_path):
            df = pd.read_csv(hist_path)
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.lineplot(x=range(len(df["accuracy"])), y=df["accuracy"], ax=ax[0], label="Train Accuracy")
            sns.lineplot(x=range(len(df["val_accuracy"])), y=df["val_accuracy"], ax=ax[0], label="Val Accuracy")
            ax[0].set_title("Accuracy over Epochs")
            ax[0].legend()

            sns.lineplot(x=range(len(df["loss"])), y=df["loss"], ax=ax[1], label="Train Loss")
            sns.lineplot(x=range(len(df["val_loss"])), y=df["val_loss"], ax=ax[1], label="Val Loss")
            ax[1].set_title("Loss over Epochs")
            ax[1].legend()

            st.pyplot(fig)
        else:
            st.info("No training history file found (results/training_history.csv).")
    except Exception as e:
        st.error(f"Error loading visualization: {e}")

# -----------------------------------------------------------
# CHAT ASSISTANT
# -----------------------------------------------------------
if 'pred' in st.session_state:
    st.markdown("---")
    st.subheader("💬 Discuss Findings with AI Assistant")

    user_input = st.text_input("Ask a question about this diagnosis:")
    if user_input and api_key:
        assistant = MedicalAssistant(api_key)
        chat_response = assistant.chat(
            user_query=user_input,
            context=st.session_state.get("insight", "")
        )

        if chat_response["status"] == "success":
            st.info(chat_response["reply"])
        else:
            st.error(chat_response["message"])
