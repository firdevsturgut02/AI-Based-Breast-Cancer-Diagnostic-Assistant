"""
ai_agent.py
------------
AI-powered Clinical Assistant for Breast Ultrasound Diagnosis (TÜBİTAK Project 2025)

Enhancements:
- Context-aware reporting (BIRADS / radiology language)
- Multilingual summaries (English + Turkish)
- Safe output sanitization
- Structured JSON return for integration with app.py
- Detailed error handling and internal logging

Author: <Your Name>
Email: <Your Email>
"""

import os
import re
import json
import datetime
from groq import Client


class MedicalAssistant:
    """
    A professional AI Radiology assistant powered by Groq's LLaMA 3.3-70B.
    Generates interpretive diagnostic reports and supports conversational Q&A.
    """

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.model = model_name
        self.client = Client(api_key=api_key)

        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"🧠 MedicalAssistant initialized with model: {self.model}")

    # -----------------------------------------------------------------
    # 🔍 CLINICAL INSIGHT GENERATION
    # -----------------------------------------------------------------
    def get_clinical_insight(self, prediction_label: str, confidence: float, language: str = "en"):
        """
        Generates a structured radiology-style interpretation for the AI model output.

        Args:
            prediction_label (str): 'Benign', 'Malignant', or 'Normal'
            confidence (float): Model confidence percentage (0–100)
            language (str): 'en' or 'tr' for bilingual reporting

        Returns:
            dict: {status, analysis, language}
        """

        if language.lower() == "tr":
            prompt = f"""
            Sen, Meme Ultrason görüntüleri konusunda uzmanlaşmış bir Radyoloji Yapay Zekâ asistanısın.
            Aşağıdaki tanısal sonucu profesyonel bir şekilde yorumla:

            🩺 **Tanı Sonucu:** {prediction_label}
            🎯 **Güven Skoru:** {confidence:.2f}%

            Lütfen raporda şunları ekle:
            1. Kısa bir tıbbi yorum (iyi huylu, kötü huylu veya normal)
            2. Görüntü bulgularına göre klinik değerlendirme
            3. BIRADS sınıflamasına göre önerilen bir sonraki adım
            4. Son olarak, bu değerlendirme **yapay zekâ destekli bir ön yorumdur** ve lisanslı bir radyolog tarafından doğrulanmalıdır.
            """
        else:
            prompt = f"""
            You are an expert AI Radiologist specializing in Breast Ultrasound interpretation.
            Provide a structured, concise report for the following AI model result:

            🩺 **Diagnostic Result:** {prediction_label}
            🎯 **Confidence Score:** {confidence:.2f}%

            Please include:
            1. A brief clinical interpretation (Benign, Malignant, or Normal)
            2. Radiological reasoning based on tissue characteristics
            3. Recommended next step following BI-RADS guidelines
            4. End with a clear disclaimer stating this is an **AI-assisted opinion**, not a medical diagnosis.
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a board-certified Radiologist Assistant AI. "
                            "Always use formal medical tone, BI-RADS terminology, and avoid exaggeration."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=800,
            )

            content = (
                response.choices[0].message.content.strip()
                if hasattr(response, "choices") and response.choices
                else "No response generated."
            )

            sanitized = self._sanitize_output(content)
            self._log_interaction("clinical_insight", prediction_label, sanitized)

            return {
                "status": "success",
                "language": language,
                "analysis": sanitized,
                "timestamp": datetime.datetime.now().isoformat(),
            }

        except Exception as e:
            error_msg = f"❌ Error generating clinical insight: {e}"
            print(error_msg)
            return {
                "status": "error",
                "message": error_msg,
            }

    # -----------------------------------------------------------------
    # 💬 CHAT FUNCTIONALITY
    # -----------------------------------------------------------------
    def chat(self, user_query: str, context: str = ""):
        """
        Handles conversational Q&A for breast ultrasound topics.
        Can be integrated with Streamlit chat interface.
        """

        prompt = f"Context: {context}\n\nUser Question: {user_query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI Medical Assistant specializing in breast imaging. "
                            "Always cite BI-RADS or radiological reasoning if applicable. "
                            "Avoid giving direct medical advice; provide educational responses."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=600,
            )

            content = (
                response.choices[0].message.content.strip()
                if hasattr(response, "choices") and response.choices
                else "No response generated."
            )

            sanitized = self._sanitize_output(content)
            self._log_interaction("chat", user_query, sanitized)

            return {
                "status": "success",
                "reply": sanitized,
                "timestamp": datetime.datetime.now().isoformat(),
            }

        except Exception as e:
            error_msg = f"❌ Error during chat: {e}"
            print(error_msg)
            return {
                "status": "error",
                "message": error_msg,
            }

    # -----------------------------------------------------------------
    # 🧹 INTERNAL UTILITIES
    # -----------------------------------------------------------------
    def _sanitize_output(self, text: str) -> str:
        """
        Cleans up model output (removes markdown, unwanted symbols).
        Ensures safe display in Streamlit or HTML contexts.
        """
        clean = re.sub(r"[#*_`]+", "", text)
        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
        return clean

    def _log_interaction(self, mode: str, query: str, response: str):
        """Logs AI interactions to a timestamped JSONL file for auditing or paper appendix."""
        log_file = os.path.join(self.log_dir, "ai_agent_log.jsonl")
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": mode,
            "input": query,
            "response": response,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

