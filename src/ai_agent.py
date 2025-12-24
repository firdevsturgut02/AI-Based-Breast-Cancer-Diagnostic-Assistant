"""
ai_agent.py
-------------------------------------------------
AI-Assisted Clinical Decision Support Module
for Breast Ultrasound Interpretation

IMPORTANT NOTICE:
- This system DOES NOT provide medical diagnosis.
- All outputs are AI-assisted, educational, and decision-support only.
- Final clinical responsibility belongs to licensed radiologists.

-------------------------------------------------
"""

import os
import re
import json
import datetime
from typing import Dict, Any
from groq import Client

# =================================================
# 🧠 MEDICAL ASSISTANT CORE
# =================================================
class MedicalAssistant:
    """
    AI-assisted Radiology Decision Support System
    designed for research and educational use.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        enable_logging: bool = True,
    ):
        self.model = model_name
        self.client = Client(api_key=api_key)

        self.enable_logging = enable_logging
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"[INFO] MedicalAssistant initialized | Model: {self.model}")

    # =================================================
    # 🔍 AI-ASSISTED CLINICAL INTERPRETATION
    # =================================================
    def get_clinical_insight(
        self,
        prediction_label: str,
        confidence: float,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Generates an AI-assisted radiological interpretation.

        DISCLAIMER:
        This output is NOT a medical diagnosis and MUST
        be reviewed by a licensed radiologist.
        """

        confidence = float(max(0.0, min(confidence, 100.0)))

        prompt = f"""
        You are an **AI-based Radiology Decision Support System**.
        You do NOT establish medical diagnoses.

        Model Output:
        - Classification: {prediction_label}
        - Confidence Score: {confidence:.2f}%

        Please structure your response as follows:

        1. Imaging-based preliminary assessment
        2. Probabilistic interpretation using BI-RADS terminology
        3. Recommended clinical next step (screening, follow-up, biopsy consideration, etc.)
        4. Mandatory disclaimer:
           "This is an AI-assisted preliminary opinion,
            not a medical diagnosis. Final evaluation
            must be performed by a licensed radiologist."
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic-grade AI radiology assistant. "
                            "Use formal medical language. "
                            "Never claim diagnostic certainty. "
                            "Always emphasize AI-assisted decision support."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=700,
            )

            raw_output = response.choices[0].message.content
            sanitized_output = self._sanitize_output(raw_output)

            if self.enable_logging:
                self._log_interaction(
                    mode="clinical_insight",
                    input_data={
                        "label": prediction_label,
                        "confidence": confidence,
                        "language": language,
                    },
                    response=sanitized_output,
                )

            return {
                "status": "success",
                "type": "AI_ASSISTED_CLINICAL_INTERPRETATION",
                "language": language,
                "confidence": confidence,
                "output": sanitized_output,
                "disclaimer": (
                    "AI-assisted output only. Not a medical diagnosis."
                ),
                "timestamp": datetime.datetime.now().isoformat(),
            }

        except Exception as e:
            return self._error_response("clinical_insight", e)

    # =================================================
    # 💬 EDUCATIONAL CHAT MODULE
    # =================================================
    def chat(self, user_query: str, context: str = "") -> Dict[str, Any]:
        """
        Educational Q&A module related to breast imaging.
        This function does NOT provide medical advice.
        """

        prompt = f"""
        Context (if available):
        {context}

        User Question:
        {user_query}

        Respond in an educational and explanatory manner.
        Avoid diagnosis or treatment recommendations.
        Reference BI-RADS terminology when appropriate.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an educational medical AI assistant. "
                            "Do not provide diagnosis or treatment advice."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=600,
            )

            output = self._sanitize_output(
                response.choices[0].message.content
            )

            if self.enable_logging:
                self._log_interaction(
                    mode="chat",
                    input_data=user_query,
                    response=output,
                )

            return {
                "status": "success",
                "type": "EDUCATIONAL_RESPONSE",
                "reply": output,
                "timestamp": datetime.datetime.now().isoformat(),
            }

        except Exception as e:
            return self._error_response("chat", e)

    # =================================================
    # 🧹 INTERNAL UTILITIES
    # =================================================
    @staticmethod
    def _sanitize_output(text: str) -> str:
        """
        Removes markdown artifacts and excessive formatting
        to ensure safe rendering in UI environments.
        """
        text = re.sub(r"[#*_`]+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _log_interaction(self, mode: str, input_data, response: str):
        """
        Logs AI interactions for auditability and reproducibility.
        """
        log_file = os.path.join(self.log_dir, "ai_agent_audit.jsonl")
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": mode,
            "input": input_data,
            "response": response,
            "model": self.model,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def _error_response(module: str, error: Exception) -> Dict[str, Any]:
        """
        Standardized error response for safe integration.
        """
        return {
            "status": "error",
            "module": module,
            "message": str(error),
            "timestamp": datetime.datetime.now().isoformat(),
        }
