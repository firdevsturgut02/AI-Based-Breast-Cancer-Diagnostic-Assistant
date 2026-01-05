"""
ai_agent.py
-------------------------------------------------
DeepScan Reasoning Engine
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
import logging
from typing import Dict, Any, Optional
from groq import Client


class MedicalAssistant:
    """
    DeepScan Reasoning Engine:
    Converts Vision Engine outputs into BI-RADS–compliant,
    structured radiological interpretations and explanations.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "llama-3.3-70b-versatile",
        enable_logging: bool = True,
        log_dir: str = "logs",
    ):
        self.model = model_name
        self.client = Client(api_key=api_key)
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(self.log_dir, "system.log"),
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        print(f"[INFO] DeepScan Reasoning Engine initialized | Model: {self.model}")

    # =================================================
    # CLINICAL REPORT GENERATOR
    # =================================================
    def generate_clinical_report(
        self,
        predicted_class: str,
        confidence_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Converts Vision Engine outputs (classification + confidence)
        into BI-RADS–aligned academic reports.
        """

        confidence_score = float(max(0.0, min(confidence_score, 100.0)))

        patient_info = ""
        if metadata:
            name = metadata.get("name", "N/A")
            age = metadata.get("age", "N/A")
            date = metadata.get("exam_date", "N/A")
            patient_info = f"Patient: {name}, Age: {age}, Exam Date: {date}"

        prompt = f"""
        You are a radiology reasoning engine for DeepScan.
        Generate a BI-RADS–aligned report based on the model prediction below.

        Model Output:
        - Predicted Category: {predicted_class}
        - Confidence Score: {confidence_score:.2f}%
        - {patient_info}

        Structure:
        
        1. Preliminary Imaging Assessment:
        
        2. BI-RADS Probabilistic Interpretation:
        
        3. Morphological Description:
        
        4. Suggested Next Step:
        
        5. Disclaimer:
        - This is an AI-assisted preliminary opinion, not a medical diagnosis.
        - Final interpretation must be made by a licensed radiologist.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional radiology assistant. "
                            "Follow BI-RADS standards and use clear academic language."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=800,
            )

            output = self._sanitize_output(response.choices[0].message.content)
            if self.enable_logging:
                self._log_interaction(
                    "generate_clinical_report",
                    {"class": predicted_class, "confidence": confidence_score},
                    output,
                )

            return self._success_response("BI-RADS_REPORT", {"report": output})

        except Exception as e:
            return self._error_response("generate_clinical_report", e)

    # =================================================
    # CLINICAL INSIGHT WRAPPER
    # =================================================
    def get_clinical_insight(self, predicted_class: str, confidence: float) -> Dict[str, Any]:
        """
        Generates a summarized, structured BI-RADS insight report.
        """
        try:
            response = self.generate_clinical_report(predicted_class, confidence)
            if response["status"] == "success":
                return {"status": "success", "output": response["report"]}
            else:
                return {
                    "status": "error",
                    "message": response.get("message", "Failed to generate report."),
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # =================================================
    # CHAT MODULES
    # =================================================
    def chat(self, user_query: str, context: str = "") -> Dict[str, Any]:
        """
        Unified chat interface — auto-selects academic or patient mode.
        """
        research_keywords = [
            "mechanism", "study", "evidence", "dataset",
            "accuracy", "architecture", "training", "validation"
        ]
        if any(word in user_query.lower() for word in research_keywords):
            return self.chat_research(user_query, context)
        else:
            return self.chat_patient(user_query, context)

    def chat_research(self, user_query: str, context: str = "") -> Dict[str, Any]:
        """
        Research-oriented chat for academic explanations.
        """
        prompt = f"""
        You are a research assistant for DeepScan.
        Provide a detailed, structured answer suitable for academic discussion.

        Context:
        {context}

        User Query:
        {user_query}

        Structure:
        1. Summary
        2. Detailed Explanation
        3. Research Insight
        4. Disclaimer: "AI-assisted educational content only, not a medical diagnosis."
        """

        return self._chat_generic("chat_research", prompt, "RESEARCH_RESPONSE")

    def chat_patient(self, user_query: str, context: str = "") -> Dict[str, Any]:
        """
        Patient-friendly chat explaining radiology terms simply and clearly.
        """
        prompt = f"""
        You are a compassionate healthcare assistant.
        Explain the following in plain, reassuring language.

        Context:
        {context}

        Patient Question:
        {user_query}

        Structure:
        1. Simple Explanation
        2. Meaning
        3. What to Do Next
        4. Reassurance
        """

        return self._chat_generic("chat_patient", prompt, "PATIENT_RESPONSE")

    # =================================================
    # INTERNAL UTILITIES
    # =================================================
    def _chat_generic(self, mode: str, prompt: str, response_type: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Provide structured, factual, and clear responses."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=700,
            )

            output = self._sanitize_output(response.choices[0].message.content)
            if self.enable_logging:
                self._log_interaction(mode, prompt, output)

            return self._success_response(response_type, {"reply": output})

        except Exception as e:
            return self._error_response(mode, e)

    @staticmethod
    def _sanitize_output(text: str) -> str:
        text = re.sub(r"[#*_`]+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _log_interaction(self, mode: str, input_data: Any, response: str):
        try:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "mode": mode,
                "input": input_data,
                "response": response,
                "model": self.model,
            }
            with open(os.path.join(self.log_dir, "reasoning_engine_audit.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.warning(f"Failed to write log: {e}")

    @staticmethod
    def _success_response(resp_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data.update({
            "status": "success",
            "type": resp_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "disclaimer": "AI-assisted output only. Not a medical diagnosis."
        })
        return data

    @staticmethod
    def _error_response(module: str, error: Exception) -> Dict[str, Any]:
        return {
            "status": "error",
            "module": module,
            "message": str(error),
            "timestamp": datetime.datetime.now().isoformat(),
        }
