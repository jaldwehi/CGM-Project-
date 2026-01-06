import os
from openai import OpenAI

def classify_glucose_risk(glucose: float) -> str:
    if glucose < 70:
        return "High risk: Hypoglycemia"
    elif glucose <= 180:
        return "Safe range"
    elif glucose <= 250:
        return "Moderate risk: Hyperglycemia"
    else:
        return "High risk: Hyperglycemia"

SYSTEM_PROMPT = """
You are a glucose monitoring decision-support assistant.
You must provide general, non-medical, non-diagnostic guidance.
Do NOT provide medication dosing (e.g., insulin units) or prescribe treatments.
Focus on lifestyle/safety suggestions and when to contact a clinician.
Keep it short and actionable (4-7 bullet points max).
"""
import os
from dotenv import load_dotenv

load_dotenv()

def generate_llm_response(predicted_glucose: float, risk_label: str, trend: str = "stable") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "LLM is not configured. Add OPENAI_API_KEY to your .env file."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system_prompt = (
            "You are a glucose monitoring assistant. Provide general, non-diagnostic guidance. "
            "Do not give medication dosing or prescriptions. Keep it short and practical."
        )

        user_msg = (
            f"Predicted glucose after 1 hour: {predicted_glucose:.1f} mg/dL\n"
            f"Risk: {risk_label}\n"
            f"Trend: {trend}\n\n"
            "Give 4-7 bullet points of user-friendly recommendations."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=180,
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM error: {type(e).__name__}: {str(e)[:200]}"
