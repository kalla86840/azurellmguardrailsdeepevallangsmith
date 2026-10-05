import os
import re
from openai import OpenAI
from langsmith import traceable

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------
# Simple / Low-Standard Guardrails
# ---------------------------------------------------------
def sanitize_input_guardrail(text: str) -> str:
    """Low-standard PII check: Replaces obvious 6-9 digit MRN patterns."""
    return re.sub(r"\b(MRN[:#\s]*)?\d{6,9}\b", "[REDACTED_MRN]", text, flags=re.IGNORECASE)

def patient_output_guardrail(text: str, role: str) -> str:
    """Low-standard safety check: Appends a basic disclaimer if missing for patients."""
    if role.lower() == "patient":
        disclaimer = "Always consult your physician for personalized medical advice."
        if "consult" not in text.lower() and "physician" not in text.lower():
            return f"{text.strip()}\n\n*Note: {disclaimer}*"
    return text

# ---------------------------------------------------------
# Core Assistant Execution
# ---------------------------------------------------------
SYSTEM_PROMPTS = {
    "physician": "You are a clinical decision support assistant. Provide concise, medical-grade summaries and ICD/pharmacology context.",
    "nurse": "You are a nursing triage assistant. Focus on vitals, symptom severity, bedside workflows, and escalation criteria.",
    "patient": "You are a hospital patient companion. Use plain language (6th-grade reading level), empathetic tone, and clear instructions."
}

@traceable(name="hospital-llm-pipeline")
def ask_hospital_assistant(prompt: str, role: str = "patient") -> str:
    """Executes hospital prompt with role context, basic guardrails, and LangSmith tracing."""
    # 1. Input Guardrail
    sanitized_prompt = sanitize_input_guardrail(prompt)

    # 2. OpenAI Model Call
    system_instruction = SYSTEM_PROMPTS.get(role.lower(), SYSTEM_PROMPTS["patient"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": sanitized_prompt}
        ],
        temperature=0.2
    )
    raw_output = response.choices[0].message.content

    # 3. Output Guardrail
    final_output = patient_output_guardrail(raw_output, role)
    return final_output
