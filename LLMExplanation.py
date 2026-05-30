from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError(
        f"OPENAI_API_KEY is not set. Add it to .env (one line: OPENAI_API_KEY=...) and save the file."
    )

def build_explanation_prompt(result):
    return f"""
You are explaining the output of an educational MRI tumor classification model.

Model prediction: {result["predicted_class"]}
Confidence: {result["confidence"] * 100:.1f}%
Class probabilities: {result["class_probabilities"]}
Model attention note: {result["explanation_signal"]}

Write a short plain-language explanation for a non-technical user.

Important:
- Do not claim this is a medical diagnosis.
- Say this is an AI model prediction.
- Mention that a clinician/radiologist should confirm the result.
- Keep it under 4 sentences.
"""

client = OpenAI(api_key=_api_key)

def generate_plain_language_explanation(result):
    prompt = build_explanation_prompt(result)

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    return response.output_text


