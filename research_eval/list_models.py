"""
Lists every Gemini model your API key can actually call, and whether each
supports generate_content. Run this before trying any new model name in
eval_model_comparison.py, instead of guessing and burning quota on 404s.

Run:
    cd Resume_Analyzer
    python3 research_eval/list_models.py
"""
from eval_scoring import GEMINI_API_KEY
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

print("Models available to this API key:\n")
for m in client.models.list():
    name = m.name.replace("models/", "")
    methods = getattr(m, "supported_actions", None) or getattr(m, "supported_generation_methods", None)
    print(f"  {name:35s} supports: {methods}")
