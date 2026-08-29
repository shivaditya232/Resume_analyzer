"""
One-off diagnostic: print the RAW gemini-2.5-flash-lite response for a single
resume/JD pair, so we can see why match_score parsed to 0 for every pair in
eval_model_comparison.py. Uses only 1 API call.

Run:
    cd Resume_Analyzer
    python3 research_eval/diagnose_flash_lite.py
"""
from eval_scoring import TEST_PAIRS, GEMINI_API_KEY
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

pair = TEST_PAIRS[0]
prompt = f"""You are an expert technical recruiter and career coach.
Given the RESUME and JOB DESCRIPTION below, produce output in EXACTLY this format
(each field on its own line, no extra text):

MATCH_SCORE: (overall match 0-100, integer only)
MISSING_KEYWORDS: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6
CRITICAL_GAPS: gap1 | gap2 | gap3

RESUME:
{pair["resume_text"][:8000]}

JOB DESCRIPTION:
{pair["jd_text"][:3000]}
"""

MODEL = "gemini-flash-lite-latest"
print(f"Calling {MODEL} on {pair['id']}...\n")
response = client.models.generate_content(model=MODEL, contents=prompt)

print("=== response.text (repr, so we can see empty/whitespace-only) ===")
print(repr(response.text))
print()
print("=== finish_reason / candidate info ===")
try:
    for c in response.candidates:
        print("finish_reason:", c.finish_reason)
        print("safety_ratings:", getattr(c, "safety_ratings", None))
except Exception as e:
    print("(could not read candidates)", e)
