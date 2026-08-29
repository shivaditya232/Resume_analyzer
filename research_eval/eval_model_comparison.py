"""
Model-variant comparison for the Resume Suite research paper (Graph 4 candidate).

THE POINT
---------
The deployed app uses gemini-2.5-flash for its LLM match score, chosen for speed.
A natural question (and one reviewers/professors often ask): would a bigger,
more "powerful" model do meaningfully better, and is Flash actually the right
tradeoff? This answers that with REAL data -- not a different vendor (no new
API keys needed), but real Gemini model variants under the same account:

    gemini-2.5-flash  -- what the app actually uses (already scored, in results.csv)
    gemini-2.5-pro    -- the larger, more capable model in the same family
    gemini-2.0-flash  -- an older/lighter generation

Mirrors the reference paper's own Fig. 5 + Fig. 6 structure (comparing model
variants within one family on both accuracy and speed), not a baseline-vs-LLM
comparison.

SAMPLE
------
Reuses N_SAMPLE already-scored SAME-FIELD pairs from results.csv (one per
category for diversity) -- their gemini-2.5-flash score/latency are read
directly from results.csv (no new call needed). Only gemini-2.5-pro and
gemini-2.0-flash get fresh calls, on the exact same resume/JD text, so the
comparison is apples-to-apples.

HOW TO RUN
----------
    cd Resume_Analyzer
    python3 research_eval/eval_model_comparison.py

Resumable/quota-safe like every other script here. Free-tier daily quotas are
tracked SEPARATELY per model by Google, so this generally does not compete
with whatever gemini-2.5-flash quota was used elsewhere today -- but if a
model-specific quota is hit, progress is saved and it picks up next run.

OUTPUT
------
    research_eval/model_comparison_results.csv
    research_eval/charts/model_comparison.png
"""
import os
import csv
import time

from eval_scoring import (
    TEST_PAIRS, GEMINI_API_KEY, OUT_DIR, CHART_DIR,
)
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

RESULTS = os.path.join(OUT_DIR, "model_comparison_results.csv")
MAIN_RESULTS = os.path.join(OUT_DIR, "results.csv")

# Our deployed choice, plus a real alternative in the same family.
# NOTE: gemini-2.5-pro and gemini-2.0-flash were tried first but returned
# "limit: 0" on this free-tier key -- zero allocated access, not a rate limit.
# gemini-2.5-flash-lite (version-pinned) returned 404 "no longer available to
# new users." gemini-flash-lite-latest is the alias Google keeps pointed at
# whatever lite model is currently supported, confirmed working via a live
# test call (diagnose_flash_lite.py). So the comparison is: could we have gone
# lighter/faster, and why we didn't.
MODELS = ["gemini-2.5-flash", "gemini-flash-lite-latest"]
MODEL_LABELS = {"gemini-2.5-flash": "Gemini 2.5 Flash\n(our choice)",
                "gemini-flash-lite-latest": "Gemini Flash-Lite\n(lighter/faster)"}

CSV_FIELDNAMES = ["pair_id", "field", "model", "match_score", "latency_s"]


class QuotaExhaustedError(Exception):
    pass


def gemini_generate(prompt: str, model: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            daily_quota = "RESOURCE_EXHAUSTED" in err and ("PerDay" in err or "per day" in err.lower())
            if daily_quota:
                raise QuotaExhaustedError(err)
            rate_limited = "429" in err or "RESOURCE_EXHAUSTED" in err
            overloaded = "503" in err or "UNAVAILABLE" in err
            if (rate_limited or overloaded) and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"    [rate limited/busy, waiting {wait}s, attempt {attempt + 1}/{retries}]")
                time.sleep(wait)
            else:
                print(f"    [generation failed] {err}")
                return ""
    return ""


def gemini_match_score(resume_text: str, jd_text: str, model: str):
    prompt = f"""You are an expert technical recruiter and career coach.
Given the RESUME and JOB DESCRIPTION below, produce output in EXACTLY this format
(each field on its own line, no extra text):

MATCH_SCORE: (overall match 0-100, integer only)
MISSING_KEYWORDS: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6
CRITICAL_GAPS: gap1 | gap2 | gap3

RESUME:
{resume_text[:8000]}

JOB DESCRIPTION:
{jd_text[:3000]}
"""
    raw = gemini_generate(prompt, model)
    score = 0
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("MATCH_SCORE:"):
            try:
                score = int("".join(filter(str.isdigit, line.split(":")[1][:3])))
            except Exception:
                pass
            break
    return score


def _pick_sample():
    """One same-field pair per category, in category order -- reused from
    the already-scored main results.csv so gemini-2.5-flash needs no new call."""
    if not os.path.exists(MAIN_RESULTS):
        raise SystemExit("results.csv not found -- run eval_scoring.py first.")
    with open(MAIN_RESULTS, newline="") as f:
        scored = {r["id"]: r for r in csv.DictReader(f)}

    seen_fields = set()
    sample = []
    for p in TEST_PAIRS:
        if "mismatch" in p["id"]:
            continue
        if p["field"] in seen_fields:
            continue
        if p["id"] not in scored:
            continue  # only use pairs we already have a real flash score for
        seen_fields.add(p["field"])
        sample.append(p)
    return sample, scored


def _load_existing():
    if not os.path.exists(RESULTS):
        return {}
    with open(RESULTS, newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["pair_id"], r["model"]): r for r in rows}


def _save_all(results_dict):
    with open(RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in results_dict.values():
            writer.writerow(row)


def main():
    sample, scored = _pick_sample()
    print(f"Sample: {len(sample)} same-field pairs (one per category).\n")

    results = _load_existing()

    # seed gemini-2.5-flash rows directly from results.csv -- no new call needed
    for p in sample:
        key = (p["id"], "gemini-2.5-flash")
        if key not in results:
            r = scored[p["id"]]
            results[key] = {"pair_id": p["id"], "field": p["field"], "model": "gemini-2.5-flash",
                             "match_score": r["llm_match_pct"], "latency_s": r["llm_latency_s"]}
    _save_all(results)

    todo = [(p, m) for p in sample for m in MODELS[1:]
            if (p["id"], m) not in results]
    print(f"{len(todo)} calls needed ({len(sample)} pairs x {len(MODELS) - 1} new models).\n")

    quota_hit = False
    for i, (p, model) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p['id']} on {model}")
        t0 = time.time()
        try:
            score = gemini_match_score(p["resume_text"], p["jd_text"], model)
        except QuotaExhaustedError as e:
            print(f"\n[QUOTA EXHAUSTED for {model}] {e}\nProgress saved -- re-run later to continue "
                  f"(other models may still have quota left this session).")
            quota_hit = True
            continue
        latency = round(time.time() - t0, 2)
        results[(p["id"], model)] = {"pair_id": p["id"], "field": p["field"], "model": model,
                                      "match_score": score, "latency_s": latency}
        _save_all(results)
        print(f"    score={score}  latency={latency}s\n")

    all_rows = list(results.values())
    complete = len(all_rows) == len(sample) * len(MODELS)
    print(f"\nSaved {RESULTS}: {len(all_rows)}/{len(sample) * len(MODELS)} cells complete.")
    if not complete:
        print("Re-run to continue once quota allows." if quota_hit else "More cells remain -- re-run.")
        return

    build_chart(all_rows)


def build_chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model = {m: {"scores": [], "latencies": []} for m in MODELS}
    for r in rows:
        by_model[r["model"]]["scores"].append(float(r["match_score"]))
        by_model[r["model"]]["latencies"].append(float(r["latency_s"]))

    labels = [MODEL_LABELS[m] for m in MODELS]
    colors = ["#1D9E75", "#B4453B", "#378ADD"]
    avg_scores = [sum(by_model[m]["scores"]) / len(by_model[m]["scores"]) for m in MODELS]
    avg_latency = [sum(by_model[m]["latencies"]) / len(by_model[m]["latencies"]) for m in MODELS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    ax = axes[0]
    bars = ax.bar(labels, avg_scores, color=colors, width=0.55)
    for bar, v in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average match score on same-field pairs (%)", fontsize=10.5)
    ax.set_title("Accuracy", fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    bars = ax.bar(labels, avg_latency, color=colors, width=0.55)
    for bar, v in zip(bars, avg_latency):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(avg_latency) * 0.02, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average response time (seconds)", fontsize=10.5)
    ax.set_title("Speed", fontsize=12.5, fontweight="bold")
    ax.set_ylim(0, max(avg_latency) * 1.2)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f"Gemini model variants: accuracy vs. speed (n={len(by_model[MODELS[0]]['scores'])} resumes each)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "model_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nchart -> {out}")

    print("\n=== Model comparison ===")
    for m, label in zip(MODELS, labels):
        avg_s = sum(by_model[m]["scores"]) / len(by_model[m]["scores"])
        avg_l = sum(by_model[m]["latencies"]) / len(by_model[m]["latencies"])
        print(f"  {label.replace(chr(10), ' '):28s} avg_score={avg_s:5.1f}%  avg_latency={avg_l:5.1f}s")


if __name__ == "__main__":
    main()
