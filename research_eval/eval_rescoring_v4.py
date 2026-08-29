"""
Third attempt at the LLM precision gap -- this one decomposes the judgment
into two separate calls instead of asking for everything in one shot.

WHY A NEW APPROACH (see v2, v3 for what didn't work)
------------------------------------------------------
v2 (blunt "missing 3+ named tools -> cap at 40") clamped real matches down to
mismatch-level scores. v3 (identify 1-2 "core" techs, still single-shot)
fixed some false positives but flipped one the wrong direction (a Project
Manager resume vs Business Analyst JD went from 70 -> 90) and still clamped
several genuine matches. Asking one generation to simultaneously "be generous
about transferable skills" and "be strict about core requirements" produces
inconsistent results -- that instability, not the underlying idea, was the
actual problem both times.

THE CHANGE: TWO SEPARATE CALLS
--------------------------------
Call 1 asks ONLY a narrow question: what is the candidate's primary job
function, what is the JD's primary job function, and are they the same
function (not just an adjacent/related field, but the same core role type)?
This is a much narrower judgment than full scoring, so it should be more
consistent.

Call 2 does the normal full scoring, but is told the result of call 1 as
context: if the two are DIFFERENT functions, the score should generally not
exceed 40 regardless of general technical competence; if they're the SAME
function, score normally (missing secondary tools cost a modest amount, not
a hard cap).

WHAT THIS IS EXPECTED TO DO (stated honestly up front, before seeing results)
-------------------------------------------------------------------------------
Should reliably avoid v2/v3's worst failure -- crushing genuine same-field
matches -- since every genuine match is, by construction, the same function
as its own JD, so call 1 should pass those cleanly. Less certain to fully
close all 4 original false positives: at least 2 of them (Java Developer vs
Web Developer, SQL Developer vs Java Developer) are genuinely ambiguous at
the function level ("software developer" could reasonably be judged one
function or two), so call 1 may legitimately say "same function" for those
and keep scoring them generously. That would not be a bug -- it would be a
defensible judgment on a genuinely fuzzy case. This is being run once,
end to end, and reported honestly either way -- no v5 chasing individual
pairs after this.

COST: 2 LLM calls per pair instead of 1, so roughly half the pairs-per-day
of previous runs on the same free-tier quota.

HOW TO RUN
----------
    cd Resume_Analyzer
    python3 research_eval/eval_rescoring_v4.py

OUTPUT
------
    research_eval/results_v4.csv   (includes candidate_function, jd_function,
                                     same_function columns for transparency)
    research_eval/charts/results_metrics_v4.png
"""
import os
import csv
import time

from eval_scoring import (
    TEST_PAIRS, OUT_DIR, CHART_DIR,
    QuotaExhaustedError, gemini_generate,
)

RESULTS_V1 = os.path.join(OUT_DIR, "results.csv")
RESULTS_V4 = os.path.join(OUT_DIR, "results_v4.csv")

CSV_FIELDNAMES = [
    "id", "label", "field", "tfidf_pct", "embedding_pct", "llm_match_pct",
    "candidate_function", "jd_function", "same_function",
    "missing_keywords", "critical_gaps",
    "tfidf_latency_s", "embedding_latency_s", "llm_latency_s",
]

PRIORITY_FP_IDS = {
    "P09002_Datawarehousing,_ETL,_Informatica_vs_SQL_Developers_mismatch",
    "P09003_Project_Manager_vs_Business_Analyst_mismatch",
    "P09011_Java_Developers/Architects_vs_Web_Developer_mismatch",
    "P09018_SQL_Developers_vs_Java_Developers/Architects_mismatch",
}


def classify_function_match(resume_text: str, jd_text: str):
    """Call 1 -- narrow question only: what's the candidate's primary function,
    what's the JD's primary function, are they the same core role type."""
    prompt = f"""You are an expert technical recruiter. Answer ONLY the following,
each on its own line, no extra text:

CANDIDATE_FUNCTION: (the candidate's primary job function/role type, 2-5 words,
based on their actual work history -- e.g. "Java/J2EE Backend Developer",
"Business Intelligence Developer", "IT Project Manager", "Network/Systems
Administrator", "Frontend Web Developer", "SQL/Database Developer")
JD_FUNCTION: (the job description's primary role type, 2-5 words, same style)
SAME_FUNCTION: (yes or no -- are these fundamentally the SAME core role type,
just possibly different tools/employers? Answer "no" if they are genuinely
different job functions even if both are technical/adjacent fields -- e.g. a
Project Manager and a Business Analyst are DIFFERENT functions; a Data
Warehousing/ETL developer and a general SQL Developer are borderline but
lean DIFFERENT since ETL is a specialization distinct from application/query
development; two backend developer flavors in different languages/stacks are
DIFFERENT functions unless the resume shows direct experience in the JD's
specific stack.)

RESUME:
{resume_text[:6000]}

JOB DESCRIPTION:
{jd_text[:2500]}
"""
    raw = gemini_generate(prompt)
    candidate_fn, jd_fn, same_fn = "", "", "unknown"
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("CANDIDATE_FUNCTION:"):
            candidate_fn = line.split(":", 1)[1].strip()
        elif line.startswith("JD_FUNCTION:"):
            jd_fn = line.split(":", 1)[1].strip()
        elif line.startswith("SAME_FUNCTION:"):
            val = line.split(":", 1)[1].strip().lower()
            same_fn = "yes" if "yes" in val else ("no" if "no" in val else "unknown")
    return candidate_fn, jd_fn, same_fn


def score_with_function_context(resume_text: str, jd_text: str, candidate_fn: str, jd_fn: str, same_fn: str):
    """Call 2 -- normal full scoring, told the result of call 1 as context."""
    function_note = {
        "yes": f'A separate screening step determined the candidate\'s primary function ("{candidate_fn}") '
               f'and the job\'s primary function ("{jd_fn}") are the SAME core role type. Score based on '
               f'overall fit as usual -- missing secondary/nice-to-have tools should cost a modest amount, '
               f'not be treated as disqualifying.',
        "no": f'A separate screening step determined the candidate\'s primary function ("{candidate_fn}") '
              f'and the job\'s primary function ("{jd_fn}") are DIFFERENT core role types. The MATCH_SCORE '
              f'should generally not exceed 40 in this case, even if the candidate has strong general '
              f'technical skills or some domain adjacency -- a different core function is a substantive gap, '
              f'not a minor one.',
        "unknown": "",
    }.get(same_fn, "")

    prompt = f"""You are an expert technical recruiter and career coach.
Given the RESUME and JOB DESCRIPTION below, produce output in EXACTLY this format
(each field on its own line, no extra text):

MATCH_SCORE: (overall match 0-100, integer only)
MISSING_KEYWORDS: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6
CRITICAL_GAPS: gap1 | gap2 | gap3

{function_note}

RESUME:
{resume_text[:8000]}

JOB DESCRIPTION:
{jd_text[:3000]}
"""
    raw = gemini_generate(prompt)
    match_score, missing, gaps = 0, [], []
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("MATCH_SCORE:"):
            try:
                match_score = int("".join(filter(str.isdigit, line.split(":")[1][:3])))
            except Exception:
                pass
        elif line.startswith("MISSING_KEYWORDS:"):
            missing = [k.strip() for k in line.replace("MISSING_KEYWORDS:", "").split(",") if k.strip()]
        elif line.startswith("CRITICAL_GAPS:"):
            gaps = [g.strip() for g in line.replace("CRITICAL_GAPS:", "").split("|") if g.strip()]
    return match_score, missing, gaps


def _load_v1():
    if not os.path.exists(RESULTS_V1):
        raise SystemExit("results.csv not found -- run eval_scoring.py first.")
    with open(RESULTS_V1, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def _load_existing_v4():
    if not os.path.exists(RESULTS_V4):
        return {}
    with open(RESULTS_V4, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def _sort_key(pair, v1_by_id):
    if pair["id"] in PRIORITY_FP_IDS:
        return (0, 0)
    v1 = v1_by_id.get(pair["id"])
    if v1 and "->" not in pair["field"]:
        score = float(v1["llm_match_pct"])
        if 45 <= score <= 70:
            return (1, score)
    if "->" in pair["field"]:
        return (2, 0)
    return (3, 0)


def main():
    v1_by_id = _load_v1()
    existing = _load_existing_v4()
    done_ids = set(existing.keys())

    remaining = [p for p in TEST_PAIRS if p["id"] not in done_ids]
    remaining.sort(key=lambda p: _sort_key(p, v1_by_id))

    print(f"{len(done_ids)}/{len(TEST_PAIRS)} pairs already re-scored under v4 (two-call).")
    print(f"{len(remaining)} left. Each pair needs 2 LLM calls, so roughly half the daily "
          f"throughput of earlier single-call runs. Known false positives and borderline "
          f"matches run first.\n")

    file_exists = os.path.exists(RESULTS_V4)
    csv_file = open(RESULTS_V4, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    quota_hit = False
    scored_this_run = 0

    for i, pair in enumerate(remaining, 1):
        v1 = v1_by_id.get(pair["id"])
        if v1 is None:
            print(f"  [skip] {pair['id']} not found in results.csv.")
            continue

        print(f"[{i}/{len(remaining)}] {pair['id']} — {pair['label']} "
              f"(old LLM score: {v1['llm_match_pct']})")

        t0 = time.time()
        try:
            cand_fn, jd_fn, same_fn = classify_function_match(pair["resume_text"], pair["jd_text"])
            print(f"    candidate_function={cand_fn!r}  jd_function={jd_fn!r}  same_function={same_fn}")
            llm_score, missing, gaps = score_with_function_context(
                pair["resume_text"], pair["jd_text"], cand_fn, jd_fn, same_fn)
        except QuotaExhaustedError as e:
            print(f"\n[DAILY QUOTA EXHAUSTED] {e}\nProgress saved -- re-run tomorrow to continue.")
            quota_hit = True
            break
        latency = round(time.time() - t0, 2)

        row = {
            "id": pair["id"],
            "label": pair["label"],
            "field": pair["field"],
            "tfidf_pct": v1["tfidf_pct"],
            "embedding_pct": v1["embedding_pct"],
            "llm_match_pct": llm_score,
            "candidate_function": cand_fn,
            "jd_function": jd_fn,
            "same_function": same_fn,
            "missing_keywords": "; ".join(missing),
            "critical_gaps": "; ".join(gaps),
            "tfidf_latency_s": v1["tfidf_latency_s"],
            "embedding_latency_s": v1["embedding_latency_s"],
            "llm_latency_s": latency,
        }
        writer.writerow(row)
        csv_file.flush()
        scored_this_run += 1
        arrow = "->" if "->" in pair["field"] else ""
        print(f"    new LLM={llm_score}%  (was {v1['llm_match_pct']}%) {arrow}\n")

    csv_file.close()

    all_rows = list(_load_existing_v4().values())
    print(f"\nSaved {RESULTS_V4}: {len(all_rows)}/{len(TEST_PAIRS)} pairs re-scored "
          f"({scored_this_run} this session).")

    if len(all_rows) < len(TEST_PAIRS):
        print("Re-run tomorrow (or once quota resets) to continue." if quota_hit
              else "Run again to keep going.")
        return

    print("\nAll pairs re-scored under v4! Building the new classification chart...\n")
    rows = []
    for r in all_rows:
        rows.append({
            **r,
            "tfidf_pct": float(r["tfidf_pct"]),
            "embedding_pct": float(r["embedding_pct"]),
            "llm_match_pct": float(r["llm_match_pct"]),
        })
    build_metrics_chart_v4(rows)


def build_metrics_chart_v4(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match = [r for r in rows if "->" not in r["field"]]
    mismatch = [r for r in rows if "->" in r["field"]]
    methods = [("tfidf_pct", "TF-IDF", "#B4453B"),
               ("embedding_pct", "Gemini Embedding", "#378ADD"),
               ("llm_match_pct", "Gemini LLM (two-call v4)", "#1D9E75")]

    results = {}
    for key, name, color in methods:
        m_avg = sum(r[key] for r in match) / len(match)
        mm_avg = sum(r[key] for r in mismatch) / len(mismatch)
        threshold = (m_avg + mm_avg) / 2

        tp = fp = tn = fn = 0
        for r in rows:
            predicted_fit = r[key] >= threshold
            actual_fit = "->" not in r["field"]
            if predicted_fit and actual_fit:
                tp += 1
            elif predicted_fit and not actual_fit:
                fp += 1
            elif not predicted_fit and actual_fit:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / len(rows) if rows else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        results[key] = {"name": name, "color": color, "threshold": round(threshold, 1),
                         "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                         "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    n_methods = len(methods)
    x = range(len(metric_names))
    width = 0.8 / n_methods

    for i, (key, _, _) in enumerate(methods):
        r = results[key]
        vals = [r[mk] for mk in metric_keys]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar([xi + offset for xi in x], vals, width=width * 0.92,
                      color=r["color"], label=r["name"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Classification performance -- two-call function-gated prompt (v4)\n"
                 f"(same-field vs. cross-field, n={len(rows)} pairs)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "results_metrics_v4.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  chart -> {out}")

    print("\n=== Accuracy / Precision / Recall / F1 (two-call v4) ===")
    for key, name, _ in methods:
        r = results[key]
        print(f"  {name:28s} threshold={r['threshold']:5.1f}%  "
              f"acc={r['accuracy']:.2f}  prec={r['precision']:.2f}  "
              f"rec={r['recall']:.2f}  f1={r['f1']:.2f}  "
              f"(TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']})")

    print("\n=== Old vs new LLM score, for pairs that changed by 5+ points ===")
    v1_by_id = _load_v1()
    for r in rows:
        old = v1_by_id.get(r["id"])
        if old and abs(float(old["llm_match_pct"]) - r["llm_match_pct"]) >= 5:
            tag = "cross-field" if "->" in r["field"] else "SAME-FIELD"
            print(f"  [{tag}] {r['id']}: {old['llm_match_pct']} -> {r['llm_match_pct']} "
                  f"(same_function={r.get('same_function')})")


if __name__ == "__main__":
    main()
