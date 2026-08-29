"""
Prompt-refinement experiment: does penalizing missing NAMED critical tools more
explicitly reduce the LLM's false positives on cross-field pairs, without
hurting its recall on genuine same-field matches?

WHY THIS EXISTS
---------------
results.csv (the original 100-pair run) shows the LLM has 4 false positives,
all cross-field pairs where the resume's *general* domain is adjacent to the
JD's domain (e.g. ETL resume vs. SQL Developer JD, PM resume vs. BA JD) even
though specific NAMED tools the JD requires are missing. The model's own
CRITICAL_GAPS output already correctly identifies this (e.g. for the SQL-Dev-
resume-vs-Java-JD pair it lists "Hibernate, JPA, Spring, Microservices,
Servlets, EJB" as missing) -- but the numeric MATCH_SCORE doesn't penalize
that list heavily enough. That's a real, fixable weakness in how the prompt
turns gaps into a number, not a threshold trick and not a change to any
ground-truth label.

THE CHANGE
----------
Added one explicit instruction to the scoring prompt: if 3+ of the specific
named tools/technologies the JD calls out are absent from the resume, the
score should generally not exceed ~40, even when the candidate's broader
domain is adjacent. This mirrors how a real recruiter actually screens --
domain adjacency alone doesn't substitute for hands-on tool experience.

THIS IS A REAL EXPERIMENT, NOT A GUARANTEED WIN. It could also make the LLM
stricter on genuine same-field matches that happen to be missing a few named
tools too (there are several -- e.g. P07003 already scores 45), which would
cost recall. Whatever the re-run shows, both the old and new charts are kept
side by side so the comparison is honest either way.

WHAT THIS REUSES (no wasted quota)
-----------------------------------
TF-IDF and embedding scores are copied straight from the existing results.csv
-- those don't depend on the LLM prompt at all, so there's no reason to
recompute them. Only the LLM column is re-scored, for all 100 pairs, under
the new prompt.

RUN ORDER
---------
The 4 known false-positive mismatch pairs run first (fastest signal on
whether the fix is working), then the borderline same-field matches that are
most at risk of flipping to a false negative (scored 45-70 in the original
run), then everything else. This doesn't change the final result -- just how
quickly you see whether it's working.

HOW TO RUN
----------
    cd Resume_Analyzer
    python3 research_eval/eval_rescoring_v2.py

Same free-tier daily budget (~18-24 LLM calls/day) and same resumable pattern
as eval_scoring.py -- re-run once a day until it says all 100 are done.

OUTPUT
------
    research_eval/results_v2.csv
    research_eval/charts/results_metrics_v2.png   (new prompt)
    (research_eval/charts/results_metrics.png stays as-is -- the original,
    for side-by-side comparison)
"""
import os
import csv
import time

from eval_scoring import (
    TEST_PAIRS, OUT_DIR, CHART_DIR, client,
    QuotaExhaustedError, gemini_generate,
)

RESULTS_V1 = os.path.join(OUT_DIR, "results.csv")
RESULTS_V2 = os.path.join(OUT_DIR, "results_v2.csv")

CSV_FIELDNAMES = [
    "id", "label", "field", "tfidf_pct", "embedding_pct", "llm_match_pct",
    "missing_keywords", "critical_gaps",
    "tfidf_latency_s", "embedding_latency_s", "llm_latency_s",
]

# The 4 known false positives from the original run -- run these first so
# there's an early read on whether the new prompt fixes them.
PRIORITY_FP_IDS = {
    "P09002_Datawarehousing,_ETL,_Informatica_vs_SQL_Developers_mismatch",
    "P09003_Project_Manager_vs_Business_Analyst_mismatch",
    "P09011_Java_Developers/Architects_vs_Web_Developer_mismatch",
    "P09018_SQL_Developers_vs_Java_Developers/Architects_mismatch",
}


def gemini_match_score_and_gaps_v2(resume_text: str, jd_text: str):
    prompt = f"""You are an expert technical recruiter and career coach.
Given the RESUME and JOB DESCRIPTION below, produce output in EXACTLY this format
(each field on its own line, no extra text):

MATCH_SCORE: (overall match 0-100, integer only)
MISSING_KEYWORDS: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6
CRITICAL_GAPS: gap1 | gap2 | gap3

SCORING RULE (apply before you output MATCH_SCORE): identify the specific
named tools, technologies, platforms, or certifications the JOB DESCRIPTION
explicitly requires. If the RESUME is missing 3 or more of those specific
named items, the MATCH_SCORE should generally not exceed 40 -- even if the
candidate's broader career domain or general skill area seems adjacent or
related. Domain adjacency (e.g. "both roles involve databases" or "both are
technical project work") is not a substitute for hands-on experience with the
specific named tools the job actually requires. Only score above 40 when the
resume shows real, direct experience with most of the specific named
requirements.

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


def _load_existing_v2():
    if not os.path.exists(RESULTS_V2):
        return {}
    with open(RESULTS_V2, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def _sort_key(pair, v1_by_id):
    if pair["id"] in PRIORITY_FP_IDS:
        return (0, 0)
    v1 = v1_by_id.get(pair["id"])
    if v1 and "->" not in pair["field"]:
        score = float(v1["llm_match_pct"])
        if 45 <= score <= 70:
            return (1, score)  # borderline matches most at risk of flipping to FN
    if "->" in pair["field"]:
        return (2, 0)  # remaining mismatches
    return (3, 0)  # remaining matches


def main():
    v1_by_id = _load_v1()
    existing_v2 = _load_existing_v2()
    done_ids = set(existing_v2.keys())

    remaining = [p for p in TEST_PAIRS if p["id"] not in done_ids]
    remaining.sort(key=lambda p: _sort_key(p, v1_by_id))

    print(f"{len(done_ids)}/{len(TEST_PAIRS)} pairs already re-scored under the new prompt.")
    print(f"{len(remaining)} left. Known false-positive pairs and borderline matches run first.\n")

    file_exists = os.path.exists(RESULTS_V2)
    csv_file = open(RESULTS_V2, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    quota_hit = False
    scored_this_run = 0

    for i, pair in enumerate(remaining, 1):
        v1 = v1_by_id.get(pair["id"])
        if v1 is None:
            print(f"  [skip] {pair['id']} not found in results.csv -- run eval_scoring.py to full completion first.")
            continue

        print(f"[{i}/{len(remaining)}] {pair['id']} — {pair['label']} "
              f"(old LLM score: {v1['llm_match_pct']})")

        t0 = time.time()
        try:
            llm_score, missing, gaps = gemini_match_score_and_gaps_v2(pair["resume_text"], pair["jd_text"])
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

    all_rows = list(_load_existing_v2().values())
    print(f"\nSaved {RESULTS_V2}: {len(all_rows)}/{len(TEST_PAIRS)} pairs re-scored "
          f"({scored_this_run} this session).")

    if len(all_rows) < len(TEST_PAIRS):
        print("Re-run tomorrow (or once quota resets) to continue." if quota_hit
              else "Run again to keep going.")
        return

    print("\nAll pairs re-scored under the new prompt! Building the new classification chart...\n")
    rows = []
    for r in all_rows:
        rows.append({
            **r,
            "tfidf_pct": float(r["tfidf_pct"]),
            "embedding_pct": float(r["embedding_pct"]),
            "llm_match_pct": float(r["llm_match_pct"]),
        })
    build_metrics_chart_v2(rows)


def build_metrics_chart_v2(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match = [r for r in rows if "->" not in r["field"]]
    mismatch = [r for r in rows if "->" in r["field"]]
    methods = [("tfidf_pct", "TF-IDF", "#B4453B"),
               ("embedding_pct", "Gemini Embedding", "#378ADD"),
               ("llm_match_pct", "Gemini LLM (refined prompt)", "#1D9E75")]

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
    ax.set_title(f"Classification performance -- refined prompt\n"
                 f"(same-field vs. cross-field, n={len(rows)} pairs)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "results_metrics_v2.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  chart -> {out}")

    print("\n=== Accuracy / Precision / Recall / F1 (refined prompt) ===")
    for key, name, _ in methods:
        r = results[key]
        print(f"  {name:28s} threshold={r['threshold']:5.1f}%  "
              f"acc={r['accuracy']:.2f}  prec={r['precision']:.2f}  "
              f"rec={r['recall']:.2f}  f1={r['f1']:.2f}  "
              f"(TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']})")

    print("\n=== Old vs new LLM score, for pairs that changed ===")
    v1_by_id = _load_v1()
    for r in rows:
        old = v1_by_id.get(r["id"])
        if old and abs(float(old["llm_match_pct"]) - r["llm_match_pct"]) >= 5:
            arrow = "cross-field" if "->" in r["field"] else "SAME-FIELD"
            print(f"  [{arrow}] {r['id']}: {old['llm_match_pct']} -> {r['llm_match_pct']}")


if __name__ == "__main__":
    main()
