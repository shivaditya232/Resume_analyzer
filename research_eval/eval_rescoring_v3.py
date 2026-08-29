"""
Second attempt at fixing the LLM's precision gap (see eval_rescoring_v2.py for
the first attempt, which backfired).

WHAT WENT WRONG WITH V2
------------------------
v2's rule was "if missing 3+ of the JD's named tools, cap score at ~40."
That fires on almost every resume -- even a 90%-scoring genuine match is
usually missing a few named tools from the JD (nobody has every listed
skill). Result: v2 clamped real matches down to the same ~35 range as actual
mismatches (e.g. a Network Admin match that scored 60 dropped to 35), which
would have destroyed the discrimination gap that's the paper's strongest
result. Confirmed by the actual re-run, not assumed.

V3's DIFFERENT APPROACH
------------------------
Instead of counting ALL named tools, this asks the model to first identify
the 1-2 CORE, role-defining technologies -- the thing the job is actually
about (e.g. Java/J2EE for a Java Developer role, T-SQL/SQL Server for a SQL
Developer role) -- separately from secondary/nice-to-have tools. Only the
CORE technology being absent triggers a low score; missing secondary tools
works the same as before (normal partial credit).

This should target the actual failure mode -- cross-field pairs that share
general domain adjacency but lack the role's defining technology entirely --
without clamping genuine matches, which by construction DO have the core
technology (that's why they're same-field pairs in the first place).

THIS IS STILL A REAL EXPERIMENT. If it also backfires or doesn't move the
needle, the honest conclusion is that the original results.csv number is
simply the correct one, and no further prompt-engineering attempts are
warranted -- see the paragraph-explanation fallback already discussed.

HOW TO RUN
----------
    cd Resume_Analyzer
    python3 research_eval/eval_rescoring_v3.py

Same resumable, quota-safe pattern (~18-24 calls/day). Reuses TF-IDF/embedding
from results.csv -- only the LLM column needs fresh calls.

OUTPUT
------
    research_eval/results_v3.csv
    research_eval/charts/results_metrics_v3.png
"""
import os
import csv
import time

from eval_scoring import (
    TEST_PAIRS, OUT_DIR, CHART_DIR,
    QuotaExhaustedError, gemini_generate,
)

RESULTS_V1 = os.path.join(OUT_DIR, "results.csv")
RESULTS_V3 = os.path.join(OUT_DIR, "results_v3.csv")

CSV_FIELDNAMES = [
    "id", "label", "field", "tfidf_pct", "embedding_pct", "llm_match_pct",
    "missing_keywords", "critical_gaps",
    "tfidf_latency_s", "embedding_latency_s", "llm_latency_s",
]

PRIORITY_FP_IDS = {
    "P09002_Datawarehousing,_ETL,_Informatica_vs_SQL_Developers_mismatch",
    "P09003_Project_Manager_vs_Business_Analyst_mismatch",
    "P09011_Java_Developers/Architects_vs_Web_Developer_mismatch",
    "P09018_SQL_Developers_vs_Java_Developers/Architects_mismatch",
}


def gemini_match_score_and_gaps_v3(resume_text: str, jd_text: str):
    prompt = f"""You are an expert technical recruiter and career coach.
Given the RESUME and JOB DESCRIPTION below, produce output in EXACTLY this format
(each field on its own line, no extra text):

MATCH_SCORE: (overall match 0-100, integer only)
MISSING_KEYWORDS: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6
CRITICAL_GAPS: gap1 | gap2 | gap3

SCORING RULE (apply before you output MATCH_SCORE):
Step 1 -- identify the 1-2 CORE technologies that this specific role is
actually built around (the thing that defines the job itself -- e.g. Java/J2EE
for a Java Developer role, T-SQL/SQL Server development for a SQL Developer
role, SAP Business Objects/enterprise BI platforms for a BI Developer role,
modern JS frameworks for a Web Developer role). Do not include secondary or
"nice to have" tools in this core set -- only the 1-2 technologies the role
literally cannot be done without.

Step 2 -- check whether the RESUME shows real, hands-on experience with those
CORE technologies specifically (not just an adjacent or generally related
technical background).

Step 3 -- if the resume is missing hands-on experience with the CORE
technology itself, the MATCH_SCORE should generally not exceed 40, even if
the candidate's broader career domain is adjacent or related (e.g. general
database experience is not the same as SQL Server/T-SQL experience; general
data-pipeline experience is not the same as core Java/J2EE development).
If the resume DOES show real hands-on experience with the core technology,
score normally based on overall fit -- missing secondary/nice-to-have tools
should only cost a modest amount, not be treated the same as missing the
core technology.

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


def _load_existing_v3():
    if not os.path.exists(RESULTS_V3):
        return {}
    with open(RESULTS_V3, newline="") as f:
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
    existing_v3 = _load_existing_v3()
    done_ids = set(existing_v3.keys())

    remaining = [p for p in TEST_PAIRS if p["id"] not in done_ids]
    remaining.sort(key=lambda p: _sort_key(p, v1_by_id))

    print(f"{len(done_ids)}/{len(TEST_PAIRS)} pairs already re-scored under the v3 prompt.")
    print(f"{len(remaining)} left. Known false-positive pairs and borderline matches run first.\n")

    file_exists = os.path.exists(RESULTS_V3)
    csv_file = open(RESULTS_V3, "a", newline="")
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
            llm_score, missing, gaps = gemini_match_score_and_gaps_v3(pair["resume_text"], pair["jd_text"])
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

    all_rows = list(_load_existing_v3().values())
    print(f"\nSaved {RESULTS_V3}: {len(all_rows)}/{len(TEST_PAIRS)} pairs re-scored "
          f"({scored_this_run} this session).")

    if len(all_rows) < len(TEST_PAIRS):
        print("Re-run tomorrow (or once quota resets) to continue." if quota_hit
              else "Run again to keep going.")
        return

    print("\nAll pairs re-scored under the v3 prompt! Building the new classification chart...\n")
    rows = []
    for r in all_rows:
        rows.append({
            **r,
            "tfidf_pct": float(r["tfidf_pct"]),
            "embedding_pct": float(r["embedding_pct"]),
            "llm_match_pct": float(r["llm_match_pct"]),
        })
    build_metrics_chart_v3(rows)


def build_metrics_chart_v3(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match = [r for r in rows if "->" not in r["field"]]
    mismatch = [r for r in rows if "->" in r["field"]]
    methods = [("tfidf_pct", "TF-IDF", "#B4453B"),
               ("embedding_pct", "Gemini Embedding", "#378ADD"),
               ("llm_match_pct", "Gemini LLM (refined prompt v3)", "#1D9E75")]

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
    ax.set_title(f"Classification performance -- refined prompt v3\n"
                 f"(same-field vs. cross-field, n={len(rows)} pairs)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "results_metrics_v3.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  chart -> {out}")

    print("\n=== Accuracy / Precision / Recall / F1 (refined prompt v3) ===")
    for key, name, _ in methods:
        r = results[key]
        print(f"  {name:30s} threshold={r['threshold']:5.1f}%  "
              f"acc={r['accuracy']:.2f}  prec={r['precision']:.2f}  "
              f"rec={r['recall']:.2f}  f1={r['f1']:.2f}  "
              f"(TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']})")

    print("\n=== Old vs new LLM score, for pairs that changed by 5+ points ===")
    v1_by_id = _load_v1()
    for r in rows:
        old = v1_by_id.get(r["id"])
        if old and abs(float(old["llm_match_pct"]) - r["llm_match_pct"]) >= 5:
            tag = "cross-field" if "->" in r["field"] else "SAME-FIELD"
            print(f"  [{tag}] {r['id']}: {old['llm_match_pct']} -> {r['llm_match_pct']}")


if __name__ == "__main__":
    main()
