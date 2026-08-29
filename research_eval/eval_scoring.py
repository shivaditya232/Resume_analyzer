"""
Standalone evaluation script for the Resume Suite research paper.

Computes THREE scoring methods on real resume/JD pairs (100 total: 80 same-field
+ 20 cross-field mismatches) and compares them:
  1. TF-IDF + cosine similarity   -- classical keyword baseline, no API needed
  2. Gemini embedding cosine similarity -- same method app.py uses (semantic, no LLM reasoning)
  3. Gemini LLM match score        -- full contextual reasoning, same prompt shape as app.py

Pairs are built from the Kaggle "Resume dataset.csv" (haidermaseeh/resume-dataset --
9,000 real resumes) via large_test_dataset.py, across 8 job categories (Recruiter
dropped -- essentially absent from this dataset, see category_jds.py). Resumes are
verified by their own job_title field against each category's real job type
(NOT the CSV's own "category" column, which was found to be unreliable -- e.g.
most rows labeled "Web Developer" are actually ETL/Data-Warehousing resumes).
N_PER_CATEGORY real resumes per category are paired against a hand-written JD for
that category (category_jds.py) -- same-field, expect a good match -- plus
N_CROSS_FIELD deliberate cross-field pairs (resume from one category vs. JD from
another) to test whether the app correctly scores mismatches lower.

This is what turns "we built a system" into "we validated a system" for the paper --
it's the evidence behind Result graphs B (descriptive output) and C (baseline comparison)
from the plan.

HOW TO RUN
----------
    cd Resume_Analyzer
    python research_eval/eval_scoring.py

Requires the same .env as app.py (GEMINI_API_KEY already present), and
"Resume dataset.csv" to be present in research_eval/ (already downloaded). Needs
outbound network access to Google's Generative Language API -- run this on your
own machine where the Streamlit app already works, not inside a network-restricted
sandbox. With ~100 pairs x (1 embedding call + 1 LLM call) and a free-tier daily cap
of ~18-20 LLM calls, this takes several days to fully complete -- just re-run the
script once a day (or whenever your quota resets); it resumes automatically from
results.csv and stops cleanly once the daily budget is used up.

OUTPUT
------
    research_eval/results.csv                        -- raw scores per pair
    research_eval/charts/comparison_grouped.html/.png -- TF-IDF vs Embedding vs LLM, per pair
    research_eval/charts/comparison_average.html/.png -- average of each method, all pairs
    research_eval/charts/score_distribution.html/.png -- histogram of Gemini match scores
    research_eval/charts/gap_frequency.html/.png      -- most frequent skill gaps flagged
    research_eval/charts/latency.html/.png            -- response time per method, per pair

PNG export needs `pip install -U kaleido` (optional -- HTML charts are saved either way
and are what you want for the "nice interactive graph" experience; PNG is what you'll
actually paste into the Word/PDF paper).
"""
import os
import re
import csv
import math
import time
from collections import Counter

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import plotly.graph_objects as go

from large_test_dataset import build_test_pairs

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "Resume dataset.csv")
# Expanded run: 20 per category x 8 categories (160 same-field) + 40 cross-field
# mismatch = 200 total. Raised from the earlier 10/8 (100-pair) run -- thanks to
# the shuffle-once-then-slice logic in large_test_dataset.py, every one of the
# original 100 pairs keeps the exact same id and resume here, so results.csv
# resumes cleanly and only the 100 new pairs (80 same-field + 20 mismatch) get
# scored from here on. Every category's real-resume pool is at least 58 (see
# large_test_dataset.py docstring), so 20/category is safe everywhere.
N_PER_CATEGORY = 20
N_CROSS_FIELD = 40
TEST_PAIRS = build_test_pairs(csv_path=CSV_PATH, n_per_category=N_PER_CATEGORY, n_cross_field=N_CROSS_FIELD)
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ── 1. TF-IDF + cosine similarity — pure python, no sklearn dependency ────────
#
# IDF is computed across the WHOLE corpus (all resumes + all JDs in the test
# set), not per-pair. Per-pair IDF is a corpus-of-2 and mathematically zeroes
# out every term shared between the two documents (log(3/3) = 0) -- exactly
# the terms cosine similarity needs. Corpus-wide IDF is what every TF-IDF
# baseline in the reference papers (e.g. Career Craft Scanner) actually does.

def _tokenize(text: str) -> list:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}", text.lower())


def _build_corpus_idf(pairs) -> dict:
    docs = []
    for p in pairs:
        docs.append(_tokenize(p["resume_text"]))
        docs.append(_tokenize(p["jd_text"]))
    n_docs = len(docs)
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    return {t: math.log((n_docs + 1) / (c + 1)) + 1 for t, c in df.items()}  # smooth idf


def _tfidf_vec(tokens: list, idf: dict) -> dict:
    tf = Counter(tokens)
    length = len(tokens) or 1
    return {t: (c / length) * idf.get(t, 0.0) for t, c in tf.items()}


def tfidf_cosine(resume_text: str, jd_text: str, idf: dict) -> float:
    rv = _tfidf_vec(_tokenize(resume_text), idf)
    jv = _tfidf_vec(_tokenize(jd_text), idf)
    common = set(rv) & set(jv)
    dot = sum(rv[t] * jv[t] for t in common)
    n1 = math.sqrt(sum(x * x for x in rv.values()))
    n2 = math.sqrt(sum(x * x for x in jv.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


# ── 2. Gemini embedding cosine similarity — mirrors app.py's embed_text_cached ─

_embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=GEMINI_API_KEY
)


def embedding_cosine(resume_text: str, jd_text: str, retries: int = 3) -> float:
    for attempt in range(retries):
        try:
            r_vec = np.array(_embedder.embed_query(resume_text[:4000]))
            j_vec = np.array(_embedder.embed_query(jd_text[:3000]))
            denom = np.linalg.norm(r_vec) * np.linalg.norm(j_vec)
            return float(np.dot(r_vec, j_vec) / denom) if denom else 0.0
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(20 * (attempt + 1))
            else:
                print(f"    [embedding failed after retries] {e}")
                return 0.0
    return 0.0


# ── 3. Gemini LLM match score — same prompt shape as app.py's analyse_gap_cached ─

class QuotaExhaustedError(Exception):
    """Raised when Google's PER-DAY free-tier quota is hit -- retrying won't help
    until the quota resets, so callers should stop the whole run immediately
    instead of burning time retrying every remaining pair."""
    pass


def gemini_generate(prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return response.text.strip()
        except Exception as e:
            err = str(e)
            daily_quota = "RESOURCE_EXHAUSTED" in err and ("PerDay" in err or "per day" in err.lower())
            if daily_quota:
                # No point retrying -- this resets tomorrow, not in 30/60/90 seconds.
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


def gemini_match_score_and_gaps(resume_text: str, jd_text: str):
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


# ── run evaluation ──────────────────────────────────────────────────────────

CSV_FIELDNAMES = [
    "id", "label", "field", "tfidf_pct", "embedding_pct", "llm_match_pct",
    "missing_keywords", "critical_gaps",
    "tfidf_latency_s", "embedding_latency_s", "llm_latency_s",
]

# Free-tier daily cap for gemini-2.5-flash generate_content is nominally 20/day,
# but observed behavior let ~22-24 calls through before failing. Rather than guess,
# don't artificially stop early -- just keep going until the real QuotaExhaustedError
# hits, and stop cleanly right then. This constant is just an upper safety ceiling
# in case something else changes; it should never be the reason a run stops.
DAILY_LLM_BUDGET = len(TEST_PAIRS)


def _load_existing_results(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    csv_path = os.path.join(OUT_DIR, "results.csv")
    existing_rows = _load_existing_results(csv_path)
    done_ids = {r["id"] for r in existing_rows}

    remaining_pairs = [p for p in TEST_PAIRS if p["id"] not in done_ids]

    # Cross-field mismatch pairs are listed LAST in TEST_PAIRS (built after all
    # same-field pairs), so a plain in-order resume would always burn the daily
    # quota on leftover same-field pairs before ever reaching them -- exactly
    # what starved the mismatch class to 8/20 while same-field reached 59/80.
    # Prioritize whatever mismatch pairs are still unscored so each day's quota
    # actually closes that gap; same-field pairs still fill in afterward with
    # whatever quota remains.
    remaining_pairs.sort(key=lambda p: 0 if "mismatch" in p["id"] else 1)

    n_mismatch_left = sum(1 for p in remaining_pairs if "mismatch" in p["id"])
    print(f"{len(done_ids)}/{len(TEST_PAIRS)} pairs already scored (resuming from results.csv).")
    print(f"{len(remaining_pairs)} pairs left ({n_mismatch_left} cross-field mismatch, prioritized first). "
          f"Running up to {DAILY_LLM_BUDGET} of them this session (free-tier daily LLM budget)...\n")

    idf = _build_corpus_idf(TEST_PAIRS)  # corpus IDF should reflect the FULL set, not just remaining

    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    scored_this_run = 0
    quota_hit = False

    for i, pair in enumerate(remaining_pairs, 1):
        if scored_this_run >= DAILY_LLM_BUDGET:
            print(f"Hit today's budget of {DAILY_LLM_BUDGET} pairs. Stopping cleanly -- "
                  f"run the script again tomorrow to continue.")
            break

        print(f"[{i}/{len(remaining_pairs)} remaining] {pair['id']} — {pair['label']}")

        t0 = time.time()
        tfidf_score = round(tfidf_cosine(pair["resume_text"], pair["jd_text"], idf) * 100, 1)
        t1 = time.time()

        emb_score = round(embedding_cosine(pair["resume_text"], pair["jd_text"]) * 100, 1)
        t2 = time.time()

        try:
            llm_score, missing, gaps = gemini_match_score_and_gaps(pair["resume_text"], pair["jd_text"])
        except QuotaExhaustedError as e:
            print(f"\n[DAILY QUOTA EXHAUSTED] {e}\n"
                  f"Stopping here -- progress is saved in results.csv. "
                  f"Run this script again tomorrow (or whenever your quota resets) to continue "
                  f"from pair {pair['id']}.")
            quota_hit = True
            break
        t3 = time.time()

        row = {
            "id": pair["id"],
            "label": pair["label"],
            "field": pair["field"],
            "tfidf_pct": tfidf_score,
            "embedding_pct": emb_score,
            "llm_match_pct": llm_score,
            "missing_keywords": "; ".join(missing),
            "critical_gaps": "; ".join(gaps),
            "tfidf_latency_s": round(t1 - t0, 3),
            "embedding_latency_s": round(t2 - t1, 2),
            "llm_latency_s": round(t3 - t2, 2),
        }
        writer.writerow(row)
        csv_file.flush()  # persist immediately -- survives Ctrl+C or a crash
        scored_this_run += 1
        print(f"    TF-IDF={tfidf_score}%  Embedding={emb_score}%  LLM={llm_score}%\n")

    csv_file.close()

    all_rows = _load_existing_results(csv_path)
    total_done = len(all_rows)
    print(f"\nSaved {csv_path} ({total_done}/{len(TEST_PAIRS)} pairs scored so far, "
          f"{scored_this_run} scored this session).")

    if total_done < len(TEST_PAIRS):
        remaining_after = len(TEST_PAIRS) - total_done
        reason = "daily quota hit" if quota_hit else "daily budget reached"
        print(f"{remaining_after} pairs still left ({reason}). "
              f"Run `python3 eval_scoring.py` again to keep going -- it will pick up where this left off.")
        print("Charts are only generated once all pairs are scored, so skipping chart generation for now.")
        return

    print("All pairs scored! Building charts from the full results.csv...\n")
    all_gaps = []
    for r in all_rows:
        if r.get("critical_gaps"):
            all_gaps.extend(g.strip() for g in r["critical_gaps"].split(";") if g.strip())
    # cast numeric fields back from strings for chart building
    rows = []
    for r in all_rows:
        rows.append({
            **r,
            "tfidf_pct": float(r["tfidf_pct"]),
            "embedding_pct": float(r["embedding_pct"]),
            "llm_match_pct": float(r["llm_match_pct"]),
            "tfidf_latency_s": float(r["tfidf_latency_s"]),
            "embedding_latency_s": float(r["embedding_latency_s"]),
            "llm_latency_s": float(r["llm_latency_s"]),
        })
    build_charts(rows, all_gaps)
    try:
        build_classification_metrics_chart(rows)
    except ImportError as e:
        print(f"\n(Skipping classification-metrics chart -- {e}. The scored data is saved either way.)")


def save_fig(fig, name: str):
    html_path = os.path.join(CHART_DIR, f"{name}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(os.path.join(CHART_DIR, f"{name}.png"), scale=2, width=1000, height=550)
        print(f"  chart -> {html_path} (+ .png)")
    except Exception:
        print(f"  chart -> {html_path} (install `kaleido` for a .png copy)")


def build_classification_metrics_chart(rows):
    """
    Accuracy / Precision / Recall / F1, treating same-field pairs as the
    positive ("good fit") class and cross-field pairs as the negative class --
    the cleanest ground truth available (real resumes verified against their
    own job_title, hand-written JDs per field), unlike noisy external labels.

    Threshold per method = the midpoint between that method's own average
    same-field score and average cross-field score. Deliberately NOT a single
    fixed cutoff shared across methods -- TF-IDF/embedding/LLM live on very
    different natural scales, so one shared threshold would unfairly penalize
    whichever method's scores simply run lower on an absolute basis. A
    self-calibrated midpoint asks the fair question of every method: does its
    own score for a genuine match sit clearly above its own score for a
    genuine mismatch?
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    match = [r for r in rows if "->" not in r["field"]]
    mismatch = [r for r in rows if "->" in r["field"]]
    methods = [("tfidf_pct", "TF-IDF", "#B4453B"),
               ("embedding_pct", "Gemini Embedding", "#378ADD"),
               ("llm_match_pct", "Gemini LLM", "#1D9E75")]

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
    ax.set_title(f"Classification performance on verified resume-JD pairs\n"
                 f"(same-field vs. cross-field, n={len(rows)} pairs)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "results_metrics.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n  classification metrics chart -> {out}")

    print("\n=== Accuracy / Precision / Recall / F1 (same-field vs. cross-field) ===")
    for key, name, _ in methods:
        r = results[key]
        print(f"  {name:18s} threshold={r['threshold']:5.1f}%  "
              f"acc={r['accuracy']:.2f}  prec={r['precision']:.2f}  "
              f"rec={r['recall']:.2f}  f1={r['f1']:.2f}  "
              f"(TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']})")


def build_charts(rows, all_gaps):
    labels = [r["id"].split("_", 1)[0] for r in rows]  # P01, P02, ... for compact x-axis

    # 1. grouped comparison bar — TF-IDF vs Embedding vs LLM per pair
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name="TF-IDF (keyword baseline)", x=labels,
                           y=[r["tfidf_pct"] for r in rows], marker_color="#B4B2A9"))
    fig1.add_trace(go.Bar(name="Embedding cosine similarity", x=labels,
                           y=[r["embedding_pct"] for r in rows], marker_color="#378ADD"))
    fig1.add_trace(go.Bar(name="Gemini LLM match score", x=labels,
                           y=[r["llm_match_pct"] for r in rows], marker_color="#1D9E75"))
    fig1.update_layout(barmode="group", title="Scoring method comparison per resume/JD pair",
                        yaxis_title="Match score (%)", xaxis_title="Test pair",
                        yaxis_range=[0, 100], height=550, legend=dict(orientation="h", y=-0.2))
    save_fig(fig1, "comparison_grouped")

    # 2. average summary bar
    avg_tfidf = round(sum(r["tfidf_pct"] for r in rows) / len(rows), 1)
    avg_emb = round(sum(r["embedding_pct"] for r in rows) / len(rows), 1)
    avg_llm = round(sum(r["llm_match_pct"] for r in rows) / len(rows), 1)
    fig2 = go.Figure(go.Bar(
        x=["TF-IDF (keyword baseline)", "Embedding cosine similarity", "Gemini LLM match score"],
        y=[avg_tfidf, avg_emb, avg_llm],
        marker_color=["#B4B2A9", "#378ADD", "#1D9E75"],
        text=[f"{avg_tfidf}%", f"{avg_emb}%", f"{avg_llm}%"], textposition="outside",
    ))
    fig2.update_layout(title="Average score by method across all test pairs",
                        yaxis_title="Average match score (%)", yaxis_range=[0, 100], height=450)
    save_fig(fig2, "comparison_average")

    # 3. score distribution histogram
    fig3 = go.Figure(go.Histogram(x=[r["llm_match_pct"] for r in rows], nbinsx=10,
                                   marker_color="#1D9E75"))
    fig3.update_layout(title="Distribution of Gemini match scores across test resumes",
                        xaxis_title="Match score (%)", yaxis_title="Number of resume/JD pairs",
                        height=450)
    save_fig(fig3, "score_distribution")

    # 4. most frequent gaps
    counts = Counter(all_gaps).most_common(10)
    if counts:
        fig4 = go.Figure(go.Bar(x=[c[1] for c in counts], y=[c[0] for c in counts],
                                 orientation="h", marker_color="#EF9F27"))
        fig4.update_layout(title="Most frequently flagged skill gaps across test resumes",
                            xaxis_title="Frequency", height=max(350, 40 * len(counts)))
        save_fig(fig4, "gap_frequency")

    # 5. latency
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(name="TF-IDF", x=labels,
                           y=[r["tfidf_latency_s"] for r in rows], marker_color="#B4B2A9"))
    fig5.add_trace(go.Bar(name="Embedding", x=labels,
                           y=[r["embedding_latency_s"] for r in rows], marker_color="#378ADD"))
    fig5.add_trace(go.Bar(name="LLM analysis", x=labels,
                           y=[r["llm_latency_s"] for r in rows], marker_color="#1D9E75"))
    fig5.update_layout(barmode="stack", title="Response time per method, per resume/JD pair",
                        yaxis_title="Time (seconds)", xaxis_title="Test pair",
                        height=550, legend=dict(orientation="h", y=-0.2))
    save_fig(fig5, "latency")

    print(f"\nDone. Averages — TF-IDF: {avg_tfidf}%  Embedding: {avg_emb}%  Gemini LLM: {avg_llm}%")
    print(f"All charts saved to {CHART_DIR}")


if __name__ == "__main__":
    main()
