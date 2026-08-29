"""
Human-labelled fit evaluation for the Resume Suite research paper.

WHY THIS EXPERIMENT (and why it's genuinely stronger than graph 1)
------------------------------------------------------------------
Our own Kaggle-based evaluation (eval_scoring.py) can only build "mismatches" by
pairing whole different fields, which even simple methods separate. That limits
what it can prove. This experiment uses an EXTERNAL, HUMAN-LABELLED dataset:

    cnamuangtoun/resume-job-description-fit  (Hugging Face, 8k real pairs)
    each pair labelled by a human: "No Fit" / "Potential Fit" / "Good Fit"

Crucially, many "No Fit" pairs are TOPICALLY SIMILAR -- e.g. an accountant resume
against an accounting-firm job, or a software developer against a software-engineer
role -- yet a human judged them not a fit. These are HARD negatives:

  - Embedding cosine similarity keys on shared vocabulary, so it scores these
    topically-similar "No Fit" pairs HIGH -- disagreeing with the human.
  - The Gemini LLM reasons about actual qualification fit, so it can score them
    LOW -- agreeing with the human.

So we measure the thing reviewers actually respect: HOW WELL EACH METHOD'S SCORE
AGREES WITH HUMAN JUDGEMENT. If the LLM's average score climbs cleanly from
No Fit -> Potential Fit -> Good Fit while the baselines stay flat, that flat-vs-
climbing contrast is the result -- on real human ground truth, not synthetic pairs.

SETUP
-----
1. Download the dataset to research_eval/fit_dataset.csv:
       pip3 install datasets --break-system-packages
       python3 -c "from datasets import load_dataset; \
         load_dataset('cnamuangtoun/resume-job-description-fit')['test'].to_csv('research_eval/fit_dataset.csv')"
   (columns: resume_text, job_description_text, label)

2. Run:
       cd Resume_Analyzer
       python3 research_eval/eval_fit.py

Resumable and quota-safe exactly like eval_scoring.py: scores incrementally to
fit_results.csv, stops cleanly on the daily quota, resumes next run. Start small
(N_PER_LABEL below) to confirm the signal, then raise it and re-run to scale.

OUTPUT
------
    research_eval/fit_results.csv
    research_eval/charts/fit_by_label.png   -- avg score per human label, per method
    research_eval/charts/fit_by_label.html
"""
import os
import csv
import time
import random

from eval_scoring import (
    tfidf_cosine, _build_corpus_idf, embedding_cosine,
    gemini_match_score_and_gaps, QuotaExhaustedError,
    OUT_DIR, CHART_DIR,
)

import plotly.graph_objects as go

FIT_CSV = os.path.join(OUT_DIR, "fit_dataset.csv")
RESULTS = os.path.join(OUT_DIR, "fit_results.csv")
RANDOM_SEED = 42

# Pilot size: pairs sampled PER human label (No Fit / Potential Fit / Good Fit).
# 10 -> 30 pairs total, ~1-2 free-tier days. Raise to 30-40 once the signal is
# confirmed to scale toward ~100 for the paper. Resumable, so raising it only
# adds new pairs (see the fixed-seed shuffle below).
N_PER_LABEL = 10

LABELS = ["No Fit", "Potential Fit", "Good Fit"]

CSV_FIELDNAMES = [
    "row_id", "label", "tfidf_pct", "embedding_pct", "llm_match_pct",
    "tfidf_latency_s", "embedding_latency_s", "llm_latency_s",
]


def _load_fit_dataset():
    if not os.path.exists(FIT_CSV):
        return None
    rows = []
    with open(FIT_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for i, r in enumerate(csv.DictReader(f)):
            resume = (r.get("resume_text") or "").strip()
            jd = (r.get("job_description_text") or "").strip()
            label = (r.get("label") or "").strip()
            if resume and jd and label in LABELS:
                rows.append({"row_id": i, "resume_text": resume, "jd_text": jd, "label": label})
    return rows


def _balanced_sample(rows):
    """Fixed-seed, shuffle-once-per-label then take the first N_PER_LABEL, so
    raising N_PER_LABEL later yields a strict superset (already-scored rows keep
    their row_id and are never re-picked)."""
    rng = random.Random(RANDOM_SEED)
    by_label = {lab: [] for lab in LABELS}
    for r in rows:
        by_label[r["label"]].append(r)
    sample = []
    for lab in LABELS:
        pool = by_label[lab]
        rng.shuffle(pool)
        sample.extend(pool[:N_PER_LABEL])
    return sample


def _load_existing():
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS, newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = _load_fit_dataset()
    if rows is None:
        print("fit_dataset.csv not found. Download it first (see the header of this file).")
        return
    print(f"Loaded {len(rows)} labelled pairs from fit_dataset.csv.")

    sample = _balanced_sample(rows)
    idf = _build_corpus_idf(sample)  # corpus IDF over the sampled pairs

    existing = _load_existing()
    done = {r["row_id"] for r in existing}
    remaining = [p for p in sample if str(p["row_id"]) not in done]

    print(f"{len(done)}/{len(sample)} scored. {len(remaining)} left this session "
          f"(each = 1 embedding + 1 LLM call).\n")

    file_exists = os.path.exists(RESULTS)
    f = open(RESULTS, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    quota_hit = False
    for i, p in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] row {p['row_id']} — human label: {p['label']}")
        t0 = time.time()
        tfidf = round(tfidf_cosine(p["resume_text"], p["jd_text"], idf) * 100, 1)
        t1 = time.time()
        emb = round(embedding_cosine(p["resume_text"], p["jd_text"]) * 100, 1)
        t2 = time.time()
        try:
            llm, _missing, _gaps = gemini_match_score_and_gaps(p["resume_text"], p["jd_text"])
        except QuotaExhaustedError as e:
            print(f"\n[DAILY QUOTA EXHAUSTED] {e}\nProgress saved -- re-run tomorrow.")
            quota_hit = True
            break
        t3 = time.time()

        writer.writerow({
            "row_id": p["row_id"], "label": p["label"],
            "tfidf_pct": tfidf, "embedding_pct": emb, "llm_match_pct": llm,
            "tfidf_latency_s": round(t1 - t0, 3),
            "embedding_latency_s": round(t2 - t1, 2),
            "llm_latency_s": round(t3 - t2, 2),
        })
        f.flush()
        print(f"    TF-IDF={tfidf}%  Embedding={emb}%  LLM={llm}%\n")

    f.close()

    all_rows = _load_existing()
    print(f"\nSaved {RESULTS} ({len(all_rows)}/{len(sample)} scored).")

    # Chart-building needs matplotlib, which isn't always installed locally --
    # the CSV data (the part that actually matters) is already saved above
    # regardless, so a missing chart library should never crash the run.
    try:
        # Latency doesn't depend on label balance -- build it from whatever is
        # scored so far, every run, even if the full sample isn't done yet.
        if all_rows:
            build_latency_chart(all_rows)

        # Metrics chart only needs enough No Fit / Good Fit examples (Potential
        # Fit isn't used for binary classification), so it can build before the
        # full 30-pair sample is complete.
        no_fit_n = sum(1 for r in all_rows if r["label"] == "No Fit")
        good_fit_n = sum(1 for r in all_rows if r["label"] == "Good Fit")
        if no_fit_n >= 8 and good_fit_n >= 8:
            build_metrics_chart(all_rows)
    except ImportError as e:
        print(f"\n(Skipping local chart generation -- {e}. "
              f"The scored data above is saved either way.)")
    else:
        print(f"\nMetrics chart (Accuracy/Precision/Recall/F1) needs at least 8 Good Fit "
              f"examples (have {good_fit_n}/{N_PER_LABEL}). Re-run to collect more.")

    if len(all_rows) < len(sample):
        reason = "daily quota hit" if quota_hit else "more pairs remain"
        print(f"\n{len(sample) - len(all_rows)} left ({reason}). Re-run to continue.")
        print("fit_by_label.png builds once all sampled pairs are scored.")
        return

    build_chart(all_rows)


def build_chart(rows):
    def mean_for(key, label):
        vals = [float(r[key]) for r in rows if r["label"] == label]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    methods = [("tfidf_pct", "TF-IDF", "#B4B2A9"),
               ("embedding_pct", "Embedding", "#378ADD"),
               ("llm_match_pct", "Gemini LLM", "#1D9E75")]

    fig = go.Figure()
    for key, name, color in methods:
        fig.add_trace(go.Bar(
            name=name, x=LABELS,
            y=[mean_for(key, lab) for lab in LABELS],
            marker_color=color,
            text=[f"{mean_for(key, lab)}%" for lab in LABELS], textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Average match score vs. human fit label (external labelled dataset)",
        xaxis_title="Human-assigned fit label", yaxis_title="Average match score (%)",
        yaxis_range=[0, 100], height=520, legend=dict(orientation="h", y=-0.2),
    )
    html = os.path.join(CHART_DIR, "fit_by_label.html")
    fig.write_html(html, include_plotlyjs="cdn")
    try:
        fig.write_image(os.path.join(CHART_DIR, "fit_by_label.png"), scale=2, width=1000, height=560)
    except Exception:
        pass

    # spread across labels = how well the method tracks human judgement
    print("\n=== Agreement with human labels (avg score per label) ===")
    for key, name, _ in methods:
        lo, hi = mean_for(key, "No Fit"), mean_for(key, "Good Fit")
        print(f"  {name:11s} No Fit={lo:5.1f}%  Potential={mean_for(key,'Potential Fit'):5.1f}%  "
              f"Good Fit={hi:5.1f}%  (No Fit->Good Fit spread: {round(hi-lo,1)} pts)")
    print(f"\n  chart -> {html} (+ .png)")


def build_latency_chart(rows):
    """
    Average processing time per method, real data (measured in every scoring
    call this script makes), matching the "average time per model" bar-chart
    style used in published papers (e.g. Fig. 6 of the Whisper-model reference
    the user shared: one bar per method, seconds on the y-axis).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [("tfidf_latency_s", "TF-IDF", "#B4453B"),
               ("embedding_latency_s", "Gemini Embedding", "#378ADD"),
               ("llm_latency_s", "Gemini LLM", "#1D9E75")]

    avgs = []
    for key, name, color in methods:
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        avgs.append(sum(vals) / len(vals) if vals else 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    names = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    bars = ax.bar(names, avgs, color=colors, width=0.55)
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(avgs) * 0.02,
                f"{v:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Average processing time per resume (seconds)", fontsize=11)
    ax.set_title(f"Average processing time per method (n={len(rows)} resumes)",
                 fontsize=13, fontweight="bold")
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(avgs) * 1.18)
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "fit_latency.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n  latency chart -> {out}")
    for name, v in zip(names, avgs):
        print(f"    {name:18s} {v:.2f}s avg")


def build_metrics_chart(rows):
    """
    Accuracy / Precision / Recall / F1, matching Fig. 7 of the reference paper
    ("TF-IDF vs BERT Performance Metrics"). Binary classification: positive =
    human-labelled "Good Fit", negative = human-labelled "No Fit" ("Potential
    Fit" is left out -- it's an intentionally ambiguous middle label, not a
    clean positive or negative, so it would only blur a precision/recall
    calculation).

    Threshold per method = the midpoint between that method's own average
    No Fit score and average Good Fit score. This is deliberately NOT a single
    fixed cutoff shared across methods -- TF-IDF, embedding, and the LLM live
    on very different natural scales, so one shared threshold would unfairly
    penalize whichever method happens to score lower on an absolute scale. A
    self-calibrated midpoint asks a fair question of each method: does *your*
    own score for a genuine match sit clearly above *your* own score for a
    genuine non-match?
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    binary_rows = [r for r in rows if r["label"] in ("No Fit", "Good Fit")]
    methods = [("tfidf_pct", "TF-IDF", "#B4453B"),
               ("embedding_pct", "Gemini Embedding", "#378ADD"),
               ("llm_match_pct", "Gemini LLM", "#1D9E75")]

    results = {}
    for key, name, color in methods:
        no_fit_scores = [float(r[key]) for r in binary_rows if r["label"] == "No Fit"]
        good_fit_scores = [float(r[key]) for r in binary_rows if r["label"] == "Good Fit"]
        threshold = (sum(no_fit_scores) / len(no_fit_scores) +
                     sum(good_fit_scores) / len(good_fit_scores)) / 2

        tp = fp = tn = fn = 0
        for r in binary_rows:
            predicted_fit = float(r[key]) >= threshold
            actual_fit = r["label"] == "Good Fit"
            if predicted_fit and actual_fit:
                tp += 1
            elif predicted_fit and not actual_fit:
                fp += 1
            elif not predicted_fit and actual_fit:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / len(binary_rows) if binary_rows else 0.0
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
    ax.set_title(f"Classification performance vs. human fit labels\n"
                 f"(Good Fit vs. No Fit, n={len(binary_rows)} pairs)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
    fig.tight_layout()
    out = os.path.join(CHART_DIR, "fit_metrics.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n  metrics chart -> {out}")

    print("\n=== Accuracy / Precision / Recall / F1 (Good Fit vs. No Fit) ===")
    for key, name, _ in methods:
        r = results[key]
        print(f"  {name:18s} threshold={r['threshold']:5.1f}%  "
              f"acc={r['accuracy']:.2f}  prec={r['precision']:.2f}  "
              f"rec={r['recall']:.2f}  f1={r['f1']:.2f}  "
              f"(TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']})")


if __name__ == "__main__":
    main()
