"""
Fig. 2 — Self-calibrated classification accuracy.

Each method's decision threshold is the midpoint of its own average
same-field score and average cross-field score (no cross-validation,
no tuning against ground truth beyond this simple midpoint — the same
calibration approach used for TF-IDF / Embedding / LLM throughout the
paper). This figure shows how well each method separates matches from
non-matches out of the box; Fig. 4 gives the paper's primary,
cross-validated headline numbers.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "results.csv"
CHART_DIR = "charts"
HYBRID_LLM_WEIGHT = 0.08

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r["tfidf_pct"] = float(r["tfidf_pct"])
    r["embedding_pct"] = float(r["embedding_pct"])
    r["llm_match_pct"] = float(r["llm_match_pct"])
    r["is_match"] = 1 if "->" not in r["field"] else 0
    r["fused_pct"] = HYBRID_LLM_WEIGHT * r["llm_match_pct"] + (1 - HYBRID_LLM_WEIGHT) * r["embedding_pct"]

match = [r for r in rows if r["is_match"] == 1]
mismatch = [r for r in rows if r["is_match"] == 0]

methods = [("tfidf_pct", "TF-IDF", "#B4453B"),
           ("embedding_pct", "Gemini Embedding", "#378ADD"),
           ("llm_match_pct", "Gemini LLM", "#1D9E75"),
           ("fused_pct", "Hybrid (LLM+Embedding)", "#8B5CF6")]

results = {}
for key, name, color in methods:
    m_avg = sum(r[key] for r in match) / len(match)
    mm_avg = sum(r[key] for r in mismatch) / len(mismatch)
    threshold = (m_avg + mm_avg) / 2
    tp = fp = tn = fn = 0
    for r in rows:
        pred = r[key] >= threshold
        actual = r["is_match"] == 1
        if pred and actual: tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1
    accuracy = (tp + tn) / len(rows)
    results[key] = dict(name=name, color=color, threshold=threshold, m_avg=m_avg, mm_avg=mm_avg,
                         accuracy=accuracy, tp=tp, fp=fp, tn=tn, fn=fn)
    print(f"{name:24s} thr={threshold:6.2f}  same={m_avg:5.1f}  mismatch={mm_avg:5.1f}  "
          f"gap={m_avg-mm_avg:5.1f}  accuracy={accuracy*100:5.1f}%  (TP={tp} FP={fp} TN={tn} FN={fn})")

fig, ax = plt.subplots(figsize=(8.5, 5.8))
names = [results[k]["name"] for k, _, _ in methods]
accs = [results[k]["accuracy"] * 100 for k, _, _ in methods]
colors = [c for _, _, c in methods]
bars = ax.bar(names, accs, color=colors, width=0.6)
for bar, v in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Self-calibrated classification accuracy (%)", fontsize=11)
ax.set_ylim(0, 108)
ax.set_title("Discriminative power: self-calibrated accuracy\n"
             "(own average-based threshold, n=200 pairs)",
             fontsize=12.5, fontweight="bold")
plt.setp(ax.get_xticklabels(), fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
fig.tight_layout()
os.makedirs(CHART_DIR, exist_ok=True)
out = os.path.join(CHART_DIR, "discrimination_comparison_hybrid.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved -> {out}")
