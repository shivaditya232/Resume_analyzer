import csv, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "results.csv"
CHART_DIR = "charts"

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r["tfidf_pct"] = float(r["tfidf_pct"])
    r["embedding_pct"] = float(r["embedding_pct"])
    r["llm_match_pct"] = float(r["llm_match_pct"])
    r["is_match"] = 1 if "->" not in r["field"] else 0

match = [r for r in rows if r["is_match"] == 1]
mismatch = [r for r in rows if r["is_match"] == 0]

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
        actual_fit = r["is_match"] == 1
        if predicted_fit and actual_fit: tp += 1
        elif predicted_fit and not actual_fit: fp += 1
        elif not predicted_fit and actual_fit: fn += 1
        else: tn += 1
    accuracy = (tp + tn) / len(rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    results[key] = dict(name=name, color=color, threshold=round(threshold, 1),
                         accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                         tp=tp, fp=fp, tn=tn, fn=fn)

# ---- Hybrid: 5-fold cross-validated fusion score (alpha*LLM + (1-alpha)*embedding) ----
pos = [r for r in rows if r["is_match"] == 1]
neg = [r for r in rows if r["is_match"] == 0]
random.seed(42)
random.shuffle(pos)
random.shuffle(neg)
K = 5
pos_folds = [pos[i::K] for i in range(K)]
neg_folds = [neg[i::K] for i in range(K)]

def metrics_at(data, alpha, thr):
    tp = fp = tn = fn = 0
    for r in data:
        s = alpha * r["llm_match_pct"] + (1 - alpha) * r["embedding_pct"]
        pred = s >= thr
        actual = r["is_match"] == 1
        if pred and actual: tp += 1
        elif pred and not actual: fp += 1
        elif not pred and actual: fn += 1
        else: tn += 1
    return tp, fp, tn, fn

def best_alpha_thr(train):
    best = None
    for ai in range(0, 101, 5):
        alpha = ai / 100
        vals = sorted(set(alpha * r["llm_match_pct"] + (1 - alpha) * r["embedding_pct"] for r in train))
        cand_thrs = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals)-1)] + [vals[0]-1, vals[-1]+1]
        for thr in cand_thrs:
            tp, fp, tn, fn = metrics_at(train, alpha, thr)
            n = len(train)
            acc = (tp + tn) / n
            prec = tp / (tp+fp) if tp+fp else 0
            rec = tp / (tp+fn) if tp+fn else 0
            f1v = 2*prec*rec/(prec+rec) if prec+rec else 0
            if best is None or f1v > best[0]:
                best = (f1v, alpha, thr)
    return best[1], best[2]

all_tp = all_fp = all_tn = all_fn = 0
chosen = []
for k in range(K):
    test = pos_folds[k] + neg_folds[k]
    train = []
    for j in range(K):
        if j != k:
            train += pos_folds[j] + neg_folds[j]
    alpha, thr = best_alpha_thr(train)
    chosen.append((alpha, thr))
    tp, fp, tn, fn = metrics_at(test, alpha, thr)
    all_tp += tp; all_fp += fp; all_tn += tn; all_fn += fn

avg_alpha = sum(c[0] for c in chosen) / K
avg_thr = sum(c[1] for c in chosen) / K
acc = (all_tp + all_tn) / len(rows)
prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0
rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0

results["hybrid"] = dict(name="Hybrid (LLM+Embedding)", color="#8B5CF6", threshold=round(avg_thr, 1),
                          accuracy=acc, precision=prec, recall=rec, f1=f1,
                          tp=all_tp, fp=all_fp, tn=all_tn, fn=all_fn)
methods.append(("hybrid", "Hybrid (LLM+Embedding)", "#8B5CF6"))

print(f"Per-fold (alpha, threshold): {chosen}")
print(f"Average alpha={avg_alpha:.3f}, average threshold={avg_thr:.2f}")
print("\n=== Pooled 5-fold cross-validated results (out-of-fold), n=200 ===")
print(f"TP={all_tp} FP={all_fp} TN={all_tn} FN={all_fn}")
print(f"acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_keys = ["accuracy", "precision", "recall", "f1"]

fig, ax = plt.subplots(figsize=(10, 6.2))
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
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(list(x))
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.14)
ax.set_title(f"Classification performance on verified resume-JD pairs\n"
             f"(same-field vs. cross-field, n={len(rows)} pairs; Hybrid = 5-fold cross-validated)",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=9.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
fig.tight_layout()
os.makedirs(CHART_DIR, exist_ok=True)
out = os.path.join(CHART_DIR, "results_metrics_n200_hybrid_FINAL.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nChart saved -> {out}")

print("\n=== Full summary ===")
for key, name, _ in methods:
    r = results[key]
    print(f"  {name:24s} threshold={r['threshold']:6.2f}  "
          f"acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
          f"rec={r['recall']:.3f}  f1={r['f1']:.3f}  "
          f"(TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']})")
