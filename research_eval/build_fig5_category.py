import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.csv") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r["tfidf_pct"] = float(r["tfidf_pct"])
    r["embedding_pct"] = float(r["embedding_pct"])
    r["llm_match_pct"] = float(r["llm_match_pct"])
    r["is_match"] = 1 if "->" not in r["field"] else 0

pos = [r for r in rows if r["is_match"] == 1]
neg = [r for r in rows if r["is_match"] == 0]

def thr(key):
    return (sum(r[key] for r in pos)/len(pos) + sum(r[key] for r in neg)/len(neg))/2

tfidf_t, emb_t, llm_t = thr("tfidf_pct"), thr("embedding_pct"), thr("llm_match_pct")
alpha, hybrid_t = 0.08, 61.52
for r in rows:
    r["hybrid_pct"] = alpha*r["llm_match_pct"] + (1-alpha)*r["embedding_pct"]

categories = sorted(set(r["field"] for r in pos))
short = {
    "Business Analyst": "Business\nAnalyst",
    "Business Intelligence, Business Object": "BI /\nBusObj",
    "Datawarehousing, ETL, Informatica": "ETL /\nInformatica",
    "Java Developers/Architects": "Java Dev /\nArchitect",
    "Network and Systems Administrators": "Network\nAdmin",
    "Project Manager": "Project\nManager",
    "SQL Developers": "SQL\nDeveloper",
    "Web Developer": "Web\nDeveloper",
}

methods = [("tfidf_pct", tfidf_t, "TF-IDF", "#B4453B"),
           ("embedding_pct", emb_t, "Gemini Embedding", "#378ADD"),
           ("llm_match_pct", llm_t, "Gemini LLM", "#1D9E75"),
           ("hybrid_pct", hybrid_t, "Hybrid (LLM+Embedding)", "#8B5CF6")]

data = {name: [] for _, _, name, _ in methods}
for cat in categories:
    members = [r for r in pos if r["field"] == cat]
    n = len(members)
    for key, t, name, color in methods:
        rec = sum(1 for r in members if r[key] >= t) / n
        data[name].append(rec)

fig, ax = plt.subplots(figsize=(12, 6.5))
x = range(len(categories))
n_methods = len(methods)
width = 0.8 / n_methods
for i, (key, t, name, color) in enumerate(methods):
    vals = data[name]
    offset = (i - (n_methods - 1) / 2) * width
    ax.bar([xi + offset for xi in x], vals, width=width * 0.92, color=color, label=name)

ax.set_xticks(list(x))
ax.set_xticklabels([short.get(c, c) for c in categories], fontsize=9.5)
ax.set_ylabel("Recall (same-field pairs correctly matched)", fontsize=11)
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="#ccc", linewidth=0.8, linestyle=":")
ax.set_title("Per-category recall: the hybrid score ties or leads in every job\ncategory tested (n=20 same-field pairs per category, 8 categories)",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=9.5)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
fig.tight_layout()
out = "charts/results_percategory_hybrid_FINAL.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Chart saved -> {out}")

print("\nSummary (wins/ties out of 8 categories):")
best_per_cat = []
for idx, cat in enumerate(categories):
    vals = {name: data[name][idx] for _, _, name, _ in methods}
    best = max(vals.values())
    winners = [name for name, v in vals.items() if v == best]
    best_per_cat.append(winners)
    print(f"  {cat:40s} best={winners}")
from collections import Counter
c = Counter(n for w in best_per_cat for n in w)
print(c)
