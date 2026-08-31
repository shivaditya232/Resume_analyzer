import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.08

rows = list(csv.DictReader(open("synonym_results.csv")))
canon = {r["skill"]: r for r in rows if r["version"] == "canonical"}
syn = {r["skill"]: r for r in rows if r["version"] == "synonym"}
skills = list(canon.keys())

def hybrid(row):
    return ALPHA*float(row["llm_score"]) + (1-ALPHA)*float(row["embedding_pct"])

methods = [
    ("TF-IDF", lambda r: float(r["tfidf_pct"]), "#B4453B"),
    ("Gemini Embedding", lambda r: float(r["embedding_pct"]), "#378ADD"),
    ("Gemini LLM", lambda r: float(r["llm_score"]), "#1D9E75"),
    ("Hybrid (LLM+Embedding)", hybrid, "#8B5CF6"),
]

fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
for ax, (name, fn, color) in zip(axes, methods):
    c_vals = [fn(canon[s]) for s in skills]
    s_vals = [fn(syn[s]) for s in skills]
    for cv, sv in zip(c_vals, s_vals):
        ax.plot([0, 1], [cv, sv], color=color, alpha=0.35, linewidth=1.1, marker="o", markersize=3)
    avg_c, avg_s = sum(c_vals)/len(c_vals), sum(s_vals)/len(s_vals)
    ax.plot([0, 1], [avg_c, avg_s], color=color, linewidth=3.2, marker="o", markersize=7, zorder=5)
    ax.text(0, avg_c+3, f"{avg_c:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.text(1, avg_s+3, f"{avg_s:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Full name", "Abbreviation"], fontsize=9.5)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)
axes[0].set_ylabel("Match score (%)", fontsize=11)

fig.suptitle("Match score for each of twelve skills before and after rewording, shown\n"
             "separately for four methods (thin lines/dots = individual skills, bold line = average)",
             fontsize=12.5, fontweight="bold", y=1.06)
fig.tight_layout()
fig.savefig("charts/synonym_smallmultiples_hybrid.png", dpi=200, bbox_inches="tight")
print("Saved -> charts/synonym_smallmultiples_hybrid.png")

print("\nAverages:")
for name, fn, _ in methods:
    avg_c = sum(fn(canon[s]) for s in skills)/len(skills)
    avg_s = sum(fn(syn[s]) for s in skills)/len(skills)
    print(f"  {name:24s} canonical={avg_c:.1f}  synonym={avg_s:.1f}  drop={avg_c-avg_s:.1f}")
