"""
Output-richness evaluation for the Resume Suite research paper.

THE POINT (and why it's different from eval_scoring.py / graph 1)
-----------------------------------------------------------------
Graph 1 shows the LLM separates genuine matches from mismatches far better than
the baselines. This chart shows the OTHER, equally important advantage -- and the
one that is the paper's actual thesis: what each method hands back to the user.

  - TF-IDF returns ONE thing: a similarity percentage.
  - Embedding cosine similarity returns ONE thing: a similarity percentage.
  - The Gemini LLM returns the match score PLUS a list of specific missing skills
    PLUS a set of explained critical gaps (and, in the app, concrete suggestions).

So a classical method tells a user "you are a 63% match" and stops. The LLM tells
them "you are a 77% match, you are missing Docker / Kubernetes / CI-CD, and your
three biggest gaps are X, Y, Z" -- feedback they can actually act on.

This is measured, not asserted: the counts below are the real averages of the
missing-keyword and critical-gap fields the LLM produced across every pair in
results.csv. No API calls are needed -- it just reads what was already collected.

HONEST FRAMING (no rigged zero-bars)
------------------------------------
Every method produces at least the score, so the baselines are shown as a bar of
height 1 (the score), not zero. The LLM's bar is stacked: score + missing skills +
critical gaps. Nothing is flattened to zero; the LLM simply towers over the others
because it genuinely returns ~10x more usable information per analysis.

HOW TO RUN
----------
    cd Resume_Analyzer
    python3 research_eval/eval_output_richness.py

OUTPUT
------
    research_eval/charts/output_richness.png
"""
import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def _count(field: str) -> int:
    return len([x for x in field.split(";") if x.strip()])


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "results.csv"))))
    n = len(rows)

    avg_keywords = sum(_count(r["missing_keywords"]) for r in rows) / n
    avg_gaps = sum(_count(r["critical_gaps"]) for r in rows) / n
    avg_keywords, avg_gaps = round(avg_keywords, 1), round(avg_gaps, 1)
    llm_total = round(1 + avg_keywords + avg_gaps, 1)

    methods = ["TF-IDF\n(keyword baseline)", "Embedding\ncosine similarity", "Gemini LLM\nmatch score"]
    x = range(len(methods))

    # every method returns the score (height 1); only the LLM adds the rest
    score_layer = [1, 1, 1]
    kw_layer = [0, 0, avg_keywords]
    gap_layer = [0, 0, avg_gaps]

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    b1 = ax.bar(x, score_layer, color="#B4B2A9", label="Match score (1 value)")
    b2 = ax.bar(x, kw_layer, bottom=score_layer, color="#378ADD",
                label=f"Named missing skills ({avg_keywords} avg)")
    b3 = ax.bar(x, gap_layer, bottom=[s + k for s, k in zip(score_layer, kw_layer)],
                color="#1D9E75", label=f"Explained critical gaps ({avg_gaps} avg)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Distinct pieces of feedback returned per resume")
    ax.set_title(f"What each method actually returns to the user (n={n} resumes)")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, llm_total + 1.5)

    # annotate totals on top of each bar
    for xi, total in zip(x, [1, 1, llm_total]):
        ax.text(xi, total + 0.15, f"{total:g}", ha="center", va="bottom", fontweight="bold")

    ax.text(0.5, 1.0 + 0.15, "score only", ha="center", va="bottom", fontsize=9, color="#555")
    ax.text(0.98, 0.97,
            "TF-IDF & embedding return a single number.\n"
            "The LLM returns the score plus specific,\nactionable feedback the user can act on.",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#F4F3EE", ec="#B4B2A9"))

    fig.tight_layout()
    out = os.path.join(CHART_DIR, "output_richness.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print("=== Output richness (n=%d) ===" % n)
    print(f"  TF-IDF returns:    1 value (score)")
    print(f"  Embedding returns: 1 value (score)")
    print(f"  Gemini LLM returns: {llm_total} items = 1 score + {avg_keywords} missing skills + {avg_gaps} gaps")
    print(f"  chart -> {out}")


if __name__ == "__main__":
    main()
