# research_eval/

Offline evaluation harness behind the paper in `../paper/draft.docx`. Compares
TF-IDF, Gemini embedding similarity, the Gemini LLM match score, and the
hybrid fusion of the LLM + embedding scores on verified resume/job-description
pairs. This is a research/reproducibility artifact, not part of the deployed
app (`../app.py`).

Trimmed to just what supports the paper's figures — historical/exploratory
scoring attempts (rejected prompt-engineering fixes, sensitivity checks,
model-comparison diagnostics, etc.) have been removed; the paper's Results
section describes what those attempts were and why they didn't ship.

## Files

- `category_jds.py` — hand-written job descriptions per category.
- `large_test_dataset.py` — builds the 200-pair same-field/cross-field test set.
- `eval_scoring.py` — scores every pair with TF-IDF, embedding, and the LLM;
  resumable (append-only), since the LLM call is rate-limited. Produces
  `results.csv`.
- `build_fig4_hybrid.py` — builds Fig. 4 (classification metrics: TF-IDF,
  embedding, LLM, and the cross-validated hybrid) from `results.csv`.
- `build_fig5_category.py` — builds Fig. 5 (per-category recall, all four
  methods) from `results.csv`.
- `results.csv` — the n=200 scored dataset used for Fig. 4 and Fig. 5.
- `charts/` — the four chart images actually embedded in the paper:
  - `discrimination_comparison.png` — Fig. 2
  - `synonym_smallmultiples.png` — Fig. 3
  - `results_metrics_n200_hybrid_FINAL.png` — Fig. 4
  - `results_percategory_hybrid_FINAL.png` — Fig. 5

  Fig. 2 and Fig. 3 come from an earlier evaluation run whose generating
  script predates this cleanup and isn't part of the current codebase; the
  images are kept since they're the exact figures published in the paper.

## Data not tracked in git

`Resume dataset.csv` (~102MB, exceeds GitHub's file size limit) is gitignored.
It's the Kaggle "Resume Dataset" referenced in the paper's Dataset Description
section — download it from Kaggle and place it directly in this folder to
re-run `eval_scoring.py`.
