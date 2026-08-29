# research_eval/

Offline evaluation harness behind the paper in `../paper/draft.docx`. Compares
TF-IDF, Gemini embedding similarity, the Gemini LLM match score, and the
hybrid fusion of the LLM + embedding scores on verified resume/job-description
pairs. This is a research/reproducibility artifact, not part of the deployed
app (`../app.py`).

## Current pipeline (what actually produced the paper's numbers)

- `category_jds.py` — hand-written job descriptions per category.
- `large_test_dataset.py` — builds the 200-pair same-field/cross-field test set.
- `eval_scoring.py` — scores every pair with all three base methods; resumable
  (append-only), since the LLM call is rate-limited. Produces `results.csv`.
- `build_fig4_hybrid.py` — builds Fig. 4 (classification metrics, all 4 methods,
  hybrid cross-validated) from `results.csv`.
- `build_fig5_category.py` — builds Fig. 5 (per-category recall) from `results.csv`.
- `results.csv` — the n=200 scored dataset used for Fig. 4 and Fig. 5.
- `charts/` — final chart images embedded in the paper.

## Historical / exploratory scripts

Kept for provenance — several of these correspond to approaches the paper
explicitly discusses trying and rejecting (see the false-positive discussion
in Results). All import from `eval_scoring.py` as a flat sibling module, so
they must stay in this directory to keep running.

- `eval_rescoring_v2.py` / `v3` / `v4` — three prompt-engineering fixes tried
  against the LLM's adjacent-domain false positives (missing-tool penalty,
  core-vs-secondary skill split, job-function gate); each fixed some cases but
  broke others, so none shipped. Outputs: `results_v2/v3/v4.csv`.
- `eval_sensitivity.py`, `eval_synonym.py` — earlier robustness checks that
  fed into Fig. 3. Outputs: `sensitivity_results.csv`, `synonym_results.csv`.
- `eval_model_comparison.py`, `eval_fit.py`, `eval_output_richness.py`,
  `diagnose_flash_lite.py`, `list_models.py`, `eval_cache.py` — one-off model
  and caching diagnostics.

## Data not tracked in git

`Resume dataset.csv` (~102MB, exceeds GitHub's file size limit) and
`fit_dataset.csv` (~15MB) are gitignored. To regenerate them:

```python
from datasets import load_dataset
load_dataset("cnamuangtoun/resume-job-description-fit")["test"].to_csv(
    "research_eval/fit_dataset.csv"
)
```

`Resume dataset.csv` is the Kaggle "Resume Dataset" referenced in the paper's
Dataset Description section — download it from Kaggle and place it directly
in this folder.
