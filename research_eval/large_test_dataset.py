"""
Builds a large TEST_PAIRS list (real resumes x hand-written JDs) from the
Kaggle 'Resume dataset.csv' (haidermaseeh/resume-dataset) for the Resume
Suite evaluation script.

IMPORTANT: the CSV's own "category" column was found to be unreliable for
several categories -- spot checks and full-dataset keyword analysis showed
e.g. ~94% of resumes labeled "Web Developer Resumes" actually contain
Data-Warehousing/ETL/Informatica content, "Recruiter Resumes" is dominated
by Java/Web developer resumes, etc. Whole-text keyword scanning also proved
too loose (real resumes list dozens of unrelated buzzwords for ATS purposes),
so resumes are instead bucketed by their short 'job_title' field, matched
against TITLE_KEYWORDS below -- e.g. a row only counts as "Web Developer" if
its job_title itself contains "web developer"/"ui developer"/etc, not just
somewhere buried in a long skills list. This scans the FULL 9,000-row dataset
for each category (ignoring the CSV's category column entirely), so a resume
ends up in a category's pool because its own job_title says so.

"Recruiter Resumes" was dropped entirely -- verified three separate ways
(category column, whole-text scan, job_title scan) that this dataset contains
essentially no genuine recruiter resumes (6 out of 9,000 by job_title). The
study uses the remaining 8 categories, each with 58-700+ verified real matches.

For each category, samples N real (job-title-verified) resumes (default 20)
and pairs each with that category's job description from category_jds.py --
these are the "same-field, expect good match" pairs.

Also builds a smaller set of deliberate cross-field mismatch pairs (real
resume from category A vs. JD from category B) so the score distribution
isn't clustered near 100 and the app's discrimination ability can be shown.

Usage:
    from large_test_dataset import build_test_pairs
    TEST_PAIRS = build_test_pairs(csv_path="Resume dataset.csv", n_per_category=20)
"""
import csv
import random

from category_jds import CATEGORY_JDS

RANDOM_SEED = 42
N_CROSS_FIELD_PAIRS = 10  # extra deliberate mismatch pairs on top of the same-field ones

# job_title phrases used to verify a resume's actual role matches a category,
# INSTEAD of trusting the CSV's category column. Matched against the CSV's short
# 'job_title' field (e.g. "Sr. Java Developer"), not the long resume body -- far
# more precise than whole-text keyword scanning. Validated against the full
# 9,000-row dataset -- every category below has 58-700+ genuine matches.
TITLE_KEYWORDS = {
    "Java Developers/Architects Resumes": ["java developer", "java architect", "j2ee developer", "sr. java", "senior java"],
    "Web Developer Resumes": ["web developer", "web ui developer", "front end developer", "front-end developer", "ui developer", "web application developer"],
    "SQL Developers Resumes": ["sql developer", "database developer", "sql server developer"],
    "Business Analyst (BA) Resumes": ["business analyst"],
    "Network and Systems Administrators Resumes": ["network administrator", "systems administrator", "system administrator", "network engineer"],
    "Datawarehousing, ETL, Informatica Resumes": ["informatica", "etl developer", "etl consultant", "data warehouse developer", "datawarehouse"],
    "Business Intelligence, Business Object Resumes": ["business intelligence", "bi developer", "business objects developer", "obiee"],
    "Project Manager Resumes": ["project manager", "program manager"],
}


def _load_resumes_by_category(csv_path: str) -> dict:
    """
    Returns {category: [resume_text, ...]} -- built by scanning ALL rows in the
    CSV (ignoring the possibly-wrong 'category' column) and bucketing each resume
    into every category whose TITLE_KEYWORDS appear in that row's own 'job_title'
    field. A resume can land in more than one category's pool if its title is
    genuinely broad (e.g. "Java/Web Developer"); that's fine and realistic.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("Text", "").strip()
            title = row.get("job_title", "").strip()
            if text:
                rows.append((title.lower(), text))

    by_cat = {cat: [] for cat in TITLE_KEYWORDS}
    for title_lower, text in rows:
        for cat, keywords in TITLE_KEYWORDS.items():
            if any(kw in title_lower for kw in keywords):
                by_cat[cat].append(text)

    return by_cat


def _short_label(category: str) -> str:
    return (
        category.replace(" Resumes", "")
        .replace("(BA)", "")
        .strip()
    )


def build_test_pairs(csv_path: str = "Resume dataset.csv", n_per_category: int = 20,
                      n_cross_field: int = None) -> list:
    """
    IMPORTANT for resumable runs: same-field resumes are chosen by shuffling each
    category's FULL pool once (fixed, independent of n_per_category) and then taking
    the first n_per_category of that shuffled list. This means raising n_per_category
    later (e.g. 4 -> 10 per category) always produces a strict SUPERSET of the earlier
    pairs with the same ids -- already-scored pairs in results.csv are never invalidated
    or silently swapped for different resumes. Same logic applies to n_cross_field.
    """
    rng = random.Random(RANDOM_SEED)
    resumes_by_cat = _load_resumes_by_category(csv_path)

    categories = list(CATEGORY_JDS.keys())
    missing = [c for c in categories if c not in resumes_by_cat]
    if missing:
        raise ValueError(f"Categories in category_jds.py not found in CSV: {missing}")

    # Shuffle each category's full pool ONCE, regardless of n_per_category, so the
    # rng state (and therefore every downstream draw, including mismatches) doesn't
    # depend on how many pairs we're building this run.
    shuffled_by_cat = {}
    for cat in categories:
        pool = list(resumes_by_cat[cat])
        rng.shuffle(pool)
        shuffled_by_cat[cat] = pool

    pairs = []

    # Fixed numeric block per category (0, 1000, 2000, ...) so a pair's id/number
    # NEVER shifts when n_per_category changes -- ids depend only on (category
    # index, position within that category's shuffled pool), not on how many
    # pairs happen to precede it in a given run. 1000 is a safe block size since
    # each category tops out at 1000 resumes in this dataset.
    BLOCK = 1000
    MISMATCH_BASE = BLOCK * len(categories) + 1000  # a range that same-field ids can never reach

    # ── same-field pairs: first N (shuffled) real resumes per category vs. that category's JD ──
    for cat_idx, cat in enumerate(categories):
        pool = shuffled_by_cat[cat]
        sample_size = min(n_per_category, len(pool))
        for local_idx, resume_text in enumerate(pool[:sample_size]):
            id_num = cat_idx * BLOCK + local_idx + 1
            pairs.append({
                "id": f"P{id_num:05d}_{_short_label(cat).replace(' ', '_').replace(',', '').replace('/', '-')}_match",
                "label": f"{_short_label(cat)} resume vs matching JD",
                "field": _short_label(cat),
                "resume_text": resume_text[:10000],  # generous cap -- app.py/eval prompts slice further (8000 chars) as needed
                "jd_text": CATEGORY_JDS[cat],
            })

    # ── cross-field mismatch pairs: resume from cat A vs JD from cat B ──
    # Drawn AFTER shuffling (which always consumes the same fixed amount of rng
    # state regardless of n_per_category), so the i-th mismatch pair is always the
    # same choice regardless of n_per_category or how many mismatches were drawn
    # in an earlier, smaller run -- raising n_cross_field only appends new ones.
    n_cross = N_CROSS_FIELD_PAIRS if n_cross_field is None else n_cross_field
    for i in range(n_cross):
        cat_resume, cat_jd = rng.sample(categories, 2)
        resume_text = rng.choice(resumes_by_cat[cat_resume])
        id_num = MISMATCH_BASE + i + 1
        pairs.append({
            "id": f"P{id_num:05d}_{_short_label(cat_resume).replace(' ', '_')}_vs_{_short_label(cat_jd).replace(' ', '_')}_mismatch",
            "label": f"{_short_label(cat_resume)} resume vs {_short_label(cat_jd)} JD (cross-field)",
            "field": f"{_short_label(cat_resume)} -> {_short_label(cat_jd)}",
            "resume_text": resume_text[:10000],
            "jd_text": CATEGORY_JDS[cat_jd],
        })

    return pairs


if __name__ == "__main__":
    # quick sanity check when run directly
    pairs = build_test_pairs(n_per_category=20)
    print(f"Built {len(pairs)} pairs")
    same_field = [p for p in pairs if "mismatch" not in p["id"]]
    mismatch = [p for p in pairs if "mismatch" in p["id"]]
    print(f"  same-field: {len(same_field)}")
    print(f"  cross-field mismatch: {len(mismatch)}")
    print("Sample pair:", pairs[0]["id"], "-", pairs[0]["label"])
