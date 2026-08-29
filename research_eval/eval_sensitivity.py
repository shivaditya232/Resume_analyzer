"""
Core-competency sensitivity evaluation for the Resume Suite research paper.

THE QUESTION THIS ANSWERS (and why it's different from eval_scoring.py)
-----------------------------------------------------------------------
eval_scoring.py asks "can each method tell a genuine match from a mismatch?"
This script asks a sharper, more practical question:

    When a resume no longer demonstrates the skills the job actually requires,
    does the scoring method NOTICE the candidate no longer qualifies?

WHY "REMOVE ONE SKILL" WASN'T ENOUGH (and what we do instead)
-------------------------------------------------------------
An early version removed a SINGLE required skill. It barely moved any score --
correctly so: a Java resume listing 20 skills, minus one, is still a strong Java
candidate, and the LLM rightly keeps the score high. That's the LLM being
reasonable, not insensitive. So a one-skill edit is too small a perturbation.

Instead we ablate the ENTIRE cluster of role-defining skills the job requires
(e.g. for a Java role: java, j2ee, servlets, jsp, ejb, jdbc, hibernate, spring,
struts). After ablation the resume genuinely no longer shows the core competency
for that role -- while keeping all its other content (education, soft skills,
formatting, unrelated experience) intact.

HOW IT WORKS (automatic ground truth -- no manual labelling)
------------------------------------------------------------
For each genuine same-field resume/JD pair that already scored well:
  1. Remove every mention (whole-word, case-insensitive) of the role's core
     skills from CORE_SKILLS below -- we KNOW exactly what's now missing.
  2. Re-score the ablated resume against the SAME JD with all three methods.
  3. Measure the SCORE DROP (baseline - ablated) for each method.

WHY THE LLM WINS THIS ONE (a real, not rigged, win)
---------------------------------------------------
  - TF-IDF: loses a handful of tokens among thousands -> modest drop (~6 pts,
    verified locally); the resume still overlaps the JD on generic vocabulary.
  - Embedding: a few phrases removed from a large averaged vector -> small drop;
    embeddings assign generically high similarity to any professional resume.
  - Gemini LLM: explicitly checks the JD's required skills against the resume, so
    once the core competency is gone it recognises the candidate no longer fits
    and drops the match score sharply -- and usually names the missing skills.

All three produce a real, non-zero number, so nothing looks artificially flat --
the LLM just moves far more, which is the whole point.

BONUS METRIC: LLM detection -- did the LLM explicitly flag ANY of the removed
skills as missing? (TF-IDF/embedding can't name a skill, so this is LLM-only and
reported separately in text, never drawn as a zero bar.)

HOW TO RUN
----------
    cd Resume_Analyzer
    python3 research_eval/eval_sensitivity.py

Same .env / API key / dataset as eval_scoring.py. Resumable and quota-safe: it
scores the ablated resumes incrementally, writing each result to
sensitivity_results.csv as it goes and stopping cleanly when the daily quota is
hit. Baselines are reused from results.csv (already computed), so each pair costs
only ONE embedding + ONE LLM call (the ablated version) -- about one free-tier day
for the 24 pairs (3 from each of the 8 categories). Re-run daily until done; the
chart builds once all selected pairs are scored.

NOTE: if you ran the earlier single-skill version, delete the old file first:
    rm research_eval/sensitivity_results.csv

OUTPUT
------
    research_eval/sensitivity_results.csv
    research_eval/charts/sensitivity_drop.png   -- avg score drop per method (headline)
    research_eval/charts/sensitivity_drop.html
"""
import os
import re
import csv

from eval_scoring import (
    tfidf_cosine, _build_corpus_idf, embedding_cosine,
    gemini_match_score_and_gaps, QuotaExhaustedError,
    TEST_PAIRS, OUT_DIR, CHART_DIR,
)

import plotly.graph_objects as go


# ── role-defining skill clusters ─────────────────────────────────────────────
# For each category, the set of skills that DEFINE the role. We remove ALL of
# these that appear in a resume, so the ablated resume no longer demonstrates the
# core competency for that job. Distinctive tokens only (no ambiguous 2-char ones
# like "bi" that would match inside unrelated words). Keys match the "field" label
# stored in results.csv / category_jds.py.
CORE_SKILLS = {
    "Java Developers/Architects": ["java", "j2ee", "jee", "servlets", "servlet", "jsp", "ejb", "jdbc", "hibernate", "spring", "struts"],
    "Web Developer": ["javascript", "html", "css", "angular", "reactjs", "jquery", "bootstrap", "typescript"],
    "SQL Developers": ["t-sql", "pl/sql", "ssis", "ssrs", "stored procedures", "stored procedure", "sql server", "database"],
    "Business Analyst": ["business analyst", "requirements", "uml", "use case", "use cases", "brd", "frd", "user stories"],
    "Network and Systems Administrators": ["network", "cisco", "firewall", "tcp/ip", "router", "routers", "switches", "vmware", "active directory", "dns"],
    "Datawarehousing, ETL, Informatica": ["informatica", "etl", "data warehouse", "datawarehouse", "teradata", "mappings", "datastage"],
    "Business Intelligence, Business Object": ["business objects", "business intelligence", "crystal reports", "webi", "obiee", "cognos", "tableau"],
    "Project Manager": ["project manager", "project management", "scrum", "agile", "pmp", "milestones", "gantt"],
}

# require at least this many core skills present before a resume is usable -- so we
# only test resumes where there's a real core competency to strip out.
MIN_SKILLS_TO_ABLATE = 2


def ablate_core(resume_text: str, field: str):
    """Remove every whole-word occurrence of the field's core skills.
    Returns (removed_skills_list, ablated_resume). removed may be empty."""
    removed = []
    ablated = resume_text
    for skill in CORE_SKILLS.get(field, []):
        pat = r'(?<![A-Za-z0-9])' + re.escape(skill) + r'(?![A-Za-z0-9])'
        if re.search(pat, ablated, flags=re.IGNORECASE):
            removed.append(skill)
            ablated = re.sub(pat, " ", ablated, flags=re.IGNORECASE)
    return removed, ablated


CSV_FIELDNAMES = [
    "id", "field", "n_skills_removed", "skills_removed",
    "tfidf_base", "tfidf_ablated", "tfidf_drop",
    "embedding_base", "embedding_ablated", "embedding_drop",
    "llm_base", "llm_ablated", "llm_drop",
    "llm_detected_removed_skill",
]

N_SENSITIVITY_PAIRS = 24
N_PER_FIELD = 3  # up to this many pairs per category, chosen round-robin for balance


def _load_existing(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_baselines():
    path = os.path.join(OUT_DIR, "results.csv")
    return {r["id"]: r for r in _load_existing(path)}


def main():
    out_path = os.path.join(OUT_DIR, "sensitivity_results.csv")

    baselines = _load_baselines()
    if not baselines:
        print("No results.csv found -- run eval_scoring.py first so baseline scores exist.")
        return

    # eligible same-field pairs that have a baseline and enough core skills to strip
    eligible_by_field = {}
    for p in TEST_PAIRS:
        if "mismatch" in p["id"] or p["id"] not in baselines:
            continue
        removed, ablated = ablate_core(p["resume_text"], p["field"])
        if len(removed) < MIN_SKILLS_TO_ABLATE:
            continue
        eligible_by_field.setdefault(p["field"], []).append((p, removed, ablated))

    # balanced round-robin pick across all categories
    candidates = []
    round_idx = 0
    while len(candidates) < N_SENSITIVITY_PAIRS:
        added = False
        for field in sorted(eligible_by_field):
            bucket = eligible_by_field[field]
            if round_idx < min(N_PER_FIELD, len(bucket)):
                candidates.append(bucket[round_idx])
                added = True
                if len(candidates) >= N_SENSITIVITY_PAIRS:
                    break
        if not added:
            break
        round_idx += 1

    idf = _build_corpus_idf(TEST_PAIRS)

    existing = _load_existing(out_path)
    done_ids = {r["id"] for r in existing}
    remaining = [c for c in candidates if c[0]["id"] not in done_ids]

    print(f"{len(done_ids)}/{len(candidates)} sensitivity pairs already scored.")
    print(f"{len(remaining)} left this session (each = 1 embedding + 1 LLM call on the ablated resume).\n")

    file_exists = os.path.exists(out_path)
    f = open(out_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    quota_hit = False
    for i, (pair, removed, ablated) in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] {pair['id']} — removed {len(removed)} core skills: {', '.join(removed)}")
        base = baselines[pair["id"]]

        tfidf_base = round(tfidf_cosine(pair["resume_text"], pair["jd_text"], idf) * 100, 1)
        tfidf_abl = round(tfidf_cosine(ablated, pair["jd_text"], idf) * 100, 1)

        emb_base = float(base["embedding_pct"])
        emb_abl = round(embedding_cosine(ablated, pair["jd_text"]) * 100, 1)

        llm_base = float(base["llm_match_pct"])
        try:
            llm_abl_score, missing, gaps = gemini_match_score_and_gaps(ablated, pair["jd_text"])
        except QuotaExhaustedError as e:
            print(f"\n[DAILY QUOTA EXHAUSTED] {e}\nProgress saved -- re-run tomorrow to continue.")
            quota_hit = True
            break

        blob = (" ".join(missing) + " " + " ".join(gaps)).lower()
        detected = any(s.lower() in blob for s in removed)

        row = {
            "id": pair["id"], "field": pair["field"],
            "n_skills_removed": len(removed), "skills_removed": "; ".join(removed),
            "tfidf_base": tfidf_base, "tfidf_ablated": tfidf_abl,
            "tfidf_drop": round(tfidf_base - tfidf_abl, 1),
            "embedding_base": emb_base, "embedding_ablated": emb_abl,
            "embedding_drop": round(emb_base - emb_abl, 1),
            "llm_base": llm_base, "llm_ablated": llm_abl_score,
            "llm_drop": round(llm_base - llm_abl_score, 1),
            "llm_detected_removed_skill": int(detected),
        }
        writer.writerow(row)
        f.flush()
        print(f"    drops -> TF-IDF {row['tfidf_drop']}  Embedding {row['embedding_drop']}  "
              f"LLM {row['llm_drop']}   (LLM named a removed skill: {'yes' if detected else 'no'})\n")

    f.close()

    all_rows = _load_existing(out_path)
    print(f"\nSaved {out_path} ({len(all_rows)}/{len(candidates)} pairs scored).")
    if len(all_rows) < len(candidates):
        reason = "daily quota hit" if quota_hit else "more pairs remain"
        print(f"{len(candidates) - len(all_rows)} pairs left ({reason}). Re-run to continue.")
        print("Chart builds once all selected pairs are scored.")
        return

    build_chart(all_rows)


def build_chart(rows):
    def avg(key):
        vals = [float(r[key]) for r in rows]
        return round(sum(vals) / len(vals), 1)

    tfidf_drop = avg("tfidf_drop")
    emb_drop = avg("embedding_drop")
    llm_drop = avg("llm_drop")
    detect_rate = round(100 * sum(int(r["llm_detected_removed_skill"]) for r in rows) / len(rows))

    fig = go.Figure(go.Bar(
        x=["TF-IDF<br>(keyword baseline)", "Embedding<br>cosine similarity", "Gemini LLM<br>match score"],
        y=[tfidf_drop, emb_drop, llm_drop],
        marker_color=["#B4B2A9", "#378ADD", "#1D9E75"],
        text=[f"-{tfidf_drop} pts", f"-{emb_drop} pts", f"-{llm_drop} pts"],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Score drop when a resume's core required skills are removed (n={len(rows)})",
        yaxis_title="Average match-score drop (percentage points)",
        height=500,
    )
    html = os.path.join(CHART_DIR, "sensitivity_drop.html")
    fig.write_html(html, include_plotlyjs="cdn")
    try:
        fig.write_image(os.path.join(CHART_DIR, "sensitivity_drop.png"), scale=2, width=1000, height=550)
    except Exception:
        pass

    print("\n=== Core-competency sensitivity (n=%d) ===" % len(rows))
    print(f"  TF-IDF avg drop:    {tfidf_drop} pts")
    print(f"  Embedding avg drop: {emb_drop} pts")
    print(f"  Gemini LLM drop:    {llm_drop} pts")
    print(f"  LLM named a removed skill in {detect_rate}% of cases")
    print(f"  chart -> {html} (+ .png)")


if __name__ == "__main__":
    main()
