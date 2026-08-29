"""
Synonym-robust skill identification for the Resume Suite research paper.

THE POINT (a genuinely different axis from graph 1)
---------------------------------------------------
Graph 1 asks "can the method separate a good match from a bad one?" This asks a
DIFFERENT question -- about identification, not scoring:

    When a resume lists a required skill using a common abbreviation or synonym
    (e.g. "JS" for JavaScript, "K8s" for Kubernetes, "Postgres" for PostgreSQL),
    does the method still recognise the skill is PRESENT?

This is measurable with clean, automatic ground truth: we control the wording, so
we KNOW the skill is present in both versions of each resume.

THREE METHODS COMPARED
-----------------------
  1. TF-IDF cosine similarity -- the same corpus-wide-IDF method used everywhere
     else in this project (eval_scoring.py). A real, working baseline, not a
     rigged yes/no check -- it gives a genuine similarity percentage.
  2. Gemini embedding cosine similarity -- the semantic method used in the
     deployed app itself. The more serious baseline: does semantic embedding
     alone already solve this, or does it also get thrown off by wording?
  3. Gemini LLM match score -- full contextual reasoning.

EXPERIMENT (minimal pairs -- a standard NLP robustness design)
--------------------------------------------------------------
For each skill we build two versions of the same resume against a JD that
requires that skill (canonical name): one using the skill's canonical name, one
using its common synonym/abbreviation, with every other line identical. All three
methods score both versions against the same JD.

EXPECTED RESULT
---------------
TF-IDF and embedding should both show at least some sensitivity to the wording
change (their score may dip when the exact/semantically-anchored term changes),
while the LLM's score should barely move -- it reasons that "JS" and "JavaScript"
are the same skill regardless of which one appears.

HOW TO RUN
----------
    cd Resume_Analyzer
    python3 research_eval/eval_synonym.py

Resumable / quota-safe, and now BACKFILL-AWARE: TF-IDF is free (computed locally,
instantly, no quota). Embedding uses a separate quota from the LLM generate calls,
so it can often be filled in even on a day the LLM quota is already used up. If a
case in synonym_results.csv already has an LLM score from an earlier run, this
script will NOT call the LLM again for it -- it only computes whatever is still
missing (e.g. just the embedding score), so nothing already scored gets wasted.

OUTPUT
------
    research_eval/synonym_results.csv
    research_eval/charts/synonym_three_way.png / .html
"""
import os
import re
import csv
import math
from collections import Counter

from eval_scoring import (
    embedding_cosine, gemini_match_score_and_gaps, QuotaExhaustedError,
    OUT_DIR, CHART_DIR,
)

RESULTS = os.path.join(OUT_DIR, "synonym_results.csv")

# (canonical skill required by the JD, common synonym/abbreviation, a resume line
#  using the canonical form, a resume line using the synonym form). The two resume
#  lines are semantically identical -- only the skill's wording differs.
GENERIC_LINE = "Contributed to sprint planning, code reviews, and on-call rotations."

SKILLS = [
    ("Kubernetes", "K8s",
     "Orchestrated containerized microservices in production using Kubernetes.",
     "Orchestrated containerized microservices in production using K8s."),
    ("JavaScript", "JS",
     "Built interactive web front-ends with JavaScript and modern tooling.",
     "Built interactive web front-ends with JS and modern tooling."),
    ("PostgreSQL", "Postgres",
     "Designed and tuned relational schemas on PostgreSQL for high-traffic apps.",
     "Designed and tuned relational schemas on Postgres for high-traffic apps."),
    ("Amazon Web Services", "AWS",
     "Deployed and scaled cloud infrastructure on Amazon Web Services.",
     "Deployed and scaled cloud infrastructure on AWS."),
    ("Continuous Integration and Continuous Deployment", "CI/CD",
     "Automated build and release with Continuous Integration and Continuous Deployment pipelines.",
     "Automated build and release with CI/CD pipelines."),
    ("Machine Learning", "ML",
     "Trained and deployed predictive models using Machine Learning techniques.",
     "Trained and deployed predictive models using ML techniques."),
    ("Natural Language Processing", "NLP",
     "Built text classification systems using Natural Language Processing.",
     "Built text classification systems using NLP."),
    ("TypeScript", "TS",
     "Developed large front-end codebases in TypeScript for type safety.",
     "Developed large front-end codebases in TS for type safety."),
    ("Node.js", "NodeJS",
     "Implemented server-side REST services with Node.js.",
     "Implemented server-side REST services with NodeJS."),
    ("React", "React.js",
     "Created reusable component-based user interfaces with React.",
     "Created reusable component-based user interfaces with React.js."),
    ("Structured Query Language", "SQL",
     "Wrote complex analytical queries in Structured Query Language.",
     "Wrote complex analytical queries in SQL."),
    ("Representational State Transfer", "REST",
     "Designed Representational State Transfer web service APIs.",
     "Designed REST web service APIs."),
]

RESUME_TEMPLATE = """Software Engineer with 5 years of professional experience delivering
scalable production software.

PROFESSIONAL EXPERIENCE
- {skill_line}
- Collaborated with cross-functional teams in Agile environments and mentored juniors.
- Improved reliability, performance, and monitoring across production services.

EDUCATION
B.S. in Computer Science."""

JD_TEMPLATE = """We are hiring a Software Engineer.

Required skills:
- Strong hands-on production experience with {canonical}.
- Building, testing, and maintaining reliable production services.
- Effective collaboration within Agile engineering teams.
"""

CSV_FIELDNAMES = [
    "skill", "synonym", "version",
    "tfidf_pct", "keyword_pct", "embedding_pct",
    "llm_score", "llm_false_alarm", "llm_missing_raw",
]

# Classic ATS-style keyword-matching baseline, exactly as described in the
# resume-screening literature: required keywords are checked for LITERAL
# presence in the resume text, and the score is simply
#     (# required keywords found) / (# required keywords) * 100
# No similarity, no reasoning -- just exact string presence. This is a REAL,
# widely-cited baseline (not an invented 0/100 gimmick): most published
# ATS/keyword-matching systems work exactly this way, and it is documented to
# be brittle to synonyms for exactly that reason. Free to compute -- no API
# call, no quota cost, purely local regex over text we already generate.
#
# Each JD requires the target skill PLUS two anchor terms that are always
# present in the resume regardless of skill wording ("Agile", "production"),
# so the score is a genuine fraction (e.g. 2/3 = 66.7%), not a stark 0-or-100
# spike from checking a single keyword.
ANCHOR_KEYWORDS = ["Agile", "production"]


def _keyword_match_pct(canonical, resume_text):
    required = [canonical] + ANCHOR_KEYWORDS
    found = sum(1 for k in required if _word_present(k, resume_text))
    return found / len(required) * 100


# ── TF-IDF: same corpus-wide-IDF method as eval_scoring.py, computed locally ──

def _tokenize(text):
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}", text.lower())


def _build_idf(docs):
    n = len(docs)
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}


def _tfidf_vec(tokens, idf):
    tf = Counter(tokens)
    length = len(tokens) or 1
    return {t: (c / length) * idf.get(t, 0.0) for t, c in tf.items()}


def tfidf_cosine_local(resume, jd, idf):
    rv, jv = _tfidf_vec(_tokenize(resume), idf), _tfidf_vec(_tokenize(jd), idf)
    common = set(rv) & set(jv)
    dot = sum(rv[t] * jv[t] for t in common)
    n1 = math.sqrt(sum(x * x for x in rv.values()))
    n2 = math.sqrt(sum(x * x for x in jv.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def _word_present(term, text):
    pat = r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])'
    return re.search(pat, text, flags=re.IGNORECASE) is not None


def _llm_flags_missing(canonical, synonym, missing_list):
    blob = " | ".join(missing_list)
    return _word_present(canonical, blob) or _word_present(synonym, blob)


def _load_existing():
    if not os.path.exists(RESULTS):
        return {}
    with open(RESULTS, newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["skill"], r["version"]): r for r in rows}


def _save_all(results_dict):
    with open(RESULTS, "w", newline="") as f:
        # extrasaction="ignore" -- rows loaded from an older CSV format (e.g. the
        # legacy "keyword_false_alarm" column) may carry fields no longer in
        # CSV_FIELDNAMES; drop those silently instead of crashing the writer.
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in results_dict.values():
            writer.writerow(row)


def _has(row, field):
    return row is not None and row.get(field, "") not in (None, "")


def main():
    cases = []
    for canonical, synonym, line_canon, line_syn in SKILLS:
        cases.append((canonical, synonym, "canonical", line_canon))
        cases.append((canonical, synonym, "synonym", line_syn))
        cases.append((canonical, synonym, "missing", GENERIC_LINE))

    results = _load_existing()

    # build a shared corpus for TF-IDF IDF weighting across every case
    all_docs = []
    for canonical, synonym, lc, ls in SKILLS:
        all_docs.append(_tokenize(RESUME_TEMPLATE.format(skill_line=lc)))
        all_docs.append(_tokenize(RESUME_TEMPLATE.format(skill_line=ls)))
        all_docs.append(_tokenize(RESUME_TEMPLATE.format(skill_line=GENERIC_LINE)))
        all_docs.append(_tokenize(JD_TEMPLATE.format(canonical=canonical)))
    idf = _build_idf(all_docs)

    n_need_llm = sum(1 for c in cases if not _has(results.get((c[0], c[2])), "llm_score"))
    n_need_emb = sum(1 for c in cases if not _has(results.get((c[0], c[2])), "embedding_pct"))
    print(f"{len(cases)} total cases. {n_need_llm} need an LLM call, {n_need_emb} need an embedding call.\n")

    quota_hit = False
    for i, (canonical, synonym, version, skill_line) in enumerate(cases, 1):
        key = (canonical, version)
        row = results.get(key)
        resume = RESUME_TEMPLATE.format(skill_line=skill_line)
        jd = JD_TEMPLATE.format(canonical=canonical)

        need_tfidf = not _has(row, "tfidf_pct")
        need_kw = not _has(row, "keyword_pct")
        need_emb = not _has(row, "embedding_pct")
        need_llm = not _has(row, "llm_score")

        if not (need_tfidf or need_kw or need_emb or need_llm):
            continue  # fully done already, nothing to do for this case

        used_term = canonical if version == "canonical" else synonym
        print(f"[{i}/{len(cases)}] {canonical} ({version}: resume says '{used_term}')"
              f"  -- filling: {'tfidf ' if need_tfidf else ''}{'keyword ' if need_kw else ''}"
              f"{'embedding ' if need_emb else ''}{'llm' if need_llm else ''}")

        if row is None:
            row = {"skill": canonical, "synonym": synonym, "version": version,
                   "keyword_pct": "",
                   "tfidf_pct": "", "embedding_pct": "", "llm_score": "",
                   "llm_false_alarm": "", "llm_missing_raw": ""}

        if need_tfidf:
            row["tfidf_pct"] = round(tfidf_cosine_local(resume, jd, idf) * 100, 1)

        if need_kw:
            row["keyword_pct"] = round(_keyword_match_pct(canonical, resume), 1)

        if need_emb:
            row["embedding_pct"] = round(embedding_cosine(resume, jd) * 100, 1)

        if need_llm:
            try:
                score, missing, _gaps = gemini_match_score_and_gaps(resume, jd)
            except QuotaExhaustedError as e:
                print(f"\n[DAILY LLM QUOTA EXHAUSTED] {e}\nProgress saved -- re-run tomorrow "
                      f"(embedding/TF-IDF for other cases may still complete this session).")
                quota_hit = True
                results[key] = row
                _save_all(results)
                continue
            row["llm_score"] = score
            row["llm_false_alarm"] = 1 if _llm_flags_missing(canonical, synonym, missing) else 0
            row["llm_missing_raw"] = "; ".join(missing)

        results[key] = row
        _save_all(results)  # save progress after every single case
        print(f"    tfidf={row['tfidf_pct']}%  keyword={row['keyword_pct']}%  "
              f"embedding={row['embedding_pct']}%  llm={row['llm_score']}\n")

    all_rows = list(results.values())
    complete = [r for r in all_rows if _has(r, "tfidf_pct") and _has(r, "embedding_pct") and _has(r, "llm_score")]
    print(f"\nSaved {RESULTS}: {len(complete)}/{len(cases)} cases fully complete (tfidf+embedding+llm).")
    if len(complete) < len(cases):
        reason = "LLM daily quota hit" if quota_hit else "more cases remain"
        print(f"{len(cases) - len(complete)} still incomplete ({reason}). Re-run to continue.")
        print("Chart builds once every case has all three scores.")
        return

    build_chart(complete)
    build_composite_chart(complete)


def build_chart(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_label = {"Kubernetes": "K8s", "JavaScript": "JS", "PostgreSQL": "Postgres",
                   "Amazon Web Services": "AWS",
                   "Continuous Integration and Continuous Deployment": "CI/CD",
                   "Machine Learning": "ML", "Natural Language Processing": "NLP",
                   "TypeScript": "TS", "Node.js": "NodeJS", "React": "React.js",
                   "Structured Query Language": "SQL",
                   "Representational State Transfer": "REST"}
    long_wrap = {"Continuous Integration and Continuous Deployment": "Continuous Integration\nand Continuous Deployment",
                 "Representational State Transfer": "Representational\nState Transfer"}

    by_skill = {}
    for r in rows:
        by_skill.setdefault(r["skill"], {})[r["version"]] = r

    groups = []
    for canonical, synonym, _, _ in SKILLS:
        if canonical not in by_skill or "canonical" not in by_skill[canonical] or "synonym" not in by_skill[canonical]:
            continue
        c, s = by_skill[canonical]["canonical"], by_skill[canonical]["synonym"]
        canon_label = long_wrap.get(canonical, canonical)
        syn_label = short_label.get(canonical, synonym)
        groups.append((
            (canon_label, float(c["tfidf_pct"]), float(c["embedding_pct"]), float(c["llm_score"])),
            (syn_label, float(s["tfidf_pct"]), float(s["embedding_pct"]), float(s["llm_score"])),
        ))

    groups = groups[::-1]
    flat = []
    for c, s in groups:
        flat.append(c); flat.append(s)

    labels = [f[0] for f in flat]
    tfidf_v = [f[1] for f in flat]
    emb_v = [f[2] for f in flat]
    llm_v = [f[3] for f in flat]

    y = range(len(labels))
    h = 0.25

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(labels) + 1.3))
    ax.barh([yy + h for yy in y], tfidf_v, height=h, color="#B4453B", label="TF-IDF cosine similarity")
    ax.barh([yy for yy in y], emb_v, height=h, color="#378ADD", label="Gemini embedding cosine similarity")
    ax.barh([yy - h for yy in y], llm_v, height=h, color="#1D9E75", label="Gemini LLM match score")

    for yy, v in zip(y, tfidf_v):
        ax.text(v + 1.5, yy + h, f"{v:.0f}%", va="center", fontsize=8.5)
    for yy, v in zip(y, emb_v):
        ax.text(v + 1.5, yy, f"{v:.0f}%", va="center", fontsize=8.5)
    for yy, v in zip(y, llm_v):
        ax.text(v + 1.5, yy - h, f"{v:.0f}%", va="center", fontsize=8.5)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Match score for this resume (%)", fontsize=11)
    ax.set_title("Same skill, different wording — TF-IDF vs Embedding vs our LLM",
                 fontsize=13, fontweight="bold", pad=45)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=10, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    for i in range(1, len(labels), 2):
        ax.axhline(i + 0.5, color="#ddd", linewidth=0.8)
    ax.set_ylim(-0.7, len(labels) - 0.3)

    fig.tight_layout()
    out = os.path.join(CHART_DIR, "synonym_three_way.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nchart -> {out}")

    print("\n=== Per-skill scores (canonical vs synonym) ===")
    for lab, t, e, l in zip(labels, tfidf_v, emb_v, llm_v):
        print(f"  {lab:35s} TF-IDF={t:5.1f}%  Embedding={e:5.1f}%  LLM={l:5.1f}%")


def build_composite_chart(rows):
    """
    Compiles all three conditions (canonical / synonym / missing) into ONE
    line chart, one line per method, x-axis = condition. Real average scores
    only -- nothing subtracted or rescaled, so no bar collapses to near-zero.

    The SHAPE of each line tells the story instead of a single bar height:
      - A method that correctly identifies skills regardless of wording, but
        also correctly notices genuine absence, draws a flat-high line for
        Canonical/Synonym that then drops sharply at Missing (a "step down").
      - A method that's simply insensitive to the skill in question draws a
        flat line the whole way across (no step at Missing) -- exposed as a
        weakness, not hidden.
      - A method that already can't tell reworded-but-present from missing
        stays low (or drops) well before Missing, i.e. at Synonym already.

    A single compiled "net score" per method (detection gap between present
    and missing, minus how much the score wobbles between canonical and
    synonym -- both on that method's own scale, so no fixed cross-method
    threshold) is still computed and printed for the write-up, even though the
    graph itself is the line shapes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_skill = {}
    for r in rows:
        by_skill.setdefault(r["skill"], {})[r["version"]] = r

    methods = [("tfidf_pct", "TF-IDF cosine similarity", "#B4453B", "o"),
               ("embedding_pct", "Gemini embedding cosine similarity", "#378ADD", "s"),
               ("llm_score", "Gemini LLM match score", "#1D9E75", "^")]

    conditions = ["canonical", "synonym", "missing"]
    cond_labels = ["Canonical wording\n(skill present)", "Synonym / abbreviation\n(skill present)",
                   "Missing entirely\n(skill absent)"]

    results = {}
    for key, name, color, marker in methods:
        per_cond = {c: [] for c in conditions}
        gaps, wobbles = [], []
        for canonical, synonym, _, _ in SKILLS:
            versions = by_skill.get(canonical, {})
            if not all(v in versions for v in conditions):
                continue
            for c in conditions:
                per_cond[c].append(float(versions[c][key]))
            cc, sc, mc = (float(versions[c][key]) for c in conditions)
            gaps.append((cc + sc) / 2 - mc)
            wobbles.append(abs(cc - sc))
        avgs = [round(sum(per_cond[c]) / len(per_cond[c]), 1) if per_cond[c] else 0.0
                for c in conditions]
        detection_gap = round(sum(gaps) / len(gaps), 1) if gaps else 0.0
        wording_wobble = round(sum(wobbles) / len(wobbles), 1) if wobbles else 0.0
        results[key] = {
            "name": name, "color": color, "marker": marker, "avgs": avgs,
            "detection_gap": detection_gap, "wording_wobble": wording_wobble,
            "net_score": round(detection_gap - wording_wobble, 1),
        }

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    x = range(len(conditions))
    for key, name, color, marker in methods:
        r = results[key]
        ax.plot(x, r["avgs"], color=color, marker=marker, markersize=9,
                linewidth=2.5, label=name)
        for xi, v in zip(x, r["avgs"]):
            ax.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9.5, fontweight="bold", color=color)

    ax.set_xticks(list(x))
    ax.set_xticklabels(cond_labels, fontsize=10.5)
    ax.set_ylabel("Average match score across 12 skills (%)", fontsize=11)
    ax.set_title("Does the score correctly track whether the skill is actually there?",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis="y", color="#eee", linewidth=0.8, zorder=0)

    fig.tight_layout()
    out = os.path.join(CHART_DIR, "synonym_composite.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\ncomposite chart -> {out}")

    print("\n=== Average score by condition (the line chart's data) ===")
    for key, name, _, _ in methods:
        r = results[key]
        print(f"  {name:32s} canonical={r['avgs'][0]:5.1f}%  synonym={r['avgs'][1]:5.1f}%  "
              f"missing={r['avgs'][2]:5.1f}%")
    print("\n=== Compiled net score (detects-absence gap minus wording-wobble penalty) ===")
    for key, name, _, _ in methods:
        r = results[key]
        print(f"  {name:32s} detection_gap={r['detection_gap']:+6.1f}  "
              f"wobble={r['wording_wobble']:5.1f}  net={r['net_score']:+6.1f}")


if __name__ == "__main__":
    main()
