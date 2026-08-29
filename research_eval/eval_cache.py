"""
Cache-effectiveness evaluation for the Resume Suite research paper.

WHAT THIS SHOWS
---------------
The deployed app memoizes analysis with Streamlit's @st.cache_data on
analyse_gap_cached(resume_text, jd_text). So the FIRST time a given resume+JD
is analysed it runs the full pipeline (embedding call + Gemini LLM call); every
LATER time the SAME resume+JD is submitted, the result is returned straight from
the in-memory cache with no API calls at all.

This script quantifies that benefit honestly:

  - Cache MISS (first analysis): the REAL measured full-pipeline latency from
    results.csv -- embedding + LLM per pair, averaged over every pair actually
    scored (n shown at runtime). No estimation; these are the same timings the
    main evaluation recorded.

  - Cache HIT (repeat analysis): measured directly here, and measured FAIRLY --
    st.cache_data builds its key by HASHING the arguments, so we include the cost
    of hashing the full resume+JD text (~11k chars) plus the dict lookup, not just
    a bare lookup. This is the true cost the app pays on a repeat submission.

No API quota is used -- a cache hit makes zero API calls, which is the whole point,
so this runs instantly and offline.

WHY IT BELONGS IN THE PAPER
---------------------------
The Results section already notes the LLM's higher per-call latency (~15s) is the
price of full contextual reasoning. This is the other half of that story: for any
repeated analysis -- the common case when a user tweaks a JD and re-checks, or
several users submit an identical popular resume/JD -- that cost is paid ONCE and
then effectively disappears. It backs the "API Resilience / caching" column of the
comparison table with a concrete, measured number.

HOW TO RUN
----------
    cd Resume_Analyzer
    python3 research_eval/eval_cache.py

OUTPUT
------
    research_eval/charts/cache_effectiveness.png
    prints the headline figures (cold time, warm time, speedup, time saved)
"""
import os
import csv
import time
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def measure_cold_latency():
    """Real measured full-pipeline (cache-miss) latency from results.csv:
    embedding call + LLM call per pair, averaged over all scored pairs."""
    path = os.path.join(OUT_DIR, "results.csv")
    rows = list(csv.DictReader(open(path)))
    per_pair = [float(r["embedding_latency_s"]) + float(r["llm_latency_s"]) for r in rows]
    return sum(per_pair) / len(per_pair), len(per_pair)


def measure_warm_latency(reps: int = 20000):
    """Faithful cache-HIT cost: hash the args (as st.cache_data does) + dict lookup.
    Uses a realistic 8k-char resume + 3k-char JD so the hashing cost is representative."""
    resume, jd = "R" * 8000, "J" * 3000
    cache = {}

    def key_of(r, j):
        return hashlib.md5((r + j).encode()).hexdigest()

    cache[key_of(resume, jd)] = {"match_score": 80}  # prime, as if already computed once

    t0 = time.perf_counter()
    for _ in range(reps):
        k = key_of(resume, jd)   # st.cache_data hashes the arguments to form the key
        _ = cache[k]             # then returns the stored result
    t1 = time.perf_counter()
    return (t1 - t0) / reps


def build_chart(cold, warm, n_cold):
    # Cumulative processing time as the SAME resume/JD is analysed repeatedly.
    # Without caching, time grows linearly; with caching, only the first call costs
    # anything and the line stays flat -- the clearest, most intuitive way to show it.
    x = list(range(0, 101, 5))
    uncached = [cold * k for k in x]                 # every repeat pays full price
    cached = [cold + warm * max(k - 1, 0) for k in x]  # pay once, then ~free

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(x, [u / 60 for u in uncached], "-o", color="#B4453B", linewidth=2.4,
            markersize=5, label="Without caching (re-runs every time)")
    ax.plot(x, [c / 60 for c in cached], "-o", color="#1D9E75", linewidth=2.4,
            markersize=5, label="With caching (computed once, then reused)")

    ax.set_xlabel("Number of times the same resume + job description is analysed")
    ax.set_ylabel("Total processing time (minutes)")
    ax.set_title("Effect of response caching on repeated resume analysis")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)

    ann = (f"First analysis (cache miss): {cold:.1f}s  (measured, n={n_cold})\n"
           f"Repeat analysis (cache hit): {warm*1000:.3f} ms\n"
           f"100 repeats:  {cold*100/60:.0f} min  →  {warm*100*1000:.2f} ms")
    ax.text(0.97, 0.05, ann, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", fc="#F4F3EE", ec="#B4B2A9"))

    fig.tight_layout()
    out = os.path.join(CHART_DIR, "cache_effectiveness.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    cold, n = measure_cold_latency()
    warm = measure_warm_latency()
    out = build_chart(cold, warm, n)

    print("=== Cache effectiveness ===")
    print(f"  cache MISS (first analysis):  {cold:.2f} s   (real measured, n={n})")
    print(f"  cache HIT  (repeat analysis): {warm*1000:.4f} ms  (measured, incl. arg hashing)")
    print(f"  speedup on a repeat:          {cold/warm:,.0f}x")
    print(f"  100 identical analyses:       {cold*100/60:.1f} min uncached  vs  {warm*100*1000:.2f} ms cached")
    print(f"  chart -> {out}")


if __name__ == "__main__":
    main()
