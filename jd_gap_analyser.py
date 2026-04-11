import streamlit as st
import fitz
import os
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
os.makedirs("uploads", exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_pdf_text(file_bytes: bytes) -> str:
    with open("uploads/temp.pdf", "wb") as f:
        f.write(file_bytes)
    doc = fitz.open("uploads/temp.pdf")
    text = ""
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] == 0:
                text += block[4] + "\n"
    doc.close()
    return text


def build_vectorstore(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.create_documents([text])
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return FAISS.from_documents(chunks, embeddings)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def embed_text(text: str) -> list[float]:
    """Embed a short string with Gemini embedding model."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return embeddings.embed_query(text)


def analyse_gap(resume_text: str, jd_text: str) -> dict:
    """
    Ask Gemini to:
      1. Extract skill clusters from the JD.
      2. For each cluster, rate how well the resume covers it (0-100).
      3. Return missing keywords and tailored suggestions.
    """
    prompt = f"""
You are an expert technical recruiter and career coach.

Given the RESUME and JOB DESCRIPTION below, produce output in EXACTLY this format
(each field on its own line, no extra text):

SKILL_CLUSTERS: cluster1, cluster2, cluster3, cluster4, cluster5
CLUSTER_SCORES: cluster1:85, cluster2:60, cluster3:40, cluster4:20, cluster5:75
MISSING_KEYWORDS: keyword1, keyword2, keyword3, keyword4, keyword5, keyword6
MATCH_SCORE: (overall match 0-100, integer only)
STRENGTHS: strength1, strength2, strength3
CRITICAL_GAPS: gap1 | gap2 | gap3
SUGGESTIONS:
1. First concrete actionable suggestion.
2. Second concrete actionable suggestion.
3. Third concrete actionable suggestion.

Rules:
- SKILL_CLUSTERS must be the 5 most important skill areas the JD demands (e.g. "Machine Learning", "Cloud Infrastructure").
- CLUSTER_SCORES maps each cluster to how well the RESUME covers it (0-100).
- MISSING_KEYWORDS are exact keywords from the JD absent from the resume.
- MATCH_SCORE is the holistic match considering experience depth, not just keyword presence.
- SUGGESTIONS must reference specific resume content and specific JD requirements.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:2000]}
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw = response.text

    result = {
        "clusters": [],
        "cluster_scores": {},
        "missing_keywords": [],
        "match_score": 0,
        "strengths": [],
        "critical_gaps": [],
        "suggestions": [],
        "raw": raw,
    }

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("SKILL_CLUSTERS:"):
            result["clusters"] = [c.strip() for c in line.replace("SKILL_CLUSTERS:", "").split(",") if c.strip()]

        elif line.startswith("CLUSTER_SCORES:"):
            for pair in line.replace("CLUSTER_SCORES:", "").split(","):
                if ":" in pair:
                    parts = pair.strip().split(":")
                    if len(parts) == 2:
                        try:
                            result["cluster_scores"][parts[0].strip()] = int(
                                "".join(filter(str.isdigit, parts[1][:3]))
                            )
                        except ValueError:
                            pass

        elif line.startswith("MISSING_KEYWORDS:"):
            result["missing_keywords"] = [
                k.strip() for k in line.replace("MISSING_KEYWORDS:", "").split(",") if k.strip()
            ]

        elif line.startswith("MATCH_SCORE:"):
            try:
                val = line.split(":")[1].strip()
                result["match_score"] = int("".join(filter(str.isdigit, val[:3])))
            except (ValueError, IndexError):
                result["match_score"] = 0

        elif line.startswith("STRENGTHS:"):
            result["strengths"] = [s.strip() for s in line.replace("STRENGTHS:", "").split(",") if s.strip()]

        elif line.startswith("CRITICAL_GAPS:"):
            result["critical_gaps"] = [g.strip() for g in line.replace("CRITICAL_GAPS:", "").split("|") if g.strip()]

    in_suggestions = False
    sugg_lines = []
    for line in raw.split("\n"):
        if line.strip().startswith("SUGGESTIONS:"):
            in_suggestions = True
            continue
        if in_suggestions and line.strip():
            sugg_lines.append(line.strip())
    import re
    result["suggestions"] = [s for s in re.split(r"\d+\.", " ".join(sugg_lines)) if s.strip()][:3]

    return result


def radar_chart(clusters: list[str], scores: list[int]) -> go.Figure:
    cats = clusters + [clusters[0]]
    vals = scores + [scores[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(29,158,117,0.15)",
        line=dict(color="#1D9E75", width=2),
        name="Coverage",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def gauge_chart(score: int) -> go.Figure:
    color = "#E24B4A" if score < 40 else "#EF9F27" if score < 70 else "#1D9E75"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Overall match", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 10}},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#FCEBEB"},
                {"range": [40, 70], "color": "#FAEEDA"},
                {"range": [70, 100], "color": "#EAF3DE"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=0, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="JD Gap Analyser", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { font-size: 1.6rem !important; }

    /* keyword chips */
    .chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .chip-missing {
        background: #FCEBEB; color: #A32D2D;
        padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 500;
        border: 1px solid #F09595;
    }
    .chip-strength {
        background: #EAF3DE; color: #3B6D11;
        padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 500;
        border: 1px solid #97C459;
    }

    /* gap cards */
    .gap-card {
        background: #fff3cd; border-left: 4px solid #EF9F27;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        margin-bottom: 8px; color: #412402; font-size: 14px;
    }

    /* suggestion cards */
    .sugg-card {
        background: #ffffff; border: 1px solid #e0e0e0;
        border-left: 4px solid #1D9E75;
        padding: 12px 16px; border-radius: 0 10px 10px 0;
        margin-bottom: 10px; color: #111; font-size: 14px;
        line-height: 1.6;
    }

    /* similarity bar */
    .sim-bar-wrap { background: #e9ecef; border-radius: 8px; height: 12px; margin: 4px 0 2px; }
    .sim-bar { height: 12px; border-radius: 8px; background: #1D9E75; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 JD Gap Analyser")
st.write("Upload your resume and paste a job description to see exactly where you match — and where you don't.")

col_left, col_right = st.columns(2)

with col_left:
    uploaded_resume = st.file_uploader("Resume (PDF)", type=["pdf"], key="resume")

with col_right:
    jd_text = st.text_area(
        "Job description",
        placeholder="Paste the full job description here…",
        height=200,
        key="jd",
    )

run = st.button("Analyse gap", use_container_width=True)

# ── session state init ────────────────────────────────────────────────────────
if "gap_result" not in st.session_state:
    st.session_state.gap_result = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "jd_snapshot" not in st.session_state:
    st.session_state.jd_snapshot = ""

# ── run analysis ─────────────────────────────────────────────────────────────
if run:
    if not uploaded_resume:
        st.error("Please upload your resume PDF.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    with st.spinner("Extracting resume text…"):
        resume_text = extract_pdf_text(uploaded_resume.read())
        st.session_state.resume_text = resume_text

    if not resume_text.strip():
        st.error("Could not extract text from the PDF. Please use a text-based (non-scanned) PDF.")
        st.stop()

    with st.spinner("Running gap analysis with Gemini…"):
        result = analyse_gap(resume_text, jd_text)
        st.session_state.gap_result = result
        st.session_state.jd_snapshot = jd_text

    with st.spinner("Computing semantic similarity…"):
        resume_vec = embed_text(resume_text[:2000])
        jd_vec = embed_text(jd_text[:2000])
        sim = cosine_similarity(resume_vec, jd_vec)
        st.session_state.gap_result["semantic_sim"] = round(sim * 100, 1)

# ── results ───────────────────────────────────────────────────────────────────
if st.session_state.gap_result:
    r = st.session_state.gap_result

    st.divider()

    # Top metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.plotly_chart(gauge_chart(r["match_score"]), use_container_width=True)
    with m2:
        st.metric("Semantic similarity", f"{r.get('semantic_sim', 0)}%",
                  help="Cosine similarity between resume and JD embeddings")
        missing_count = len(r["missing_keywords"])
        st.metric("Missing keywords", missing_count,
                  delta=f"-{missing_count}" if missing_count else "0",
                  delta_color="inverse")
    with m3:
        st.metric("Critical gaps", len(r["critical_gaps"]))
        strength_count = len(r["strengths"])
        st.metric("Matched strengths", strength_count,
                  delta=f"+{strength_count}" if strength_count else "0")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Skill coverage", "Keywords", "Critical gaps", "Action plan"])

    # ── Tab 1: radar + bar chart ──────────────────────────────────────────
    with tab1:
        clusters = r["clusters"]
        scores = [r["cluster_scores"].get(c, 0) for c in clusters]

        if clusters and scores:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Coverage radar")
                st.plotly_chart(radar_chart(clusters, scores), use_container_width=True)

            with c2:
                st.subheader("Cluster breakdown")
                for cluster, score in zip(clusters, scores):
                    color = "#E24B4A" if score < 40 else "#EF9F27" if score < 70 else "#1D9E75"
                    st.markdown(f"**{cluster}** — {score}%")
                    st.markdown(
                        f'<div class="sim-bar-wrap"><div class="sim-bar" style="width:{score}%;background:{color}"></div></div>',
                        unsafe_allow_html=True,
                    )
                    label = "Weak" if score < 40 else "Partial" if score < 70 else "Strong"
                    st.caption(label)
        else:
            st.info("Could not parse cluster scores — see raw output below.")

    # ── Tab 2: keywords ───────────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Missing from your resume")
            if r["missing_keywords"]:
                chips = "".join(f'<span class="chip-missing">{k}</span>' for k in r["missing_keywords"])
                st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
                st.caption("These exact terms appear in the JD but not your resume. Adding them (where truthful) helps pass ATS filters.")
            else:
                st.success("No critical missing keywords found!")

        with col_b:
            st.subheader("Your matching strengths")
            if r["strengths"]:
                chips = "".join(f'<span class="chip-strength">{s}</span>' for s in r["strengths"])
                st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
            else:
                st.info("No specific strengths extracted.")

    # ── Tab 3: critical gaps ──────────────────────────────────────────────
    with tab3:
        st.subheader("Critical gaps")
        if r["critical_gaps"]:
            for gap in r["critical_gaps"]:
                st.markdown(f'<div class="gap-card">⚠️ {gap}</div>', unsafe_allow_html=True)
        else:
            st.success("No critical gaps identified — strong match!")

    # ── Tab 4: action plan ────────────────────────────────────────────────
    with tab4:
        st.subheader("What to do next")
        if r["suggestions"]:
            icons = ["1.", "2.", "3."]
            for i, sugg in enumerate(r["suggestions"]):
                label = icons[i] if i < len(icons) else f"{i+1}."
                st.markdown(
                    f'<div class="sugg-card"><b>{label}</b> {sugg}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.write("No suggestions parsed — see raw output below.")

        with st.expander("Raw Gemini output (debug)"):
            st.text(r.get("raw", ""))
