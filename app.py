import streamlit as st
import fitz
import os
import re
import numpy as np
import requests
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
os.makedirs("uploads", exist_ok=True)

# ── shared helpers ────────────────────────────────────────────────────────────

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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return FAISS.from_documents(chunks, embeddings)


def embed_text(text: str) -> list:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    return embeddings.embed_query(text)


def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def search_jobs(job_role: str) -> list:
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }
    params = {"query": f"{job_role} jobs in India", "num_pages": "1", "page": "1"}
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    jobs = []
    if "data" in data:
        for job in data["data"][:5]:
            jobs.append({
                "title": job.get("job_title", "N/A"),
                "company": job.get("employer_name", "N/A"),
                "location": job.get("job_city", "N/A"),
                "link": job.get("job_apply_link", "#"),
            })
    return jobs


# ── page config & global styles ───────────────────────────────────────────────

st.set_page_config(page_title="Resume Suite", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4CAF50; color: white;
        padding: 10px 24px; border-radius: 8px;
        border: none; font-size: 16px; width: 100%;
    }
    .stButton>button:hover { background-color: #45a049; }

    /* resume analyser cards */
    .job-card {
        background: #ffffff; padding: 15px; border-radius: 10px;
        margin-bottom: 10px; border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); color: #000 !important;
    }
    .gap-card-yellow {
        background: #fff3cd; padding: 10px 15px; border-radius: 8px;
        margin-bottom: 8px; border-left: 4px solid #ffc107; color: #000 !important;
    }
    .tip-card {
        background: #ffffff; padding: 20px; border-radius: 12px;
        margin-bottom: 15px; border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    /* JD gap analyser cards */
    .chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .chip-missing {
        background: #FCEBEB; color: #A32D2D;
        padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 500; border: 1px solid #F09595;
    }
    .chip-strength {
        background: #EAF3DE; color: #3B6D11;
        padding: 4px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 500; border: 1px solid #97C459;
    }
    .gap-card-orange {
        background: #fff3cd; border-left: 4px solid #EF9F27;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        margin-bottom: 8px; color: #412402; font-size: 14px;
    }
    .sugg-card {
        background: #ffffff; border: 1px solid #e0e0e0;
        border-left: 4px solid #1D9E75;
        padding: 12px 16px; border-radius: 0 10px 10px 0;
        margin-bottom: 10px; color: #111; font-size: 14px; line-height: 1.6;
    }
    .sim-bar-wrap { background: #e9ecef; border-radius: 8px; height: 12px; margin: 4px 0 2px; }
    .sim-bar { height: 12px; border-radius: 8px; background: #1D9E75; }
</style>
""", unsafe_allow_html=True)

# ── sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Resume Suite")
    st.markdown("---")
    page = st.radio(
        "Choose a tool",
        ["Resume Analyser", "JD Gap Analyser"],
        index=0,
    )
    st.markdown("---")
    st.caption("Built with Gemini AI · LangChain · FAISS")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — RESUME ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

if page == "Resume Analyser":

    st.title("Resume Analyser")
    st.write("Upload your resume and get instant AI-powered analysis with job matching!")

    uploaded_file = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])

    if uploaded_file is not None:
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name
            st.session_state.analysis = None
            st.session_state.vectorstore = None
            st.session_state.jobs = None
            st.session_state.score = None
            st.session_state.skills = []
            st.session_state.soft_skills = []
            st.session_state.job_roles = []
            st.session_state.gaps = []
            st.session_state.match_percentages = {}

            resume_bytes = uploaded_file.read()
            text = extract_pdf_text(resume_bytes)
            st.session_state.resume_text = text

            if text.strip():
                with st.spinner("Processing resume..."):
                    vectorstore = build_vectorstore(text)
                    st.session_state.vectorstore = vectorstore

        if st.session_state.get("resume_text", "").strip():
            st.success("Resume uploaded and processed successfully!")

            if st.session_state.analysis is None:
                if st.button("Analyse Resume"):
                    with st.spinner("Analysing your resume with Gemini AI..."):
                        prompt = f"""
Analyse this resume and provide output in EXACTLY this format, each on its own line:

RESUME_SCORE: (number out of 100 only)
SKILLS: (comma separated technical skills)
SOFT_SKILLS: (comma separated soft skills)
EXPERIENCE_SUMMARY: (2-3 sentences)
SUITABLE_JOB_ROLES: (exactly 5 roles comma separated)
MATCH_PERCENTAGES: Role1:85, Role2:78, Role3:72, Role4:65, Role5:60
SKILLS_GAP: skill1, skill2, skill3
IMPROVEMENT_TIPS: 1. tip one. 2. tip two. 3. tip three.

Resume:
{st.session_state.resume_text}
"""
                        response = client.models.generate_content(
                            model="gemini-2.5-flash", contents=prompt
                        )
                        st.session_state.analysis = response.text

                        lines = response.text.split("\n")
                        score = 75
                        skills, soft_skills, job_roles, gaps = [], [], [], []
                        match_percentages = {}

                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("RESUME_SCORE:"):
                                try:
                                    val = line.split(":")[1].strip()
                                    score = int("".join(filter(str.isdigit, val[:3])))
                                except Exception:
                                    score = 75
                            elif line.startswith("SKILLS:"):
                                skills = [s.strip() for s in line.replace("SKILLS:", "").split(",") if s.strip()]
                            elif line.startswith("SOFT_SKILLS:"):
                                soft_skills = [s.strip() for s in line.replace("SOFT_SKILLS:", "").split(",") if s.strip()]
                            elif line.startswith("SUITABLE_JOB_ROLES:"):
                                job_roles = [r.strip().replace("*", "") for r in line.replace("SUITABLE_JOB_ROLES:", "").split(",") if r.strip()]
                            elif line.startswith("MATCH_PERCENTAGES:"):
                                for pair in line.replace("MATCH_PERCENTAGES:", "").split(","):
                                    if ":" in pair:
                                        parts = pair.strip().split(":")
                                        if len(parts) == 2:
                                            role = parts[0].strip().replace("*", "")
                                            try:
                                                pct = int("".join(filter(str.isdigit, parts[1][:3])))
                                                match_percentages[role] = pct
                                            except Exception:
                                                pass
                            elif line.startswith("SKILLS_GAP:"):
                                gaps = [g.strip() for g in line.replace("SKILLS_GAP:", "").split(",") if g.strip()]

                        st.session_state.score = score
                        st.session_state.skills = skills
                        st.session_state.soft_skills = soft_skills
                        st.session_state.job_roles = job_roles
                        st.session_state.match_percentages = match_percentages
                        st.session_state.gaps = gaps
                        st.session_state.job_role = job_roles[0] if job_roles else "Software Developer"

                    with st.spinner("Finding live job listings..."):
                        st.session_state.jobs = search_jobs(st.session_state.job_role)

            if st.session_state.analysis:
                tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Skills Analysis", "Job Matches", "Improvement Tips"])

                with tab1:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=st.session_state.score,
                            title={"text": "Resume Score"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#4CAF50"},
                                "steps": [
                                    {"range": [0, 40], "color": "#ffcccc"},
                                    {"range": [40, 70], "color": "#fff3cc"},
                                    {"range": [70, 100], "color": "#ccffcc"},
                                ],
                            },
                        ))
                        fig.update_layout(height=250, margin=dict(t=30, b=0, l=20, r=20))
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("Experience Summary")
                        for line in st.session_state.analysis.split("\n"):
                            if line.startswith("EXPERIENCE_SUMMARY:"):
                                st.write(line.replace("EXPERIENCE_SUMMARY:", "").strip())
                                break
                        st.subheader("Skills Gap")
                        if st.session_state.gaps:
                            for gap in st.session_state.gaps:
                                st.markdown(f"<div class='gap-card-yellow'>❌ <b>{gap}</b></div>", unsafe_allow_html=True)
                        else:
                            st.info("No major skill gaps found!")

                with tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Technical Skills")
                        if st.session_state.skills:
                            fig = go.Figure(go.Bar(
                                x=list(range(len(st.session_state.skills[:10]))),
                                y=[85 - i * 3 for i in range(len(st.session_state.skills[:10]))],
                                text=st.session_state.skills[:10],
                                textposition="outside",
                                marker_color="#4CAF50",
                            ))
                            fig.update_layout(
                                height=400,
                                xaxis={"showticklabels": False},
                                yaxis={"title": "Proficiency %"},
                                margin=dict(t=20, b=20),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("Soft Skills")
                        if st.session_state.soft_skills:
                            fig = go.Figure(go.Bar(
                                x=list(range(len(st.session_state.soft_skills[:10]))),
                                y=[90 - i * 2 for i in range(len(st.session_state.soft_skills[:10]))],
                                text=st.session_state.soft_skills[:10],
                                textposition="outside",
                                marker_color="#2196F3",
                            ))
                            fig.update_layout(
                                height=400,
                                xaxis={"showticklabels": False},
                                yaxis={"title": "Proficiency %"},
                                margin=dict(t=20, b=20),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.subheader("Job Role Match Percentages")
                    if st.session_state.match_percentages:
                        for role, pct in st.session_state.match_percentages.items():
                            st.markdown(f"**{role}**")
                            st.progress(pct / 100)
                            st.markdown(f"Match: **{pct}%**")
                            st.markdown("---")
                    elif st.session_state.job_roles:
                        for i, role in enumerate(st.session_state.job_roles):
                            pct = max(60, 90 - i * 7)
                            st.markdown(f"**{role}**")
                            st.progress(pct / 100)
                            st.markdown(f"Match: **{pct}%**")
                            st.markdown("---")

                    st.subheader(f"Live Job Listings for: {st.session_state.job_role}")
                    if st.session_state.jobs:
                        for job in st.session_state.jobs:
                            st.markdown(f"""<div class='job-card'>
<span style='font-size:16px;font-weight:bold;color:#000;'>{job['title']}</span>
<span style='color:#555;'> at </span>
<span style='font-weight:bold;color:#000;'>{job['company']}</span><br>
<span style='color:#555;'>📍 {job['location']}</span><br>
<a href='{job['link']}' target='_blank' style='color:#4CAF50;font-weight:bold;'>Apply Here →</a>
</div>""", unsafe_allow_html=True)

                with tab4:
                    st.subheader("Improvement Tips")
                    in_tips = False
                    current_tip = ""
                    for line in st.session_state.analysis.split("\n"):
                        if line.startswith("IMPROVEMENT_TIPS:"):
                            in_tips = True
                            current_tip += line.replace("IMPROVEMENT_TIPS:", "").strip() + " "
                        elif in_tips and line.strip():
                            current_tip += line.strip() + " "

                    tips = re.split(r"\d+\.", current_tip)
                    tips = [t.strip() for t in tips if t.strip()]
                    icons = ["💡", "📝", "🚀"]
                    titles = ["Tip 1", "Tip 2", "Tip 3"]

                    if tips:
                        for i, tip in enumerate(tips[:3]):
                            st.markdown(f"""<div class='tip-card'>
<h4 style='color:#4CAF50;margin:0 0 8px 0;'>{icons[i] if i < len(icons) else "💡"} {titles[i] if i < len(titles) else f"Tip {i+1}"}</h4>
<p style='color:#000;margin:0;font-size:15px;line-height:1.6;'>{tip}</p>
</div>""", unsafe_allow_html=True)
                    else:
                        st.write(current_tip)
        else:
            st.error("Could not extract text. Please upload a text-based PDF.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — JD GAP ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "JD Gap Analyser":

    def analyse_gap(resume_text: str, jd_text: str) -> dict:
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
- SKILL_CLUSTERS must be the 5 most important skill areas the JD demands.
- CLUSTER_SCORES maps each cluster to how well the RESUME covers it (0-100).
- MISSING_KEYWORDS are exact keywords from the JD absent from the resume.
- MATCH_SCORE is the holistic match considering experience depth.
- SUGGESTIONS must reference specific resume content and specific JD requirements.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:2000]}
"""
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        raw = response.text

        result = {
            "clusters": [], "cluster_scores": {}, "missing_keywords": [],
            "match_score": 0, "strengths": [], "critical_gaps": [],
            "suggestions": [], "raw": raw,
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
                result["missing_keywords"] = [k.strip() for k in line.replace("MISSING_KEYWORDS:", "").split(",") if k.strip()]
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
        result["suggestions"] = [s for s in re.split(r"\d+\.", " ".join(sugg_lines)) if s.strip()][:3]

        return result

    def radar_chart(clusters, scores):
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

    def gauge_chart_jd(score):
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

    # ── JD Gap UI ─────────────────────────────────────────────────────────────

    st.title("🎯 JD Gap Analyser")
    st.write("Upload your resume and paste a job description to see exactly where you match — and where you don't.")

    col_left, col_right = st.columns(2)
    with col_left:
        uploaded_resume = st.file_uploader("Resume (PDF)", type=["pdf"], key="jd_resume")
    with col_right:
        jd_text = st.text_area(
            "Job description",
            placeholder="Paste the full job description here…",
            height=200,
            key="jd_text",
        )

    run = st.button("Analyse gap", use_container_width=True)

    if "gap_result" not in st.session_state:
        st.session_state.gap_result = None

    if run:
        if not uploaded_resume:
            st.error("Please upload your resume PDF.")
            st.stop()
        if not jd_text.strip():
            st.error("Please paste a job description.")
            st.stop()

        with st.spinner("Extracting resume text…"):
            resume_text = extract_pdf_text(uploaded_resume.read())

        if not resume_text.strip():
            st.error("Could not extract text from the PDF. Please use a text-based PDF.")
            st.stop()

        with st.spinner("Running gap analysis with Gemini…"):
            result = analyse_gap(resume_text, jd_text)
            st.session_state.gap_result = result

        with st.spinner("Computing semantic similarity…"):
            resume_vec = embed_text(resume_text[:2000])
            jd_vec = embed_text(jd_text[:2000])
            sim = cosine_similarity(resume_vec, jd_vec)
            st.session_state.gap_result["semantic_sim"] = round(sim * 100, 1)

    if st.session_state.gap_result:
        r = st.session_state.gap_result
        st.divider()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.plotly_chart(gauge_chart_jd(r["match_score"]), use_container_width=True)
        with m2:
            st.metric("Semantic similarity", f"{r.get('semantic_sim', 0)}%")
            missing_count = len(r["missing_keywords"])
            st.metric("Missing keywords", missing_count,
                      delta=f"-{missing_count}" if missing_count else "0",
                      delta_color="inverse")
        with m3:
            st.metric("Critical gaps", len(r["critical_gaps"]))
            st.metric("Matched strengths", len(r["strengths"]),
                      delta=f"+{len(r['strengths'])}" if r["strengths"] else "0")

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(["Skill coverage", "Keywords", "Critical gaps", "Action plan"])

        with tab1:
            clusters = r["clusters"]
            scores = [r["cluster_scores"].get(c, 0) for c in clusters]
            if clusters and scores:
                c1, c2 = st.columns(2)
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
                        st.caption("Weak" if score < 40 else "Partial" if score < 70 else "Strong")
            else:
                st.info("Could not parse cluster scores.")

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Missing from your resume")
                if r["missing_keywords"]:
                    chips = "".join(f'<span class="chip-missing">{k}</span>' for k in r["missing_keywords"])
                    st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
                    st.caption("These keywords appear in the JD but not your resume. Adding them (where truthful) helps pass ATS filters.")
                else:
                    st.success("No critical missing keywords found!")
            with col_b:
                st.subheader("Your matching strengths")
                if r["strengths"]:
                    chips = "".join(f'<span class="chip-strength">{s}</span>' for s in r["strengths"])
                    st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.info("No specific strengths extracted.")

        with tab3:
            st.subheader("Critical gaps")
            if r["critical_gaps"]:
                for gap in r["critical_gaps"]:
                    st.markdown(f'<div class="gap-card-orange">⚠️ {gap}</div>', unsafe_allow_html=True)
            else:
                st.success("No critical gaps identified — strong match!")

        with tab4:
            st.subheader("What to do next")
            if r["suggestions"]:
                for i, sugg in enumerate(r["suggestions"]):
                    st.markdown(
                        f'<div class="sugg-card"><b>{i+1}.</b> {sugg}</div>',
                        unsafe_allow_html=True,
                    )
            with st.expander("Raw Gemini output (debug)"):
                st.text(r.get("raw", ""))
