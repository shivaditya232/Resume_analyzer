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


def retrieve_context(vectorstore: FAISS, query: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)


def chat_with_resume(
    vectorstore: FAISS,
    resume_text: str,
    user_message: str,
    chat_history: list,
) -> str:
    context = retrieve_context(vectorstore, user_message)
    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a smart, friendly career coach who has deeply read the candidate's resume.
Answer the user's question using ONLY the resume content provided.
Be concise, specific, and encouraging. If the answer is not in the resume, say so honestly.
Never make up experience or skills that are not in the resume.

RESUME CONTEXT (most relevant sections):
{context}

FULL RESUME SUMMARY (first 1500 chars):
{resume_text[:1500]}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {user_message}

Answer:"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()


def analyse_gap(resume_text: str, jd_text: str) -> dict:
    prompt = f"""
You are an expert technical recruiter and career coach.
Given the RESUME and JOB DESCRIPTION below, produce output in EXACTLY this format:

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

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:2000]}
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw = response.text
    result = {"clusters": [], "cluster_scores": {}, "missing_keywords": [],
              "match_score": 0, "strengths": [], "critical_gaps": [], "suggestions": [], "raw": raw}

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
                                "".join(filter(str.isdigit, parts[1][:3])))
                        except ValueError:
                            pass
        elif line.startswith("MISSING_KEYWORDS:"):
            result["missing_keywords"] = [k.strip() for k in line.replace("MISSING_KEYWORDS:", "").split(",") if k.strip()]
        elif line.startswith("MATCH_SCORE:"):
            try:
                result["match_score"] = int("".join(filter(str.isdigit, line.split(":")[1].strip()[:3])))
            except (ValueError, IndexError):
                pass
        elif line.startswith("STRENGTHS:"):
            result["strengths"] = [s.strip() for s in line.replace("STRENGTHS:", "").split(",") if s.strip()]
        elif line.startswith("CRITICAL_GAPS:"):
            result["critical_gaps"] = [g.strip() for g in line.replace("CRITICAL_GAPS:", "").split("|") if g.strip()]

    in_s, sugg_lines = False, []
    for line in raw.split("\n"):
        if line.strip().startswith("SUGGESTIONS:"):
            in_s = True
            continue
        if in_s and line.strip():
            sugg_lines.append(line.strip())
    result["suggestions"] = [s for s in re.split(r"\d+\.", " ".join(sugg_lines)) if s.strip()][:3]
    return result


def radar_chart(clusters, scores):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]], theta=clusters + [clusters[0]], fill="toself",
        fillcolor="rgba(29,158,117,0.15)", line=dict(color="#1D9E75", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, margin=dict(t=20, b=20, l=40, r=40),
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def gauge_chart(score):
    color = "#E24B4A" if score < 40 else "#EF9F27" if score < 70 else "#1D9E75"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text": "Overall match", "font": {"size": 13}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 40], "color": "#FCEBEB"},
                          {"range": [40, 70], "color": "#FAEEDA"},
                          {"range": [70, 100], "color": "#EAF3DE"}]},
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=0, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── page config & styles ──────────────────────────────────────────────────────

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
    .chip-wrap { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .chip-missing {
        background: #FCEBEB; color: #A32D2D; padding: 4px 12px;
        border-radius: 20px; font-size: 13px; font-weight: 500; border: 1px solid #F09595;
    }
    .chip-strength {
        background: #EAF3DE; color: #3B6D11; padding: 4px 12px;
        border-radius: 20px; font-size: 13px; font-weight: 500; border: 1px solid #97C459;
    }
    .gap-card {
        background: #fff3cd; border-left: 4px solid #EF9F27;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        margin-bottom: 8px; color: #412402; font-size: 14px;
    }
    .sugg-card {
        background: #ffffff; border: 1px solid #e0e0e0;
        border-left: 4px solid #1D9E75; padding: 12px 16px;
        border-radius: 0 10px 10px 0; margin-bottom: 10px;
        color: #111; font-size: 14px; line-height: 1.6;
    }
    .sim-bar-wrap { background: #e9ecef; border-radius: 8px; height: 12px; margin: 4px 0 2px; }
    .sim-bar { height: 12px; border-radius: 8px; }
    .chat-bubble-user {
        background: #4CAF50; color: white;
        padding: 10px 16px; border-radius: 18px 18px 4px 18px;
        margin: 6px 0; max-width: 80%; margin-left: auto;
        font-size: 14px; line-height: 1.5;
    }
    .chat-bubble-bot {
        background: #ffffff; color: #111;
        padding: 10px 16px; border-radius: 18px 18px 18px 4px;
        margin: 6px 0; max-width: 80%;
        font-size: 14px; line-height: 1.5; border: 1px solid #e0e0e0;
    }
    .resume-status {
        background: #EAF3DE; border: 1px solid #97C459;
        border-radius: 8px; padding: 10px 16px;
        color: #3B6D11; font-size: 14px; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 Resume Suite")
    st.markdown("---")

    # resume upload lives in sidebar — shared across both pages
    st.subheader("Step 1 — Upload Resume")
    sidebar_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"], key="sidebar_resume")

    if sidebar_resume is not None:
        if "last_file" not in st.session_state or st.session_state.last_file != sidebar_resume.name:
            st.session_state.last_file = sidebar_resume.name
            st.session_state.vectorstore = None
            st.session_state.resume_text = ""
            st.session_state.chat_history = []
            st.session_state.gap_result = None

            with st.spinner("Reading resume..."):
                text = extract_pdf_text(sidebar_resume.read())
                st.session_state.resume_text = text

            if text.strip():
                with st.spinner("Building knowledge base..."):
                    st.session_state.vectorstore = build_vectorstore(text)
                st.success("Resume ready!")
            else:
                st.error("Could not extract text. Use a text-based PDF.")

    elif st.session_state.get("resume_text"):
        st.markdown("<div class='resume-status'>Resume loaded and ready.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Step 2 — Choose a tool")
    page = st.radio(
        "",
        ["JD Gap Analyser", "Resume Chatbot"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")

    if page == "Resume Chatbot" and st.session_state.get("chat_history"):
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

    st.caption("Built with Gemini AI · LangChain · FAISS")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — JD GAP ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

if page == "JD Gap Analyser":

    st.title("🎯 JD Gap Analyser")
    st.write("Paste a job description below and see exactly where your resume matches — and where it doesn't.")

    if not st.session_state.get("resume_text"):
        st.warning("Upload your resume in the sidebar first.")
        st.stop()

    jd_text = st.text_area(
        "Job description",
        placeholder="Paste the full job description here…",
        height=220,
        key="jd_text",
    )

    if "gap_result" not in st.session_state:
        st.session_state.gap_result = None

    if st.button("Analyse gap", use_container_width=True):
        if not jd_text.strip():
            st.error("Please paste a job description.")
            st.stop()

        with st.spinner("Running gap analysis with Gemini…"):
            result = analyse_gap(st.session_state.resume_text, jd_text)
            st.session_state.gap_result = result

        with st.spinner("Computing semantic similarity…"):
            sim = cosine_similarity(
                embed_text(st.session_state.resume_text[:2000]),
                embed_text(jd_text[:2000]),
            )
            st.session_state.gap_result["semantic_sim"] = round(sim * 100, 1)

    if st.session_state.gap_result:
        r = st.session_state.gap_result
        st.divider()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.plotly_chart(gauge_chart(r["match_score"]), use_container_width=True)
        with m2:
            st.metric("Semantic similarity", f"{r.get('semantic_sim', 0)}%")
            mc = len(r["missing_keywords"])
            st.metric("Missing keywords", mc,
                      delta=f"-{mc}" if mc else "0", delta_color="inverse")
        with m3:
            st.metric("Critical gaps", len(r["critical_gaps"]))
            st.metric("Matched strengths", len(r["strengths"]),
                      delta=f"+{len(r['strengths'])}" if r["strengths"] else "0")

        st.divider()
        t1, t2, t3, t4 = st.tabs(["Skill coverage", "Keywords", "Critical gaps", "Action plan"])

        with t1:
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
                            unsafe_allow_html=True)
                        st.caption("Weak" if score < 40 else "Partial" if score < 70 else "Strong")
            else:
                st.info("Could not parse cluster scores.")

        with t2:
            ca, cb = st.columns(2)
            with ca:
                st.subheader("Missing from your resume")
                if r["missing_keywords"]:
                    chips = "".join(f'<span class="chip-missing">{k}</span>' for k in r["missing_keywords"])
                    st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
                    st.caption("Adding these keywords (where truthful) helps pass ATS filters.")
                else:
                    st.success("No critical missing keywords!")
            with cb:
                st.subheader("Your matching strengths")
                if r["strengths"]:
                    chips = "".join(f'<span class="chip-strength">{s}</span>' for s in r["strengths"])
                    st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
                else:
                    st.info("No specific strengths extracted.")

        with t3:
            st.subheader("Critical gaps")
            if r["critical_gaps"]:
                for gap in r["critical_gaps"]:
                    st.markdown(f'<div class="gap-card">⚠️ {gap}</div>', unsafe_allow_html=True)
            else:
                st.success("No critical gaps — strong match!")

        with t4:
            st.subheader("What to do next")
            for i, sugg in enumerate(r["suggestions"]):
                st.markdown(f'<div class="sugg-card"><b>{i+1}.</b> {sugg}</div>', unsafe_allow_html=True)
            with st.expander("Raw Gemini output (debug)"):
                st.text(r.get("raw", ""))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RESUME CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Resume Chatbot":

    st.title("💬 Resume Chatbot")
    st.write("Ask anything about your resume — strengths, gaps, interview prep, or career advice.")

    if not st.session_state.get("vectorstore"):
        st.warning("Upload your resume in the sidebar first to enable the chatbot.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # suggested questions
    st.markdown("**Try asking:**")
    suggestions = [
        "What are my strongest technical skills?",
        "Am I ready for a senior role?",
        "What projects stand out in my resume?",
        "What skills should I learn next?",
        "How would you summarise my experience?",
        "What kind of companies should I target?",
    ]
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"sugg_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": suggestion})
                with st.spinner("Thinking..."):
                    reply = chat_with_resume(
                        st.session_state.vectorstore,
                        st.session_state.get("resume_text", ""),
                        suggestion,
                        st.session_state.chat_history,
                    )
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    st.markdown("---")

    # chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='display:flex;justify-content:flex-end;margin:4px 0;'>"
                    f"<div class='chat-bubble-user'>{msg['content']}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='display:flex;justify-content:flex-start;margin:4px 0;'>"
                    f"<div class='chat-bubble-bot'>{msg['content']}</div></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No messages yet. Click a suggestion above or type your question below.")

    st.markdown("---")

    # chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask anything about your resume…",
                placeholder="e.g. What are my weakest areas?",
                label_visibility="collapsed",
            )
        with col2:
            send = st.form_submit_button("Send", use_container_width=True)

    if send and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("Thinking..."):
            reply = chat_with_resume(
                st.session_state.vectorstore,
                st.session_state.get("resume_text", ""),
                user_input.strip(),
                st.session_state.chat_history,
            )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
