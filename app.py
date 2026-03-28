import streamlit as st
import fitz
import os
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

def search_jobs(job_role):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {
        "query": f"{job_role} jobs in India",
        "num_pages": "1",
        "page": "1"
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    jobs = []
    if "data" in data:
        for job in data["data"][:5]:
            jobs.append({
                "title": job.get("job_title", "N/A"),
                "company": job.get("employer_name", "N/A"),
                "location": job.get("job_city", "N/A"),
                "link": job.get("job_apply_link", "#")
            })
    return jobs

st.set_page_config(page_title="Resume Analyser", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        width: 100%;
    }
    .stButton>button:hover { background-color: #45a049; }
    .job-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .job-card b { color: #000000 !important; }
    .job-card a { color: #4CAF50 !important; }
    .gap-card {
        background-color: #fff3cd;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #ffc107;
        color: #000000 !important;
    }
    .score-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

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
        st.session_state.skills = None
        st.session_state.soft_skills = None
        st.session_state.job_roles = None
        st.session_state.gaps = None
        st.session_state.match_percentages = {}

        with open("uploads/temp_resume.pdf", "wb") as f:
            f.write(uploaded_file.read())

        doc = fitz.open("uploads/temp_resume.pdf")
        text = ""
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                if block[6] == 0:
                    text += block[4] + "\n"
        doc.close()

        st.session_state.resume_text = text

        if text.strip():
            with st.spinner("Processing resume..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.create_documents([text])
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=os.getenv("GEMINI_API_KEY")
                )
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state.vectorstore = vectorstore

    if st.session_state.resume_text.strip():
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
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    st.session_state.analysis = response.text

                    lines = response.text.split("\n")
                    score = 75
                    skills = []
                    soft_skills = []
                    job_roles = []
                    match_percentages = {}
                    gaps = []

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        if line.startswith("RESUME_SCORE:"):
                            try:
                                val = line.split(":")[1].strip()
                                score = int(''.join(filter(str.isdigit, val[:3])))
                            except:
                                score = 75

                        elif line.startswith("SKILLS:"):
                            skills = [s.strip() for s in line.replace("SKILLS:", "").split(",") if s.strip()]

                        elif line.startswith("SOFT_SKILLS:"):
                            soft_skills = [s.strip() for s in line.replace("SOFT_SKILLS:", "").split(",") if s.strip()]

                        elif line.startswith("SUITABLE_JOB_ROLES:"):
                            job_roles = [r.strip().replace("*", "") for r in line.replace("SUITABLE_JOB_ROLES:", "").split(",") if r.strip()]

                        elif line.startswith("MATCH_PERCENTAGES:"):
                            pairs = line.replace("MATCH_PERCENTAGES:", "").split(",")
                            for pair in pairs:
                                if ":" in pair:
                                    parts = pair.strip().split(":")
                                    if len(parts) == 2:
                                        role = parts[0].strip().replace("*", "")
                                        try:
                                            pct = int(''.join(filter(str.isdigit, parts[1][:3])))
                                            match_percentages[role] = pct
                                        except:
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
                                {"range": [70, 100], "color": "#ccffcc"}
                            ]
                        }
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
                            st.markdown(f"<div class='gap-card'>❌ <b>{gap}</b></div>", unsafe_allow_html=True)
                    else:
                        st.info("No major skill gaps found!")

            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Technical Skills")
                    if st.session_state.skills:
                        fig = go.Figure(go.Bar(
                            x=list(range(len(st.session_state.skills[:10]))),
                            y=[85 - i*3 for i in range(len(st.session_state.skills[:10]))],
                            text=st.session_state.skills[:10],
                            textposition="outside",
                            marker_color="#4CAF50"
                        ))
                        fig.update_layout(
                            height=400,
                            xaxis={"showticklabels": False},
                            yaxis={"title": "Proficiency %"},
                            margin=dict(t=20, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Soft Skills")
                    if st.session_state.soft_skills:
                        fig = go.Figure(go.Bar(
                            x=list(range(len(st.session_state.soft_skills[:10]))),
                            y=[90 - i*2 for i in range(len(st.session_state.soft_skills[:10]))],
                            text=st.session_state.soft_skills[:10],
                            textposition="outside",
                            marker_color="#2196F3"
                        ))
                        fig.update_layout(
                            height=400,
                            xaxis={"showticklabels": False},
                            yaxis={"title": "Proficiency %"},
                            margin=dict(t=20, b=20)
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
<span style='font-size:16px; font-weight:bold; color:#000000;'>{job['title']}</span> 
<span style='color:#555555;'>at</span> 
<span style='font-weight:bold; color:#000000;'>{job['company']}</span><br>
<span style='color:#555555;'>📍 {job['location']}</span><br>
<a href='{job['link']}' target='_blank' style='color:#4CAF50; font-weight:bold;'>Apply Here →</a>
</div>""", unsafe_allow_html=True)

            with tab4:
                st.subheader("Improvement Tips")
                in_tips = False
                tips_text = ""
                for line in st.session_state.analysis.split("\n"):
                    if line.startswith("IMPROVEMENT_TIPS:"):
                        in_tips = True
                        tips_text += line.replace("IMPROVEMENT_TIPS:", "").strip() + "\n"
                    elif in_tips and line.strip():
                        tips_text += line.strip() + "\n"
                if tips_text.strip():
                    st.write(tips_text)
                else:
                    st.write(st.session_state.analysis)
    else:
        st.error("Could not extract text. Please upload a text-based PDF.")
