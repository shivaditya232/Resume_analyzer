import streamlit as st
import fitz
import os
import requests
from dotenv import load_dotenv
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

st.set_page_config(page_title="Resume Analyser", page_icon="📄")
st.title("Resume Analyser")
st.write("Upload your resume and get instant analysis!")

uploaded_file = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        st.session_state.analysis = None
        st.session_state.vectorstore = None
        st.session_state.jobs = None

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
                with st.spinner("Analysing your resume with Gemini..."):
                    prompt = f"""
                    Analyse this resume and provide the following in a clean format:

                    1. SKILLS: List all technical and soft skills found
                    2. EXPERIENCE SUMMARY: Brief summary of experience
                    3. SUITABLE JOB ROLES: Top 5 job roles that match this resume (just the role names, comma separated on one line after the colon)
                    4. IMPROVEMENT TIPS: Top 3 suggestions to improve the resume

                    Resume:
                    {st.session_state.resume_text}
                    """

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    st.session_state.analysis = response.text

                    lines = response.text.split("\n")
                    job_role = "Software Developer"
                    for line in lines:
                        if "SUITABLE JOB ROLES" in line.upper() and ":" in line:
                            roles_part = line.split(":")[1].strip()
                            if roles_part:
                                job_role = roles_part.split(",")[0].strip()
                                job_role = job_role.replace("*", "").strip()
                                break

                with st.spinner("Finding real job listings..."):
                    st.session_state.jobs = search_jobs(job_role)
                    st.session_state.job_role = job_role

        if st.session_state.analysis:
            st.subheader("Resume Analysis")
            st.write(st.session_state.analysis)

            if st.session_state.jobs:
                st.subheader(f"Live Job Listings for: {st.session_state.job_role}")
                for job in st.session_state.jobs:
                    st.markdown(f"""
                    **{job['title']}** at **{job['company']}**
                    {job['location']}
                    [Apply Here]({job['link']})
                    ---
                    """)
            else:
                st.info("No job listings found. Try again later.")
    else:
        st.error("Could not extract text. Please upload a text-based PDF.")
