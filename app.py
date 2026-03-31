import streamlit as st
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🚀",
)

from PyPDF2 import PdfReader
import re
import spacy
import sqlite3
import matplotlib.pyplot as plt

users = {
    "admin": "1234",
    "shubham": "abcd"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

    st.stop()

nlp = spacy.load("en_core_web_sm")

conn = sqlite3.connect("resume_data.db")
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS results
             (name TEXT, score REAL)''')
conn.commit()

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_nlp_text(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ])

st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.markdown("""
# 🚀 AI Resume Analyzer Pro
### Smart ATS Scoring & Resume Insights
""")

uploaded_files = st.file_uploader(
    "Upload Resumes (Multiple Allowed)", 
    type=["pdf"], 
    accept_multiple_files=True
)

st.markdown("---")   # spacing line (optional but nice)

job_desc = st.text_area("Paste Job Description")

role = st.selectbox(
    "Select Job Role", 
    [
        "Data Scientist",
        "Web Developer",
        "Data Analyst",
        "ML Engineer",
        "Backend Developer",
        "Frontend Developer",
        "DevOps Engineer",
        "Software Engineer"
    ]
)

data_scientist_skills = [
    "python", "machine learning", "pandas", "numpy",
    "sql", "data analysis", "statistics", "deep learning"
]

web_dev_skills = [
    "html", "css", "javascript", "react", "nodejs"
]

data_scientist_skills = ["python", "machine learning", "data analysis", "pandas", "numpy"]

web_dev_skills = ["html", "css", "javascript", "react", "nodejs"]

data_analyst_skills = [
    "sql", "excel", "tableau", "power bi",
    "data analysis", "statistics"
]

ml_engineer_skills = [
    "python", "machine learning", "deep learning",
    "tensorflow", "pytorch", "model deployment"
]

backend_skills = [
    "python", "java", "nodejs", "sql", "api", "database"
]

frontend_skills = [
    "html", "css", "javascript", "react", "angular", "ui"
]

devops_skills = [
    "docker", "kubernetes", "aws", "ci/cd", "linux"
]

software_engineer_skills = [
    "python", "java", "c++", "data structures", "algorithms"
]

if uploaded_files and job_desc:
    c.execute("DELETE FROM results")
    conn.commit()

    if role == "Data Scientist":
        required_skills = data_scientist_skills

    elif role == "Web Developer":
        required_skills = web_dev_skills

    elif role == "Data Analyst":
        required_skills = data_analyst_skills

    elif role == "ML Engineer":
        required_skills = ml_engineer_skills

    elif role == "Backend Developer":
        required_skills = backend_skills

    elif role == "Frontend Developer":
        required_skills = frontend_skills

    elif role == "DevOps Engineer":
        required_skills = devops_skills

    else:
        required_skills = software_engineer_skills

    job_desc_clean = clean_nlp_text(job_desc)

    results = []

    for file in uploaded_files:
        resume_raw = extract_text(file)
        resume_text = clean_nlp_text(resume_raw)
        resume_text = resume_text.lower()

        matched_skills = [skill for skill in required_skills if skill in resume_text]
        missing_skills = [skill for skill in required_skills if skill not in resume_text]

        skills_score = (len(matched_skills) / len(required_skills)) * 100
        experience_score = 5  
        education_score = 10   

        final_score = (skills_score * 0.7) + (experience_score * 0.2) + (education_score * 0.1)

        results.append((file.name, final_score))

        c.execute("INSERT INTO results VALUES (?, ?)", (file.name, final_score))
        conn.commit()

        st.divider()

        clean_name = file.name[:25] + "..." if len(file.name) > 25 else file.name
        st.subheader(f"📄 {clean_name}")

        st.markdown("📊 Overall Match Summary")
        st.metric("Skill Match", f"{round(skills_score,2)}%")

        st.subheader("📊 ATS Score Breakdown")
        st.write(f"🧠 Skills: {round(skills_score,2)}%")
        st.write(f"💼 Experience: {experience_score}%")
        st.write(f"🎓 Education: {education_score}%")

        st.subheader("🏆 Final ATS Score")
        st.metric("Final ATS Score", f"{round(final_score,2)}%")
        st.progress(int(final_score))

        st.subheader("📊 Skill Match Visualization")

        labels = ['Matched', 'Missing']
        values = [len(matched_skills), len(missing_skills)]

        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(labels, values)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig)

        st.subheader("✅ Matched Skills")
        for skill in matched_skills:
            st.write(f"✔️ {skill}")

        st.subheader("❌ Missing Skills")
        for skill in missing_skills:
            st.write(f"❌ {skill}")

        if final_score > 70:
            st.success("🔥 Strong match for this role")
        elif final_score > 40:
            st.warning("⚠️ Moderate match, improve your skills")
        else:
            st.error("❌ Low match, you need improvement")

        st.subheader("💡 Recommendations")
        for skill in missing_skills:
            st.write(f"👉 Consider learning {skill} to improve your ATS score")

    st.divider()
    st.subheader("📊 Overall Resume Insights")

    total_resumes = len(results)
    avg_score = sum([score for _, score in results]) / total_resumes
    top_score = max([score for _, score in results])

    col1, col2, col3 = st.columns(3)

    col1.metric("📄 Total Resumes", total_resumes)
    col2.metric("📈 Average Score", f"{round(avg_score,2)}%")
    col3.metric("🏆 Highest Score", f"{round(top_score,2)}%")

    st.divider()
    st.subheader("🏆 Resume Ranking")

    results = sorted(results, key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(results, 1):
        clean_name = name[:25] + "..." if len(name) > 25 else name
        st.write(f"{i}. {clean_name} — {round(score,2)}%")

    st.divider()
    st.subheader("📁 Saved Results (Database)")

    for row in c.execute("SELECT * FROM results"):
        clean_name = row[0][:25] + "..." if len(row[0]) > 25 else row[0]
        st.write(f"{clean_name} — {row[1]}%")