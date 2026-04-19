import streamlit as st
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🚀")

from PyPDF2 import PdfReader
import spacy
import sqlite3
import matplotlib.pyplot as plt

# ---------------- LOGIN ---------------- #
users = {"admin": "1234", "shubham": "abcd"}

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
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# ---------------- NLP ---------------- #
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("resume_data.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS results (name TEXT, score REAL)")
conn.commit()

# ---------------- FUNCTIONS ---------------- #
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def clean_nlp_text(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ])

# ---------------- UI ---------------- #
st.sidebar.write(f"👤 {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.title("🚀 AI Resume Analyzer Pro")
st.write("Smart ATS Scoring & Resume Insights")

uploaded_files = st.file_uploader(
    "Upload Resumes (Multiple Allowed)", 
    type=["pdf"], 
    accept_multiple_files=True
)

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

# ---------------- SKILLS ---------------- #
skills_dict = {
    "Data Scientist": ["python", "machine learning", "data analysis", "pandas", "numpy"],
    "Web Developer": ["html", "css", "javascript", "react", "nodejs"],
    "Data Analyst": ["sql", "excel", "tableau", "power bi", "statistics"],
    "ML Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch"],
    "Backend Developer": ["python", "java", "nodejs", "sql", "api", "database"],
    "Frontend Developer": ["html", "css", "javascript", "react", "angular"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "ci/cd", "linux"],
    "Software Engineer": ["python", "java", "c++", "data structures", "algorithms"]
}

# ---------------- MAIN LOGIC ---------------- #
if uploaded_files and job_desc:

    c.execute("DELETE FROM results")
    conn.commit()

    required_skills = skills_dict[role]
    job_desc_clean = clean_nlp_text(job_desc)

    results = []

    for i, file in enumerate(uploaded_files):

        resume_raw = extract_text(file)
        resume_text = clean_nlp_text(resume_raw)

        matched_skills = [s for s in required_skills if s in resume_text]
        missing_skills = [s for s in required_skills if s not in resume_text]

        skills_score = (len(matched_skills) / len(required_skills)) * 100

        # simple scoring
        experience_score = 5
        education_score = 10

        final_score = (skills_score * 0.7) + (experience_score * 0.2) + (education_score * 0.1)

        results.append((file.name, final_score))

        c.execute("INSERT INTO results VALUES (?, ?)", (file.name, final_score))
        conn.commit()

        # ---------------- DISPLAY ---------------- #
        st.divider()

        clean_name = file.name[:25] + "..." if len(file.name) > 25 else file.name
        st.subheader(f"📄 {clean_name}")

        st.metric("Skill Match", f"{round(skills_score,2)}%")

        st.subheader("📊 ATS Score Breakdown")
        st.write(f"🧠 Skills: {round(skills_score,2)}%")
        st.write(f"💼 Experience: {experience_score}%")
        st.write(f"🎓 Education: {education_score}%")

        st.subheader("🏆 Final ATS Score")
        st.metric("Final Score", f"{round(final_score,2)}%")
        st.progress(int(final_score))

        # -------- FIXED GRAPH (NO ERROR) -------- #
        st.subheader("📊 Skill Match Visualization")

        labels = ['Matched', 'Missing']
        values = [len(matched_skills), len(missing_skills)]

        fig, ax = plt.subplots()
        ax.bar(labels, values)

        st.pyplot(fig)   # matplotlib doesn't need key → no error

        # -------- SKILLS -------- #
        st.subheader("✅ Matched Skills")
        for s in matched_skills:
            st.write(f"✔️ {s}")

        st.subheader("❌ Missing Skills")
        for s in missing_skills:
            st.write(f"❌ {s}")

        # -------- FEEDBACK -------- #
        if final_score > 70:
            st.success("🔥 Strong match")
        elif final_score > 40:
            st.warning("⚠️ Moderate match")
        else:
            st.error("❌ Low match")

        # -------- RECOMMENDATIONS -------- #
        st.subheader("💡 Recommendations")
        for s in missing_skills:
            st.write(f"👉 Learn {s}")

    # ---------------- OVERALL ---------------- #
    st.divider()
    st.subheader("📊 Overall Resume Insights")

    total_resumes = len(results)
    avg_score = sum([r[1] for r in results]) / total_resumes
    top_score = max([r[1] for r in results])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resumes", total_resumes)
    col2.metric("Average Score", round(avg_score, 2))
    col3.metric("Top Score", round(top_score, 2))

    # -------- RANKING -------- #
    st.subheader("🏆 Resume Ranking")

    results = sorted(results, key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(results, 1):
        clean_name = name[:25] + "..." if len(name) > 25 else name
        st.write(f"{i}. {clean_name} — {round(score,2)}%")

    # -------- DATABASE -------- #
    st.subheader("📁 Saved Results")

    for row in c.execute("SELECT * FROM results"):
        clean_name = row[0][:25] + "..." if len(row[0]) > 25 else row[0]
        st.write(f"{clean_name} — {round(row[1],2)}%")