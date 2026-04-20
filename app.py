import streamlit as st
st.set_page_config(page_title="AI Resume Analyzer Pro", page_icon="🚀")

from PyPDF2 import PdfReader
import spacy
import sqlite3
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

def clean_filename(name):
    return name[:25] + "..." if len(name) > 25 else name

# ---------------- UI ---------------- #
st.sidebar.write(f"👤 Logged in as: {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.title("🚀 AI Resume Analyzer Pro")
st.write("Smart ATS Scoring with NLP + ML")

uploaded_files = st.file_uploader(
    "Upload Resumes", type=["pdf"], accept_multiple_files=True
)

job_desc = st.text_area("Paste Job Description")

role = st.selectbox(
    "Select Job Role",
    ["Data Scientist", "Data Analyst", "ML Engineer", "Web Developer"]
)

# ---------------- SKILLS ---------------- #
skills_dict = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "sql", "statistics"],
    "Data Analyst": ["sql", "excel", "tableau", "power bi", "statistics"],
    "ML Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch"],
    "Web Developer": ["html", "css", "javascript", "react", "nodejs"]
}

# ---------------- MAIN LOGIC ---------------- #
if uploaded_files and job_desc:

    c.execute("DELETE FROM results")
    conn.commit()

    required_skills = skills_dict[role]
    job_clean = clean_nlp_text(job_desc)

    results = []

    for i, file in enumerate(uploaded_files):

        # -------- TEXT PROCESSING -------- #
        resume_raw = extract_text(file)
        resume_clean = clean_nlp_text(resume_raw)

        # -------- SKILL MATCHING -------- #
        matched_skills = [s for s in required_skills if s in resume_clean]
        missing_skills = [s for s in required_skills if s not in resume_clean]

        skills_score = (len(matched_skills) / len(required_skills)) * 100

        # -------- ML SIMILARITY -------- #
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, job_clean])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        similarity_score = similarity * 100

        # -------- FINAL SCORE -------- #
        experience_score = 5
        education_score = 10

        final_score = (
            skills_score * 0.4 +
            similarity_score * 0.4 +
            experience_score * 0.1 +
            education_score * 0.1
        )

        results.append((file.name, final_score))

        c.execute("INSERT INTO results VALUES (?, ?)", (file.name, final_score))
        conn.commit()

        # -------- DISPLAY -------- #
        st.divider()
        st.subheader(f"📄 {clean_filename(file.name)}")

        st.metric("🧠 Skill Match", f"{round(skills_score,2)}%")
        st.metric("📊 Resume-JD Similarity", f"{round(similarity_score,2)}%")

        st.subheader("🏆 Final ATS Score")
        st.metric("Final Score", f"{round(final_score,2)}%")
        st.progress(int(final_score))

        # -------- GRAPH -------- #
        st.subheader("📊 Skill Match Visualization")
        labels = ["Matched", "Missing"]
        values = [len(matched_skills), len(missing_skills)]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        st.pyplot(fig)

        # -------- SKILLS -------- #
        st.subheader("✅ Matched Skills")
        for s in matched_skills:
            st.write(f"✔️ {s}")

        st.subheader("❌ Missing Skills")
        for s in missing_skills:
            st.write(f"❌ {s}")

        # -------- FEEDBACK -------- #
        if final_score > 75:
            st.success("🔥 Strong match")
        elif final_score > 50:
            st.warning("⚠️ Moderate match")
        else:
            st.error("❌ Low match")

        # -------- RECOMMENDATIONS -------- #
        st.subheader("💡 Recommendations")
        for s in missing_skills:
            st.write(f"👉 Learn {s}")

    # ---------------- RANKING ---------------- #
    st.divider()
    st.subheader("🏆 Resume Ranking")

    results = sorted(results, key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(results, 1):
        st.write(f"{i}. {clean_filename(name)} — {round(score,2)}%")

    # ---------------- DATABASE ---------------- #
    st.divider()
    st.subheader("📁 Saved Results")

    for row in c.execute("SELECT * FROM results"):
        st.write(f"{clean_filename(row[0])} — {round(row[1],2)}%")