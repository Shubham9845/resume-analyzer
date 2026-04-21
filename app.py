import streamlit as st
from PyPDF2 import PdfReader
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer Pro", page_icon="🚀", layout="wide")

# ---------------- NLP (no spacy needed) ---------------- #
def clean_nlp_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    stop_words = {"the","and","for","are","was","were","with","this","that","from",
                  "have","has","had","not","but","what","all","been","they","will",
                  "one","can","our","you","your","his","her","its","their","there",
                  "when","which","who","how","any","more","also","into","than","then"}
    words = [w for w in text.split() if len(w) > 2 and w not in stop_words]
    return " ".join(words)

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("resume_data.db", check_same_thread=False)
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS results")
c.execute("""CREATE TABLE results (
    name TEXT,
    final_score REAL,
    skill_score REAL,
    similarity_score REAL,
    experience_score REAL,
    education_score REAL
)""")
conn.commit()

# ---------------- FUNCTIONS ---------------- #
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + " "
    return text

def clean_filename(name):
    return name[:30] + "..." if len(name) > 30 else name

def detect_experience(text):
    text_lower = text.lower()
    score = 0
    patterns = [
        r'\b(\d+)\s*\+?\s*year[s]?\s*(of)?\s*experience',
        r'\bexperience\s*of\s*(\d+)\s*year',
        r'\b(\d+)\s*month[s]?\s*(of)?\s*(internship|experience)',
    ]
    for pat in patterns:
        match = re.search(pat, text_lower)
        if match:
            val = int(match.group(1))
            score = min(val * 5, 20)
            return score
    if any(w in text_lower for w in ["internship", "intern", "trainee", "apprentice"]):
        score = 10
    elif any(w in text_lower for w in ["project", "developed", "built", "created", "designed"]):
        score = 7
    return score

def detect_education(text):
    text_lower = text.lower()
    if any(w in text_lower for w in ["ph.d", "phd", "doctorate"]):
        return 20
    elif any(w in text_lower for w in ["m.tech", "m.e.", "mtech", "masters", "m.sc", "mba", "pgdm"]):
        return 18
    elif any(w in text_lower for w in ["b.tech", "b.e.", "btech", "b.sc", "bca", "bachelor", "b.com", "bba"]):
        return 15
    elif any(w in text_lower for w in ["diploma", "polytechnic"]):
        return 10
    return 8

def smart_skill_match(resume_text, skills):
    resume_lower = resume_text.lower()
    matched = []
    missing = []
    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower in resume_lower:
            matched.append(skill)
            continue
        skill_compact = skill_lower.replace(" ", "").replace("-", "")
        resume_compact = resume_lower.replace(" ", "").replace("-", "")
        if skill_compact in resume_compact:
            matched.append(skill)
            continue
        words = skill_lower.split()
        if len(words) > 1 and all(w in resume_lower for w in words):
            matched.append(skill)
            continue
        missing.append(skill)
    return matched, missing

def keyword_density(text, keywords):
    density = {}
    text_lower = text.lower()
    for kw in keywords:
        count = text_lower.count(kw.lower())
        density[kw] = count
    return density

def get_resume_tips(missing_skills, role, final_score):
    tips = []
    if final_score < 40:
        tips.append("🔴 **Overall:** Your resume needs significant improvement for this role.")
    elif final_score < 65:
        tips.append("🟡 **Overall:** You have a moderate match. Adding more relevant skills will help.")
    else:
        tips.append("🟢 **Overall:** Strong match! A few tweaks can make it even better.")
    for skill in missing_skills[:8]:
        tips.append(f"➕ Add **{skill}** to your resume — it's required for {role}")
    if not missing_skills:
        tips.append("✅ Your resume covers all required skills for this role!")
    tips.append("📌 **Tip:** Use exact keywords from the job description in your resume.")
    tips.append("📌 **Tip:** Add measurable achievements e.g. 'Improved performance by 30%'")
    tips.append("📌 **Tip:** Include project links (GitHub / Live Demo) wherever possible.")
    return tips

# ---------------- SKILLS DICT ---------------- #
skills_dict = {
    "Data Analyst": [
        "sql", "excel", "tableau", "power bi", "statistics", "python",
        "pandas", "numpy", "matplotlib", "seaborn", "data visualization",
        "data cleaning", "google sheets", "looker", "reporting",
        "mysql", "postgresql", "pivot table", "vlookup", "dashboard"
    ],
    "Data Scientist": [
        "python", "machine learning", "pandas", "numpy", "sql", "statistics",
        "scikit-learn", "tensorflow", "keras", "deep learning", "nlp",
        "data visualization", "matplotlib", "seaborn", "jupyter",
        "feature engineering", "model evaluation", "hypothesis testing", "r"
    ],
    "ML Engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit-learn", "mlops", "docker", "kubernetes", "aws", "model deployment",
        "feature engineering", "neural networks", "transformers", "fastapi",
        "airflow", "mlflow", "ci/cd", "rest api"
    ],
    "Business Analyst": [
        "sql", "excel", "power bi", "tableau", "requirements gathering",
        "stakeholder management", "jira", "agile", "scrum", "data analysis",
        "process improvement", "documentation", "reporting", "brd", "uml",
        "gap analysis", "use cases", "wireframing", "ms office"
    ],
    "Web Developer": [
        "html", "css", "javascript", "react", "nodejs", "sql",
        "git", "rest api", "responsive design", "bootstrap",
        "typescript", "mongodb", "express", "github", "deployment"
    ],
    "Frontend Developer": [
        "html", "css", "javascript", "react", "vuejs", "typescript",
        "tailwind", "bootstrap", "git", "figma", "responsive design",
        "redux", "webpack", "sass", "next.js", "jest", "accessibility"
    ],
    "Backend Developer": [
        "python", "nodejs", "java", "sql", "mongodb", "rest api",
        "django", "flask", "express", "postgresql", "docker", "git",
        "microservices", "authentication", "redis", "kafka", "aws"
    ],
    "Full Stack Developer": [
        "html", "css", "javascript", "react", "nodejs", "python",
        "sql", "mongodb", "git", "rest api", "docker", "typescript",
        "django", "flask", "aws", "ci/cd", "github"
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "azure", "ci/cd", "jenkins",
        "linux", "terraform", "ansible", "git", "python", "bash",
        "monitoring", "prometheus", "grafana", "helm", "gitlab"
    ],
    "Cloud Engineer": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "linux", "python", "networking", "security", "s3", "ec2",
        "lambda", "cloudformation", "iam", "vpc", "load balancer"
    ],
    "Cybersecurity Analyst": [
        "network security", "firewalls", "penetration testing", "siem",
        "vulnerability assessment", "linux", "python", "encryption",
        "incident response", "compliance", "ethical hacking", "wireshark",
        "nmap", "metasploit", "iso 27001", "owasp"
    ],
    "Software Engineer": [
        "python", "java", "c++", "data structures", "algorithms",
        "sql", "git", "rest api", "docker", "agile", "oop",
        "system design", "testing", "debugging", "design patterns", "github"
    ],
    "Android Developer": [
        "java", "kotlin", "android sdk", "xml", "firebase", "git",
        "rest api", "sqlite", "mvvm", "jetpack compose", "retrofit",
        "room database", "coroutines", "material design"
    ],
    "iOS Developer": [
        "swift", "objective-c", "xcode", "ios sdk", "git", "firebase",
        "rest api", "coredata", "mvvm", "swiftui", "cocoapods",
        "combine", "uikit", "app store"
    ],
    "UI/UX Designer": [
        "figma", "adobe xd", "sketch", "wireframing", "prototyping",
        "user research", "usability testing", "html", "css",
        "design systems", "typography", "color theory", "user journey",
        "information architecture", "accessibility"
    ],
    "Product Manager": [
        "product roadmap", "agile", "scrum", "jira", "stakeholder management",
        "user stories", "data analysis", "sql", "market research",
        "a/b testing", "product strategy", "kpi", "okr",
        "competitive analysis", "prioritization", "go to market"
    ],
    "Database Administrator": [
        "sql", "mysql", "postgresql", "oracle", "mongodb", "database design",
        "backup recovery", "performance tuning", "indexing", "stored procedures",
        "replication", "security", "linux", "redis", "nosql"
    ],
    "AI Engineer": [
        "python", "machine learning", "deep learning", "nlp", "llm",
        "tensorflow", "pytorch", "transformers", "langchain", "openai",
        "vector databases", "mlops", "docker", "aws", "fine tuning",
        "rag", "prompt engineering", "fastapi"
    ],
}

# ===================== UI ===================== #

st.title("🚀 AI Resume Analyzer Pro")
st.markdown("##### Smart ATS Scoring & Resume Insights for Campus Placements")
st.markdown("---")

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
1. Upload one or more resumes (PDF)
2. Paste the job description
3. Select the job role
4. Click **Analyze Resumes**
5. View scores, charts, tips
6. Download results as Excel
    """)
    st.markdown("---")
    st.markdown("**Scoring Breakdown:**")
    st.markdown("""
- 🧠 Skill Match: **40%**
- 📊 JD Similarity: **30%**
- 💼 Experience: **20%**
- 🎓 Education: **10%**
    """)
    st.markdown("---")
    st.markdown("**Built with:** Python · Streamlit · NLP · ML")
    st.markdown("**Developer:** Shubham Kumar")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repo-black)](https://github.com/Shubham9845/resume-analyzer)")

# ---------------- INPUTS ---------------- #
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_files = st.file_uploader(
        "📁 Upload Resumes (PDF — Multiple Allowed)",
        type=["pdf"],
        accept_multiple_files=True
    )

with col2:
    role = st.selectbox(
        "🎯 Select Job Role",
        list(skills_dict.keys())
    )
    st.info(f"📌 {len(skills_dict[role])} skills tracked for **{role}**")

job_desc = st.text_area(
    "📝 Paste Job Description Here",
    height=200,
    placeholder="Paste the full job description here — the more detail, the better the analysis..."
)

analyze_btn = st.button("🔍 Analyze Resumes", type="primary", use_container_width=True)

# ===================== MAIN LOGIC ===================== #

if analyze_btn:
    if not uploaded_files:
        st.error("⚠️ Please upload at least one resume PDF!")
        st.stop()
    if not job_desc.strip():
        st.error("⚠️ Please paste a job description!")
        st.stop()

    c.execute("DELETE FROM results")
    conn.commit()

    required_skills = skills_dict[role]
    job_clean = clean_nlp_text(job_desc)

    all_results = []
    export_data = []

    st.markdown("---")
    st.header("📊 Analysis Results")

    for file in uploaded_files:
        resume_raw = extract_text(file)

        if not resume_raw.strip():
            st.warning(f"⚠️ Could not extract text from {file.name} — skipping.")
            continue

        resume_clean = clean_nlp_text(resume_raw)

        # -------- SKILL MATCHING -------- #
        matched_skills, missing_skills = smart_skill_match(resume_raw, required_skills)
        skills_score = (len(matched_skills) / len(required_skills)) * 100

        # -------- JD SIMILARITY -------- #
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            vectors = vectorizer.fit_transform([resume_clean, job_clean])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            similarity_score = round(similarity * 100, 2)
        except:
            similarity_score = 0.0

        # -------- EXPERIENCE & EDUCATION -------- #
        experience_score = detect_experience(resume_raw)
        education_score = detect_education(resume_raw)

        # -------- FINAL SCORE -------- #
        final_score = (
            skills_score * 0.40 +
            similarity_score * 0.30 +
            experience_score * 0.20 +
            education_score * 0.10
        )
        final_score = round(min(final_score, 100), 2)

        all_results.append((file.name, final_score, skills_score, similarity_score, experience_score, education_score))
        export_data.append({
            "Resume": file.name,
            "Final ATS Score (%)": round(final_score, 2),
            "Skill Match (%)": round(skills_score, 2),
            "JD Similarity (%)": round(similarity_score, 2),
            "Experience Score": round(experience_score, 2),
            "Education Score": round(education_score, 2),
            "Matched Skills": ", ".join(matched_skills),
            "Missing Skills": ", ".join(missing_skills)
        })

        c.execute("INSERT INTO results VALUES (?, ?, ?, ?, ?, ?)",
                  (file.name, final_score, skills_score, similarity_score, experience_score, education_score))
        conn.commit()

        # -------- DISPLAY CARD -------- #
        with st.expander(f"📄 {clean_filename(file.name)}  |  ATS Score: {round(final_score, 1)}%", expanded=True):

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("🏆 Final Score", f"{round(final_score, 1)}%")
            m2.metric("🧠 Skill Match", f"{round(skills_score, 1)}%")
            m3.metric("📊 JD Similarity", f"{round(similarity_score, 1)}%")
            m4.metric("💼 Experience", f"{round(experience_score, 1)}/20")
            m5.metric("🎓 Education", f"{round(education_score, 1)}/20")

            st.progress(min(int(final_score), 100))

            if final_score >= 75:
                st.success("🔥 Strong Match — Excellent candidate for this role!")
            elif final_score >= 50:
                st.warning("⚠️ Moderate Match — Needs some improvements")
            elif final_score >= 30:
                st.warning("🟡 Below Average — Several key skills missing")
            else:
                st.error("❌ Low Match — Resume needs significant improvement for this role")

            # -------- CHARTS -------- #
            ch1, ch2 = st.columns(2)

            with ch1:
                st.subheader("📊 Skill Match")
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                bars = ax1.bar(
                    ["Matched", "Missing"],
                    [len(matched_skills), len(missing_skills)],
                    color=["#2ecc71", "#e74c3c"],
                    edgecolor="white",
                    width=0.5
                )
                for bar in bars:
                    h = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(int(h)),
                             ha="center", va="bottom", fontsize=11, fontweight="bold")
                ax1.set_ylabel("Number of Skills")
                ax1.set_ylim(0, len(required_skills) + 2)
                ax1.spines["top"].set_visible(False)
                ax1.spines["right"].set_visible(False)
                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            with ch2:
                st.subheader("🥧 Score Breakdown")
                labels = ["Skill Match", "JD Similarity", "Experience", "Education"]
                values = [skills_score * 0.40, similarity_score * 0.30,
                          experience_score * 0.20, education_score * 0.10]
                colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                wedges, texts, autotexts = ax2.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    colors=colors,
                    startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
                )
                for at in autotexts:
                    at.set_fontsize(9)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            # -------- KEYWORD DENSITY -------- #
            st.subheader("🔑 Keyword Density (Top 10)")
            density = keyword_density(resume_raw, required_skills)
            density_df = pd.DataFrame(
                list(density.items()), columns=["Keyword", "Count"]
            ).sort_values("Count", ascending=False).head(10)
            density_df = density_df[density_df["Count"] > 0]
            if not density_df.empty:
                st.dataframe(density_df, use_container_width=True, hide_index=True)
            else:
                st.info("No matching keywords found in this resume.")

            # -------- SKILLS -------- #
            sk1, sk2 = st.columns(2)
            with sk1:
                st.subheader(f"✅ Matched Skills ({len(matched_skills)})")
                if matched_skills:
                    for s in matched_skills:
                        st.markdown(f"✔️ `{s}`")
                else:
                    st.info("No matching skills found.")

            with sk2:
                st.subheader(f"❌ Missing Skills ({len(missing_skills)})")
                if missing_skills:
                    for s in missing_skills:
                        st.markdown(f"❌ `{s}`")
                else:
                    st.success("All skills matched!")

            # -------- TIPS -------- #
            st.subheader("💡 Personalized Resume Tips")
            tips = get_resume_tips(missing_skills, role, final_score)
            for tip in tips:
                st.markdown(tip)

    # ===================== RANKING ===================== #
    if all_results:
        st.markdown("---")
        st.header("🏆 Candidate Ranking")

        all_results_sorted = sorted(all_results, key=lambda x: x[1], reverse=True)

        rank_df = pd.DataFrame(all_results_sorted,
                               columns=["Resume", "Final Score (%)", "Skill Match (%)",
                                        "JD Similarity (%)", "Experience Score", "Education Score"])
        rank_df.index = range(1, len(rank_df) + 1)
        rank_df["Final Score (%)"] = rank_df["Final Score (%)"].round(2)
        rank_df["Skill Match (%)"] = rank_df["Skill Match (%)"].round(2)
        rank_df["JD Similarity (%)"] = rank_df["JD Similarity (%)"].round(2)

        st.dataframe(rank_df, use_container_width=True)

        # -------- COMPARISON CHART -------- #
        if len(all_results_sorted) > 1:
            st.subheader("📊 Candidate Score Comparison")
            fig3, ax3 = plt.subplots(figsize=(10, max(3, len(all_results_sorted) * 0.8)))
            names = [clean_filename(r[0]) for r in all_results_sorted]
            scores = [r[1] for r in all_results_sorted]
            colors_bar = ["#2ecc71" if s >= 60 else "#f39c12" if s >= 40 else "#e74c3c" for s in scores]
            bars = ax3.barh(names, scores, color=colors_bar, edgecolor="white")
            ax3.set_xlabel("ATS Score (%)")
            ax3.set_xlim(0, 105)
            for bar, score in zip(bars, scores):
                ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                         f"{round(score, 1)}%", va="center", fontsize=10, fontweight="bold")
            ax3.spines["top"].set_visible(False)
            ax3.spines["right"].set_visible(False)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        # ===================== EXPORT ===================== #
        st.markdown("---")
        st.header("📥 Export Results")

        export_df = pd.DataFrame(export_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Resume Analysis")
        output.seek(0)

        st.download_button(
            label="⬇️ Download Full Results as Excel",
            data=output,
            file_name="resume_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        # ===================== SAVED DB ===================== #
        st.markdown("---")
        st.subheader("📁 Saved in Database")
        db_df = pd.read_sql("SELECT * FROM results ORDER BY final_score DESC", conn)
        db_df.index = range(1, len(db_df) + 1)
        st.dataframe(db_df, use_container_width=True)
