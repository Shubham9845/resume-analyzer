import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re 
def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.lower()

st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

job_desc = st.text_area("Paste Job Description")

stopwords = ["we", "are", "the", "is", "and", "a", "an", "to", "for", "of", "in", "on", "with", "looking"]

if uploaded_file and job_desc:
    resume_text = clean_text(extract_text(uploaded_file))
    job_desc = clean_text(job_desc)

    documents = [resume_text, job_desc]

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

    score = round(similarity * 100, 2)

    st.subheader("Match Score")
    st.write(f"{score}%")

    resume_words = set(resume_text.split())
    job_words = set(job_desc.lower().split())

    missing = [word for word in job_words if word not in resume_words and word not in stopwords]

    st.subheader("Missing Keywords")
    st.write(list(missing)[:10]) 



    
