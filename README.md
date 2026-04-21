# 🚀 AI Resume Analyzer Pro

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://resume-analyzer-pro-2026.streamlit.app/)

> Smart ATS Scoring & Resume Insights for Campus Placements

## 🔗 Live Demo
👉 **[https://resume-analyzer-pro-2026.streamlit.app/](https://resume-analyzer-pro-2026.streamlit.app/)**

---

## 📌 About
AI Resume Analyzer Pro is a smart web application that analyzes resumes against job descriptions and provides ATS (Applicant Tracking System) scores using NLP and Machine Learning techniques.

---

## ✨ Features
- 📁 Upload multiple resumes (PDF) at once
- 🎯 Select from 18+ job roles
- 📊 ATS scoring based on 4 parameters
- 🧠 Skill match analysis
- 📈 JD similarity scoring using TF-IDF
- 💡 Personalized resume improvement tips
- 🏆 Candidate ranking & comparison charts
- 📥 Export results to Excel

---

## 📊 Scoring Breakdown
| Parameter | Weight |
|-----------|--------|
| 🧠 Skill Match | 40% |
| 📊 JD Similarity | 30% |
| 💼 Experience | 20% |
| 🎓 Education | 10% |

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Backend:** Python
- **ML/NLP:** Scikit-learn, TF-IDF, Cosine Similarity
- **Data:** Pandas, SQLite
- **Visualization:** Matplotlib
- **PDF Parsing:** PyPDF2

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Shubham9845/resume-analyzer.git
cd resume-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure
```
resume-analyzer/
│
├── app.py               # Main application
├── requirements.txt     # Dependencies
├── resume_data.db       # SQLite database
└── README.md            # Project documentation
```

---

## 👨‍💻 Developer
**Shubham Kumar**
- GitHub: [@Shubham9845](https://github.com/Shubham9845)
- Live App: [resume-analyzer-pro-2026.streamlit.app](https://resume-analyzer-pro-2026.streamlit.app/)

---

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
