import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import nltk

nltk.download('stopwords')

# Function to clean text
@st.cache_data()
def clean_text(text):
    """
    Clean the input text by removing URLs, emails, special characters, and stop words.

    :param text: The string to be cleaned
    :return: The cleaned string
    """
    # Compile patterns for URLs and emails to speed up cleaning process
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    # Remove URLs
    cleaned_text = url_pattern.sub('', text)

    # Remove emails
    cleaned_text = email_pattern.sub('', cleaned_text)

    # Remove special characters (keeping only words and whitespace)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    # Remove stop words by filtering the split words of the text
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word.lower() not in stop_words)

    return cleaned_text

def calculate_resume_score(tfidf_vectorizer, job_description, resume_text):
    # Transform job description and resume text using TF-IDF vectorizer
    job_desc_features = tfidf_vectorizer.transform([job_description])
    resume_features = tfidf_vectorizer.transform([resume_text])

    # Calculate cosine similarity between job description and resume
    similarity_score = cosine_similarity(job_desc_features, resume_features)[0][0]

    return similarity_score

def load_models():
    # Load the trained TF-IDF vectorizer
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidf_loaded = pickle.load(tfidf_file)

    # Load the trained logistic regression classifier
    with open('clf.pkl', 'rb') as clf_file:
        clf_loaded = pickle.load(clf_file)

    return tfidf_loaded, clf_loaded

def get_job_recommendations(category):
    recommendations = {
        "Java Developer": ["Senior Java Developer", "Java Software Engineer", "Backend Developer", "Java Full Stack Developer", "Java Architect"],
        "Testing": ["QA Engineer", "Software Tester", "Automation Test Engineer", "Performance Test Engineer", "Quality Assurance Manager"],
        "DevOps Engineer": ["DevOps Specialist", "Cloud DevOps Engineer", "Site Reliability Engineer (SRE)", "DevOps Automation Engineer", "Infrastructure Engineer"],
        "Python Developer": ["Python Software Engineer", "Django Developer", "Python Full Stack Developer", "Data Engineer (Python)", "Python Automation Engineer"],
        "Web Designing": ["Frontend Web Developer", "UI/UX Designer", "Web Graphic Designer", "Web Development Team Lead", "Web Design Manager"],
        "HR": ["Human Resources Manager", "Talent Acquisition Specialist", "HR Business Partner", "HR Generalist", "Compensation and Benefits Analyst"],
        "Hadoop": ["Big Data Engineer (Hadoop)", "Hadoop Administrator", "Hadoop Developer", "Hadoop Architect", "Hadoop Data Analyst"],
        "Blockchain": ["Blockchain Developer", "Blockchain Architect", "Blockchain Engineer", "Blockchain Consultant", "Smart Contract Developer"],
        "ETL Developer": ["ETL Specialist", "Data Integration Developer", "ETL Architect", "Data Warehouse Developer", "Informatica Developer"],
        "Operations Manager": ["Operations Director", "Business Operations Manager", "Operations Analyst", "Operations Team Lead", "Supply Chain Manager"],
        "Data Science": ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "AI Research Scientist", "Data Engineer (Data Science)"],
        "Sales": ["Sales Manager", "Account Executive", "Business Development Representative", "Sales Operations Analyst", "Inside Sales Representative"],
        "Mechanical Engineer": ["Mechanical Design Engineer", "Automotive Engineer", "HVAC Engineer", "Manufacturing Engineer", "Robotics Engineer"],
        "Arts": ["Art Director", "Graphic Designer", "Creative Director", "Multimedia Artist", "Visual Development Artist"],
        "Database": ["Database Administrator (DBA)", "SQL Developer", "Database Architect", "Data Warehouse Analyst", "Database Engineer"],
        "Electrical Engineering": ["Electrical Design Engineer", "Power Systems Engineer", "Electronics Engineer", "Control Systems Engineer", "Embedded Systems Engineer"],
        "Health and fitness": ["Fitness Instructor", "Personal Trainer", "Nutritionist", "Physical Therapist", "Wellness Coach"],
        "PMO": ["Project Manager", "Program Management Office Lead", "PMO Analyst", "Portfolio Manager", "PMO Director"],
        "Business Analyst": ["Business Systems Analyst", "Financial Analyst", "Product Analyst", "Business Process Analyst", "Data Analyst (Business)"],
        "DotNet Developer": [".NET Developer", "C# Developer", "ASP.NET Developer", "Full Stack .NET Developer", "Backend .NET Developer"],
        "Automation Testing": ["Test Automation Engineer", "Quality Assurance Automation Engineer", "Selenium Test Engineer", "API Test Automation Engineer", "Test Automation Architect"],
        "Network Security Engineer": ["Cyber Security Engineer", "Information Security Analyst", "Network Security Administrator", "Security Operations Center (SOC) Analyst", "Penetration Tester"],
        "SAP Developer": ["SAP ABAP Developer", "SAP Fiori Developer", "SAP Basis Administrator", "SAP Functional Consultant", "SAP HANA Developer"],
        "Civil Engineer": ["Structural Engineer", "Construction Project Manager", "Geotechnical Engineer", "Transportation Engineer", "Environmental Engineer"],
        "Advocate": ["Lawyer", "Legal Counsel", "Attorney", "Legal Advisor", "Corporate Counsel"]
    }

    return recommendations.get(category, [])

def main():
    st.title("Automated Resume Screening App")

    # Use sidebar to navigate between pages (PDF and Text)
    selected_page = st.sidebar.radio("Navigate", ["PDF", "Text", "Resume Score", "Compare 2 Resumes"])

    if selected_page == "PDF":
        st.subheader("Upload your Resume in .pdf format")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extracted_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()

            st.write("### Extracted Text from PDF:")
            st.write(f"- <span style='color:#FFFFCC'>{extracted_text}</span>", unsafe_allow_html=True)

            if extracted_text:
                cleaned_text = clean_text(extracted_text)
                st.write("### Cleaned Text:")
                st.write(f"- <span style='color:#FDD7E4'>{cleaned_text}</span>", unsafe_allow_html=True)

                if st.button("Predict job role from PDF"):
                    tfidf_loaded, clf_loaded = load_models()
                    input_features = tfidf_loaded.transform([cleaned_text])
                    prediction_id = clf_loaded.predict(input_features)[0]

                    category_mapping = {
                        15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
                        24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
                        18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
                        1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
                        19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
                        17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
                    }

                    predicted_category = category_mapping.get(prediction_id, "Unknown")
                    st.markdown(f"<p style='font-size:25px; font-weight:bold'>The given resume is best suited for the role of: <span style='color:orange'>{predicted_category}</span></p>", unsafe_allow_html=True)

                    recommendations = get_job_recommendations(predicted_category)
                    if recommendations:
                        st.write("Job role recommendations:")
                        for job_role in recommendations:
                            st.write(f"- <span style='color:cyan'>{job_role}</span>", unsafe_allow_html=True)

    elif selected_page == "Text":
        st.subheader("Enter Text Resume")
        text_resume = st.text_area("Paste your Resume text here", height=300)

        # Automatically clean the text as the user types
        cleaned_text = clean_text(text_resume)
        st.write("### Cleaned Text:")
        st.write(f"- <span style='color:#FDD7E4'>{cleaned_text}</span>", unsafe_allow_html=True)
        

        if st.button("Predict Category from Text"):
            tfidf_loaded, clf_loaded = load_models()
            input_features = tfidf_loaded.transform([cleaned_text])
            prediction_id = clf_loaded.predict(input_features)[0]

            category_mapping = {
                15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
                24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
                18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
                1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
                19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
                17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
            }

            predicted_category = category_mapping.get(prediction_id, "Unknown")
            st.markdown(f"<p style='font-size:25px; font-weight:bold'>The given resume is best suited for the role of: <span style='color:lightgreen'>{predicted_category}</span></p>", unsafe_allow_html=True)

            recommendations = get_job_recommendations(predicted_category)
            if recommendations:
                st.write("Job role recommendations:")
                for job_role in recommendations:
                    st.write(f"- <span style='color:#7FFFD4'>{job_role}</span>", unsafe_allow_html=True)

    elif selected_page == "Resume Score":
        st.subheader("Calculate Resume Similarity Score")
        job_description = st.text_area("Enter Job Description", height=200)
        resume_text = st.text_area("Paste Resume Text to Compare", height=300)

        if st.button("Calculate Resume Score"):
            tfidf_loaded, _ = load_models()
            similarity_score = calculate_resume_score(tfidf_loaded, job_description, resume_text)
            st.write(f"### The chances of getting the above job is :<span style='color:Cyan'>{similarity_score * 100:.0f}%</span>", unsafe_allow_html=True)

    elif selected_page == "Compare 2 Resumes":
        st.subheader("Compare Two Resumes")
        uploaded_file1 = st.file_uploader("Upload Resume 1 (PDF)", type=["pdf"])
        uploaded_file2 = st.file_uploader("Upload Resume 2 (PDF)", type=["pdf"])

        if uploaded_file1 is not None and uploaded_file2 is not None:
            resume_texts = []
            for uploaded_file in [uploaded_file1, uploaded_file2]:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                extracted_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_text += page.extract_text()
                cleaned_text = clean_text(extracted_text)
                resume_texts.append(cleaned_text)

            if st.button("Compare Resumes"):
                tfidf_loaded, _ = load_models()
                similarity_score = calculate_resume_score(tfidf_loaded, resume_texts[0], resume_texts[1])
                st.write(f"### The Similarity between the 2 Resumes is:<span style='color:Magenta'>{similarity_score * 100:.0f}%</span>", unsafe_allow_html=True)


# Run the main function to start the app
if __name__ == "__main__":
    main()
