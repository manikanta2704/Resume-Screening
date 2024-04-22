import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import re
import pickle

# Function to clean text
@st.cache
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
        "Data Science": ["Data Analyst", "Data Scientist", "Power BI Analyst", "Data Associate"],
        "Java Developer": ["Senior Java Developer", "Java Software Engineer", "Backend Developer"],
        "Testing": ["QA Engineer", "Software Tester", "Automation Test Engineer"],
        "DevOps Engineer": ["DevOps Specialist", "Cloud Engineer", "Site Reliability Engineer"],
        "Python Developer": ["Python Software Developer", "Python Engineer", "Full Stack Developer"],
        "Web Designing": ["UI/UX Designer", "Frontend Developer", "Web Developer"],
        "HR": ["HR Manager", "Talent Acquisition Specialist", "HR Business Partner"],
        "Hadoop": ["Big Data Engineer", "Hadoop Developer", "Data Engineer"],
        "Blockchain": ["Blockchain Developer", "Smart Contract Developer", "Crypto Engineer"],
        "ETL Developer": ["ETL Specialist", "Data Integration Engineer", "Database Developer"],
        "Operations Manager": ["Operations Director", "Operations Analyst", "Business Operations Manager"],
        "Sales": ["Sales Executive", "Business Development Manager", "Sales Operations Specialist"],
        "Mechanical Engineer": ["Mechanical Design Engineer", "Manufacturing Engineer", "Quality Engineer"],
        "Arts": ["Art Director", "Graphic Designer", "Creative Consultant"],
        "Database": ["Database Administrator", "Database Analyst", "Database Architect"],
        "Electrical Engineering": ["Electrical Design Engineer", "Power Systems Engineer", "Embedded Systems Engineer"],
        "Health and fitness": ["Fitness Trainer", "Health Coach", "Nutrition Specialist"],
        "PMO": ["Project Manager", "Program Management Officer", "Project Coordinator"],
        "Business Analyst": ["Business Systems Analyst", "Financial Analyst", "Market Research Analyst"],
        "DotNet Developer": [".NET Software Developer", "C# Developer", "ASP.NET Developer"],
        "Automation Testing": ["Automation Engineer", "Quality Assurance Analyst", "Test Automation Specialist"],
        "Network Security Engineer": ["Cybersecurity Engineer", "Network Analyst", "Information Security Specialist"],
        "SAP Developer": ["SAP Consultant", "SAP ABAP Developer", "SAP Basis Administrator"],
        "Civil Engineer": ["Civil Design Engineer", "Structural Engineer", "Construction Manager"],
        "Advocate": ["Legal Counsel", "Legal Advisor", "Attorney"]
        # Add more categories and corresponding job recommendations as needed
    }
    
    return recommendations.get(category, [])

    
    return recommendations.get(category, [])

def main():
    
    st.title("Automated Resume Screening App")

    # Use sidebar to navigate between pages (PDF and Text)
    selected_page = st.sidebar.radio("Navigate", ["PDF", "Text"])

    if selected_page == "PDF":
        st.subheader("Upload your Resume in .pdf format")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extracted_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()

            st.write("### Extracted Text from pdf:")
            st.write(extracted_text)

            if extracted_text:
                cleaned_text = clean_text(extracted_text)
                st.write("### Cleaned Text:")
                st.write(cleaned_text)

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
                    st.markdown(f"<p style='font-size:25px; font-weight:bold'>Predicted Category: <span style='color:orange'>{predicted_category}</span></p>", unsafe_allow_html=True)

                    recommendations = get_job_recommendations(predicted_category)
                    if recommendations:
                        st.write("Job role recommendations:")
                        for job_role in recommendations:
                            st.write(f"- {job_role}")

    elif selected_page == "Text":
        st.subheader("Enter Text Resume")
        text_resume = st.text_area("Paste your Resume text here", height=300)

        # Automatically clean the text as the user types
        cleaned_text = clean_text(text_resume)
        st.write("### Cleaned Text:")
        st.write(cleaned_text)

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
            st.markdown(f"<p style='font-size:25px; font-weight:bold'>Predicted Category: <span style='color:lightgreen'>{predicted_category}</span></p>", unsafe_allow_html=True)

            recommendations = get_job_recommendations(predicted_category)
            if recommendations:
                st.write("Job role recommendations:")
                for job_role in recommendations:
                    st.write(f"- {job_role}")

# Run the main function to start the app
if __name__ == "__main__":
    main()
