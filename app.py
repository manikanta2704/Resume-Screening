import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

    return tfidf_loaded

def calculate_resume_score(tfidf_vectorizer, job_description, resume_text):
    # Transform job description and resume text using TF-IDF vectorizer
    job_desc_features = tfidf_vectorizer.transform([job_description])
    resume_features = tfidf_vectorizer.transform([resume_text])

    # Calculate cosine similarity between job description and resume
    similarity_score = cosine_similarity(job_desc_features, resume_features)[0][0]

    return similarity_score

def main():
    
    st.title("Automated Resume Screening App")

    # Use sidebar to navigate between pages (PDF, Text, Resume Score, Compare 2 Resumes)
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

            st.write("### Extracted Text from pdf:")
            st.write(extracted_text)

            if extracted_text:
                cleaned_text = clean_text(extracted_text)
                st.write("### Cleaned Text:")
                st.write(cleaned_text)

                if st.button("Predict job role from PDF"):
                    tfidf_loaded = load_models()
                    input_features = tfidf_loaded.transform([cleaned_text])
                    prediction_id = 0  # Replace with your model prediction logic

                    category_mapping = {
                        15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
                        24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
                        18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
                        1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
                        19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
                        17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
                    }

                    predicted_category = category_mapping.get(prediction_id, "Unknown")
                    st.markdown(f"<p style='font-size:25px; font-weight:bold'>The Given Resume is best suited for the role of: <span style='color:orange'>{predicted_category}</span></p>", unsafe_allow_html=True)

    elif selected_page == "Text":
        st.subheader("Enter Text Resume")
        text_resume = st.text_area("Paste your Resume text here", height=300)

        # Automatically clean the text as the user types
        cleaned_text = clean_text(text_resume)
        st.write("### Cleaned Text:")
        st.write(cleaned_text)

        if st.button("Predict Category from Text"):
            tfidf_loaded = load_models()
            input_features = tfidf_loaded.transform([cleaned_text])
            prediction_id = 0  # Replace with your model prediction logic

            category_mapping = {
                15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
                24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
                18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
                1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
                19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
                17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
            }

            predicted_category = category_mapping.get(prediction_id, "Unknown")
            st.markdown(f"<p style='font-size:25px; font-weight:bold'>The Given Resume is best suited for the role of: <span style='color:lightgreen'>{predicted_category}</span></p>", unsafe_allow_html=True)

    elif selected_page == "Resume Score":
        st.subheader("Calculate Resume Similarity Score")
        job_description = st.text_area("Enter Job Description", height=200)
        resume_text = st.text_area("Enter Resume Text", height=400)

        if st.button("Calculate Resume Score"):
            tfidf_loaded = load_models()
            resume_score = calculate_resume_score(tfidf_loaded, job_description, resume_text)
            st.write(f"### Resume Similarity Score: {resume_score:.2f}")

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
                tfidf_loaded = load_models()
                similarity_score = calculate_resume_score(tfidf_loaded, resume_texts[0], resume_texts[1])
                st.write(f"### Similarity Score between Resumes: {similarity_score:.2f}")

# Run the main function to start the app
if __name__ == "__main__":
    main()
