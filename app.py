import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
import pickle
from streamlit.components.v1 import html

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

def calculate_resume_score(tfidf_vectorizer, job_description, resume_text):
    # Transform job description and resume text using TF-IDF vectorizer
    job_desc_features = tfidf_vectorizer.transform([job_description])
    resume_features = tfidf_vectorizer.transform([resume_text])

    # Calculate cosine similarity between job description and resume
    similarity_score = cosine_similarity(job_desc_features, resume_features)[0][0]

    return similarity_score

def main():
    st.title("Automated Resume Screening App")

    # Define custom CSS for styling tabs
    custom_css = """
    <style>
    .tab-container {
      overflow: hidden;
      position: relative;
      height: 40px;
      margin-bottom: 20px;
    }

    .tab-container .tab-content {
      display: none;
      padding: 20px;
      position: absolute;
      left: 0;
      width: 100%;
      box-sizing: border-box;
    }

    .tab-container .tab-content.active {
      display: block;
    }

    .tab-links {
      display: flex;
      overflow-x: auto;
    }

    .tab-links button {
      flex: 1;
      background-color: #f2f2f2;
      border: none;
      outline: none;
      cursor: pointer;
      padding: 10px 15px;
      transition: background-color 0.3s;
    }

    .tab-links button.active {
      background-color: #ddd;
    }

    .tab-content p {
      margin-top: 10px;
    }
    </style>
    """

    # Define JavaScript for tab functionality
    custom_js = """
    <script>
    function openTab(evt, tabName) {
      var i, tabContent, tabLinks;
      tabContent = document.getElementsByClassName("tab-content");
      for (i = 0; i < tabContent.length; i++) {
        tabContent[i].style.display = "none";
      }
      tabLinks = document.getElementsByClassName("tab-links");
      for (i = 0; i < tabLinks.length; i++) {
        tabLinks[i].className = tabLinks[i].className.replace(" active", "");
      }
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }

    document.addEventListener("DOMContentLoaded", function() {
      var tabLinks = document.getElementsByClassName("tab-links");
      for (var i = 0; i < tabLinks.length; i++) {
        var buttons = tabLinks[i].getElementsByTagName("button");
        for (var j = 0; j < buttons.length; j++) {
          buttons[j].addEventListener("click", function(event) {
            openTab(event, this.dataset.tabName);
          });
        }
      }
    });
    </script>
    """

    # Render custom CSS and JavaScript
    html(custom_css + custom_js)

    # Define tab content
    tabs = {
        "PDF": pdf_tab,
        "Text": text_tab,
        "Resume Score": resume_score_tab,
        "Compare Resumes": compare_resumes_tab
    }

    # Use tabs to organize different sections
    selected_tab = st.sidebar.selectbox("Select Section", list(tabs.keys()))

    # Display selected tab content
    tabs[selected_tab]()

def pdf_tab():
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
                st.markdown(f"<p style='font-size:25px; font-weight:bold'>The Given Resume is best suited for the role of: <span style='color:orange'>{predicted_category}</span></p>", unsafe_allow_html=True)

def text_tab():
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
        st.markdown(f"<p style='font-size:25px; font-weight:bold'>The Given Resume is best suited for the role of: <span style='color:lightgreen'>{predicted_category}</span></p>", unsafe_allow_html=True)

def resume_score_tab():
    st.subheader("Calculate Resume Similarity Score")
    job_description = st.text_area("Enter Job Description", height=200)
    resume_text = st.text_area("Enter Resume Text", height=400)

    if st.button("Calculate Resume Score"):
        tfidf_loaded, _ = load_models()
        resume_score = calculate_resume_score(tfidf_loaded, job_description, resume_text)
        formatted_score = f"{resume_score * 100:.2f}%"
        st.write(f"### Similarity Score between Resumes: <span style='color:blue'>{formatted_score}</span>", unsafe_allow_html=True)

def compare_resumes_tab():
    st.subheader("Compare Two Resumes")
    uploaded_file1 = st.file_uploader("Upload Resume 1 (PDF or Text)", type=["pdf", "txt"])
    uploaded_file2 = st.file_uploader("Upload Resume 2 (PDF or Text)", type=["pdf", "txt"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        resume1_text = ""
        resume2_text = ""

        # Read text from uploaded files
        if uploaded_file1.type == "application/pdf":
            pdf_reader1 = PyPDF2.PdfReader(uploaded_file1)
            for page in pdf_reader1.pages:
                resume1_text += page.extract_text()
        else:
            resume1_text = uploaded_file1.getvalue().decode("utf-8")

        if uploaded_file2.type == "application/pdf":
            pdf_reader2 = PyPDF2.PdfReader(uploaded_file2)
            for page in pdf_reader2.pages:
                resume2_text += page.extract_text()
        else:
            resume2_text = uploaded_file2.getvalue().decode("utf-8")

        # Clean text
        cleaned_resume1 = clean_text(resume1_text)
        cleaned_resume2 = clean_text(resume2_text)

        # Calculate similarity
        tfidf_loaded, _ = load_models()
        similarity_score = calculate_resume_score(tfidf_loaded, cleaned_resume1, cleaned_resume2)
        formatted_score = f"{similarity_score * 100:.2f}%"
        st.write(f"### Similarity Score between Resumes: <span style='color:blue'>{formatted_score}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
