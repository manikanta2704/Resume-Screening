import streamlit as st
import re
import pickle
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import set_config

# Disable sklearn warnings about version mismatches
set_config(display='diagram')

# Load the trained TF-IDF vectorizer and Logistic Regression classifier
def load_models():
    # Load TF-IDF vectorizer
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidf_vectorizer = pickle.load(tfidf_file)
    
    # Load Logistic Regression classifier
    with open('clf.pkl', 'rb') as clf_file:
        clf = pickle.load(clf_file)
    
    return tfidf_vectorizer, clf

# Function to clean resume text
def clean_resume_text(data):
    data = re.sub(r'http\S+', '', data)  # Remove URLs
    data = re.sub(r'RT|cc', '', data)  # Remove RT and cc
    data = re.sub(r'#\S+', '', data)  # Remove hashtags
    data = re.sub(r'@\S+', '', data)  # Remove mentions
    data = data.lower()  # Convert to lowercase
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)  # Remove special characters except spaces
    data = re.sub(r'\s+', ' ', data)  # Replace multiple spaces with a single space
    return data.strip()  # Strip leading/trailing spaces

# Streamlit app
def main():
    st.title("Resume Category Prediction")
    st.write("Upload a resume (PDF) to predict its category.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read PDF file and extract text
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            resume_text += page.extract_text()
        
        st.write("Extracted Text from Resume:")
        st.write(resume_text)

        # Load models
        tfidf_vectorizer, clf = load_models()

        # Process the cleaned resume text for prediction
        cleaned_text = clean_resume_text(resume_text)
        input_features = tfidf_vectorizer.transform([cleaned_text])

        # Make prediction using the loaded classifier
        prediction_id = clf.predict(input_features)[0]
        category_name = get_category_name(prediction_id)

        # Display predicted category
        st.success(f"Predicted Category: {category_name}")

import streamlit as st
import re
import pickle
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import set_config

# Disable sklearn warnings about version mismatches
set_config(display='diagram')

# Load the trained TF-IDF vectorizer and Logistic Regression classifier
def load_models():
    # Load TF-IDF vectorizer
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidf_vectorizer = pickle.load(tfidf_file)
    
    # Load Logistic Regression classifier
    with open('clf.pkl', 'rb') as clf_file:
        clf = pickle.load(clf_file)
    
    return tfidf_vectorizer, clf

# Function to clean resume text
def clean_resume_text(data):
    data = re.sub(r'http\S+', '', data)  # Remove URLs
    data = re.sub(r'RT|cc', '', data)  # Remove RT and cc
    data = re.sub(r'#\S+', '', data)  # Remove hashtags
    data = re.sub(r'@\S+', '', data)  # Remove mentions
    data = data.lower()  # Convert to lowercase
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data)  # Remove special characters except spaces
    data = re.sub(r'\s+', ' ', data)  # Replace multiple spaces with a single space
    return data.strip()  # Strip leading/trailing spaces

# Streamlit app
def main():
    st.title("Resume Category Prediction")
    st.write("Upload a resume (PDF) to predict its category.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read PDF file and extract text
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            resume_text += page.extract_text()
        
        st.write("Extracted Text from Resume:")
        st.write(resume_text)

        # Load models
        tfidf_vectorizer, clf = load_models()

        # Process the cleaned resume text for prediction
        cleaned_text = clean_resume_text(resume_text)
        input_features = tfidf_vectorizer.transform([cleaned_text])

        # Make prediction using the loaded classifier
        prediction_id = clf.predict(input_features)[0]
        category_name = get_category_name(prediction_id)

        # Display predicted category
        st.success(f"Predicted Category: {category_name}")

# Function to map category ID to category name
def get_category_name(prediction_id):
    category_mapping = {
        # Define your category mappings based on prediction IDs
        0: "Advocate",
        1: "Arts",
        2: "Automation Testing",
        # Add more category mappings as needed
    }
    return category_mapping.get(prediction_id, "Unknown")

if __name__ == '__main__':
    main()

    return category_mapping.get(prediction_id, "Unknown")

if __name__ == '__main__':
    main()
