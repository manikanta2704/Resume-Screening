import pickle
import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)      # Remove URLs
    text = re.sub(r'RT|cc', '', text)        # Remove RT and cc
    text = re.sub(r'#\S+', '', text)         # Remove hashtags
    text = re.sub(r'@\S+', '', text)         # Remove mentions
    text = text.lower()                      # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)         # Replace multiple spaces with single space
    text = text.strip()                      # Strip leading/trailing whitespace
    return text

# Streamlit app
def main():
    st.title("Resume Category Prediction")
    st.write("Upload a resume (PDF) to predict its category.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read and preprocess the uploaded PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            resume_text += page.extract_text()

        # Preprocess the resume text
        processed_resume_text = preprocess_text(resume_text)

        # Recreate and fit TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer()
        X_train = [processed_resume_text]  # Example: Use your training data here
        tfidf_vectorizer.fit(X_train)

        # Recreate and fit Logistic Regression model
        clf = LogisticRegression()
        y_train = [0]  # Example: Use your training labels here
        clf.fit(tfidf_vectorizer.transform(X_train), y_train)

        # Save the models
        with open('tfidf.pkl', 'wb') as tfidf_file:
            pickle.dump(tfidf_vectorizer, tfidf_file)

        with open('clf.pkl', 'wb') as clf_file:
            pickle.dump(clf, clf_file)

        st.success("Models saved successfully!")

if __name__ == '__main__':
    main()
