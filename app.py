import pickle
import streamlit as st
import PyPDF2
import re
from sklearn.exceptions import NotFittedError

# Load the saved TF-IDF vectorizer and Logistic Regression model
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

with open('clf.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)

import re

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

# Function to map category ID to category name
def get_category_name(prediction_id):
    category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Designing",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operations Manager",
        6: "Data Science",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts",
        7: "Database",
        11: "Electrical Engineering",
        14: "Health and Fitness",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Security Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
        # Add more category mappings as needed
    }
    return category_mapping.get(prediction_id, "Unknown")

# Streamlit app
def main():
    st.title("Resume Category Prediction")
    st.write("Upload a resume (PDF) to predict its category.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read and preprocess the uploaded PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        
        for page in pdf_reader.pages:
            resume_text += page.extract_text()

        # Preprocess the resume text
        processed_resume_text = preprocess_text(resume_text)

        # Transform the preprocessed text using TF-IDF vectorizer
        input_features = tfidf_vectorizer.transform([processed_resume_text])

        try:
            # Make prediction using the loaded classifier
            prediction_id = clf.predict(input_features)[0]
            category_name = get_category_name(prediction_id)

            # Display predicted category
            st.success(f"Predicted Category: {category_name}")
        except NotFittedError:
            st.error("Model is not fitted. Please retrain the model and try again.")

if __name__ == '__main__':
    main()
