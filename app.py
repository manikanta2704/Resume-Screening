import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import io

# Download NLTK resources (stopwords) if not already downloaded
nltk.download('stopwords')

# Load the saved TF-IDF vectorizer and classifier
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf_loaded = pickle.load(tfidf_file)

with open('clf.pkl', 'rb') as clf_file:
    clf_loaded = pickle.load(clf_file)

# Function to clean text
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
    clean_text = url_pattern.sub('', text)

    # Remove emails
    clean_text = email_pattern.sub('', clean_text)

    # Remove special characters (keeping only words and whitespace)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)

    # Remove stop words by filtering the split words of the text
    stop_words = set(stopwords.words('english'))
    clean_text = ' '.join(word for word in clean_text.split() if word.lower() not in stop_words)

    return clean_text

# Define the category mapping
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

# Define the Streamlit app
def main():
    st.title("Resume Screening App")

    # Sidebar options
    option = st.sidebar.selectbox("Choose Option", ["Upload Resume", "Enter Resume Text"])

    if option == "Upload Resume":
        uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

        if uploaded_file is not None:
            try:
                # Read the uploaded file as bytes
                file_bytes = uploaded_file.read()

                # Attempt to decode as UTF-8
                try:
                    text = file_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    # If UTF-8 decoding fails, attempt other common encodings
                    decoded_text = None
                    common_encodings = ['latin1', 'utf-16', 'iso-8859-1']
                    for encoding in common_encodings:
                        try:
                            text = file_bytes.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise ValueError("Unable to decode the uploaded file using any common encoding.")

                st.write("Resume Text:")
                st.write(text)

                # Clean and predict on uploaded resume
                cleaned_text = clean_text(text)
                input_features = tfidf_loaded.transform([cleaned_text])
                prediction_id = clf_loaded.predict(input_features)[0]
                predicted_category = category_mapping.get(prediction_id, "Unknown")

                st.success(f"Predicted Category: {predicted_category}")

            except Exception as e:
                st.error(f"Error occurred: {e}")

    elif option == "Enter Resume Text":
        user_input = st.text_area("Enter your resume text here")

        if st.button("Predict"):
            cleaned_text = clean_text(user_input)
            input_features = tfidf_loaded.transform([cleaned_text])
            prediction_id = clf_loaded.predict(input_features)[0]
            predicted_category = category_mapping.get(prediction_id, "Unknown")

            st.success(f"Predicted Category: {predicted_category}")

if __name__ == '__main__':
    main()
