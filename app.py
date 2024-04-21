import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pdfplumber
import pickle


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
    option = st.sidebar.selectbox("Choose Option", ["Upload Resume (PDF)", "Enter Resume Text"])

    if option == "Upload Resume (PDF)":
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

        if uploaded_file is not None:
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    pdf_text = ""
                    for page in pdf.pages:
                        pdf_text += page.extract_text()

                st.write("Resume Text:")
                st.write(pdf_text)

                # Clean and predict on uploaded resume text
                cleaned_text = clean_text(pdf_text)
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
