import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import re
import pickle
import PyPDF2

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

def load_models():
    # Load the trained TF-IDF vectorizer
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidf_loaded = pickle.load(tfidf_file)

    # Load the trained logistic regression classifier
    with open('clf.pkl', 'rb') as clf_file:
        clf_loaded = pickle.load(clf_file)

    return tfidf_loaded, clf_loaded

def main():
    st.title("Resume Screening App")

    # Use sidebar to navigate between pages (PDF and Text)
    selected_page = st.sidebar.radio("Navigate", ["PDF", "Text"])

    if selected_page == "PDF":
        st.subheader("Upload PDF Resume")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if pdf_file is not None:
            st.write("### Extracted Text:")
            pdf_text = pdf_file.read().decode('utf-8', errors='ignore')
            st.write(pdf_text)

            cleaned_text = clean_text(pdf_text)

            if st.button("Predict Category"):
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
                st.write("### Predicted Category:")
                st.write(predicted_category)

    elif selected_page == "Text":
        st.subheader("Enter Text Resume")
        text_resume = st.text_area("Paste your text here", height=300)

        if st.button("Clean Text"):
            cleaned_text = clean_text(text_resume)
            st.write("### Cleaned Text:")
            st.write(cleaned_text)

            if st.button("Predict Category"):
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
                st.write("### Predicted Category:")
                st.write(predicted_category)

# Run the main function to start the app
if __name__ == "__main__":
    main()
