import streamlit as st
import PyPDF2
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Function to clean text
@st.cache
def clean_text(text):
    """
    Clean the input text by removing URLs, emails, special characters, and stop words.

    :param text: The string to be cleaned
    :return: The cleaned string
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    cleaned_text = url_pattern.sub('', text)
    cleaned_text = email_pattern.sub('', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    stop_words = set(stopwords.words('english'))
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word.lower() not in stop_words)

    return cleaned_text

def load_models():
    # Load the trained TF-IDF vectorizer
    with open('tfidf.pkl', 'rb') as tfidf_file:
        tfidf_loaded = pickle.load(tfidf_file)

    # Load the trained Logistic Regression classifier
    with open('clf.pkl', 'rb') as clf_file:
        clf_loaded = pickle.load(clf_file)

    return tfidf_loaded, clf_loaded

def predict_category(cleaned_resume_text, tfidf_vectorizer, classifier):
    # Transform cleaned resume text using TF-IDF vectorizer
    input_features = tfidf_vectorizer.transform([cleaned_resume_text])

    # Make prediction using the loaded classifier
    prediction_id = classifier.predict(input_features)[0]

    # Map category ID to category name
    category_mapping = {
        15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
        24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
        18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
        1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
        19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
        17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
    }

    predicted_category = category_mapping.get(prediction_id, "Unknown")
    return predicted_category

def main():
    st.title("Automated Resume Screening App")

    # Load models (TF-IDF vectorizer and Logistic Regression classifier)
    tfidf_loaded, clf_loaded = load_models()

    # Select the mode (Text or PDF)
    selected_mode = st.sidebar.radio("Choose Input Mode", ["Text", "PDF"])

    if selected_mode == "Text":
        st.subheader("Enter Text Resume")
        text_resume = st.text_area("Paste your Resume text here", height=300)

        if st.button("Predict Category"):
            cleaned_text = clean_text(text_resume)
            predicted_category = predict_category(cleaned_text, tfidf_loaded, clf_loaded)
            st.write(f"Predicted Category: {predicted_category}")

    elif selected_mode == "PDF":
        st.subheader("Upload your Resume in .pdf format")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extracted_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()

            if extracted_text:
                cleaned_text = clean_text(extracted_text)
                predicted_category = predict_category(cleaned_text, tfidf_loaded, clf_loaded)
                st.write(f"Predicted Category: {predicted_category}")

# Run the main function to start the app
if __name__ == "__main__":
    main()
