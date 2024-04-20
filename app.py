import streamlit as st
import PyPDF2
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to preprocess text
def clean_resume(data):
    """
    Clean the input resume text by removing URLs, emails, special characters, and stopwords.
    
    :param data: The input text (resume)
    :return: The cleaned text
    """
    # Remove URLs
    data = re.sub(r'http\S+', '', data)
    # Remove emails
    data = re.sub(r'\S*@\S*\s?', '', data)
    # Remove special characters
    data = re.sub(r'[^a-zA-Z\s]', '', data)
    # Convert to lowercase
    data = data.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    data = ' '.join(word for word in data.split() if word not in stop_words)
    return data

# Load the trained TF-IDF vectorizer and Logistic Regression classifier
with open('tfidf.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)

# Function to predict category
def predict_category(resume_text):
    """
    Predict the category of the input resume text.
    
    :param resume_text: The input resume text
    :return: The predicted category name
    """
    # Clean the resume text
    cleaned_resume = clean_resume(resume_text)
    # Transform the cleaned resume text using TF-IDF vectorizer
    input_features = tfidf_vectorizer.transform([cleaned_resume])
    # Make prediction using the loaded classifier
    prediction_id = clf.predict(input_features)[0]
    # Map category ID to category name
    category_mapping = {
        15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
        20: "Python Developer", 24: "Web Designing", 12: "HR",
        13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
        18: "Operations Manager", 6: "Data Science", 22: "Sales",
        16: "Mechanical Engineer", 1: "Arts", 7: "Database",
        11: "Electrical Engineering", 14: "Health and Fitness",
        19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
        2: "Automation Testing", 17: "Network Security Engineer",
        21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
    }
    predicted_category = category_mapping.get(prediction_id, "Unknown")
    return predicted_category

# Streamlit app
def main():
    st.title("Resume Category Prediction")
    st.write("Upload a resume (PDF) to predict its category.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read PDF file and extract text
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()

        st.write("Extracted Text from Resume:")
        st.write(resume_text)

        # Predict category
        predicted_category = predict_category(resume_text)

        # Display predicted category
        st.success(f"Predicted Category: {predicted_category}")

if __name__ == '__main__':
    main()
