import streamlit as st
import re
import pickle
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
word_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('clf.pkl.gz', 'rb'))

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
        pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
        resume_text = ""
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            resume_text += page.extractText()
        
        cleaned_resume_text = clean_resume_text(resume_text)

        # Process the cleaned resume text for prediction
        input_features = word_vectorizer.transform([cleaned_resume_text])

        # Make prediction using the loaded classifier
        prediction_id = clf.predict(input_features)[0]
        category_name = get_category_name(prediction_id)

        # Display predicted category
        st.success(f"Predicted Category: {category_name}")

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

if __name__ == '__main__':
    main()
