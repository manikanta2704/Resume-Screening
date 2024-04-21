import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords data if not already downloaded
nltk.download('stopwords')

# Load the TF-IDF vectorizer and Logistic Regression model
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

with open('clf.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)

# Function to preprocess text
def preprocess_text(text):
    # Remove URLs, emails, special characters, and stopwords
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # Remove URLs and emails
    text = url_pattern.sub('', text)
    text = email_pattern.sub('', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.lower().split() if word not in stop_words]
    
    return ' '.join(words)

# Streamlit app
def main():
    st.title("Resume Category Prediction")
    st.write("Enter your resume text below:")
    
    user_input = st.text_area("Input Your Resume Here")

    if st.button("Predict"):
        if user_input:
            # Preprocess user input
            cleaned_input = preprocess_text(user_input)

            # Transform input using TF-IDF vectorizer
            input_features = tfidf_vectorizer.transform([cleaned_input])

            # Make prediction using loaded classifier
            prediction_id = clf.predict(input_features)[0]

            # Map prediction ID to category name
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
                0: "Advocate"
            }

            predicted_category = category_mapping.get(prediction_id, "Unknown")

            # Display predicted category
            st.success(f"Predicted Category: {predicted_category}")
        else:
            st.warning("Please enter your resume text.")

if __name__ == '__main__':
    main()
