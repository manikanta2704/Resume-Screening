import streamlit as st
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download NLTK resources (stopwords) if not already downloaded
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    """
    Clean the input text by removing URLs, emails, special characters, and stop words.

    :param text: The string to be cleaned
    :return: The cleaned string
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    clean_text = url_pattern.sub('', text)
    clean_text = email_pattern.sub('', clean_text)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    clean_text = ' '.join(word for word in clean_text.split() if word.lower() not in stop_words)
    return clean_text

# Load TF-IDF vectorizer and Logistic Regression classifier
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf_loaded = pickle.load(tfidf_file)

with open('clf.pkl', 'rb') as clf_file:
    clf_loaded = pickle.load(clf_file)

# Define the Streamlit app
def main():
    st.title("Resume Category Prediction")

    # Text area for user to enter resume
    user_input = st.text_area("Paste your resume here:", "")

    if st.button("Submit"):
        if user_input:
            # Clean and transform user's input
            cleaned_input = clean_text(user_input)
            input_features = tfidf_loaded.transform([cleaned_input])

            # Make prediction using loaded classifier
            prediction_id = clf_loaded.predict(input_features)[0]

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

            # Display predicted category
            st.success(f"Predicted Category: {predicted_category}")
        else:
            st.warning("Please enter your resume.")

# Run the app
if __name__ == "__main__":
    main()
