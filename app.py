import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import re
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

# Load the trained TF-IDF vectorizer and logistic regression classifier
with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf_loaded = pickle.load(tfidf_file)

with open('clf.pkl', 'rb') as clf_file:
    clf_loaded = pickle.load(clf_file)

# Main function to run the app
def main():
    st.title("Resume Screening App")

    styles = """
    <style>
    body {
        background-color: #f0f0f0; /* Light gray background */
        color: #333; /* Dark text color */
        padding: 20px; /* Add padding for content */
    }
    .content-container {
        display: flex;
        justify-content: space-between;
    }
    .left-panel {
        flex: 1;
        padding: 20px;
        background-color: #e0e0e0; /* Light background color for left panel */
        margin-right: 10px; /* Add margin between left and right panels */
    }
    .right-panel {
        flex: 1;
        padding: 20px;
        background-color: #d3d3d3; /* Light background color for right panel */
    }
    </style>
    """

    # Render CSS styles
    st.markdown(styles, unsafe_allow_html=True)

    # Render content in left and right panels
    st.write("<div class='content-container'>", unsafe_allow_html=True)
    st.write("<div class='left-panel'>Left Panel Content (PDF)</div>", unsafe_allow_html=True)
    st.write("<div class='right-panel'>Right Panel Content (Text)</div>", unsafe_allow_html=True)
    st.write("</div>", unsafe_allow_html=True)
    
    # Use sidebar to navigate between pages (PDF and Text)
    selected_page = st.sidebar.radio("Navigate", ["PDF", "Text"])

    if selected_page == "PDF":
        st.subheader("Upload PDF Resume")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            page_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text += page.extract_text()

            st.write("### Extracted Text:")
            st.write(page_text)

            cleaned_text = clean_text(page_text)
            st.write("### Cleaned Text:")
            st.write(cleaned_text)

            if st.button("Predict"):
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

        if st.button("Predict"):
            # Clean the text
            cleaned_text = clean_text(text_resume)
            st.write("### Cleaned Text:")
            st.write(cleaned_text)

            # Vectorize the cleaned text
            input_features = tfidf_loaded.transform([cleaned_text])

            # Make predictions using the loaded classifier
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
            st.write("### Predicted Category:")
            st.write(predicted_category)

# Run the main function to start the app
if __name__ == "__main__":
    main()
