import streamlit as st
import PyPDF2
from nltk.corpus import stopwords
import re

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

# Main function to run the app
def main():
    st.title("Resume Screening App")
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(45deg, #3a1c71, #d76d77, #ffaf7b);
            color: white;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(45deg, #3a1c71, #d76d77, #ffaf7b);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use sidebar to navigate between pages (PDF and Text)
    selected_page = st.sidebar.radio("Navigate", ["PDF", "Text"])

    if selected_page == "PDF":
        st.subheader("Upload PDF Resume")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            page_text = ""
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                page_text += page.extract_text()

            st.write("### Extracted Text:")
            st.write(page_text)

            cleaned_text = clean_text(page_text)
            st.write("### Cleaned Text:")
            st.write(cleaned_text)

    elif selected_page == "Text":
        st.subheader("Enter Text Resume")
        text_resume = st.text_area("Paste your text here", height=300)

        if st.button("Predict"):
            cleaned_text = clean_text(text_resume)
            st.write("### Cleaned Text:")
            st.write(cleaned_text)

# Run the main function to start the app
if __name__ == "__main__":
    main()
