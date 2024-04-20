import streamlit as st
import re
import pickle
import PyPDF2

# Load the trained model and TF-IDF vectorizer
word_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('clf.pkl.gz', 'rb'))

# Function to clean resume text
def clean_resume_text(data):
    data = re.sub('httpS+s*', ' ', data)  # Remove URLs
    data = re.sub('RT|cc', ' ', data)  # Remove RT and cc
    data = re.sub('#S+', ' ', data)  # Remove hashtags
    data = re.sub('@S+', ' ', data)  # Remove mentions
    data = data.lower()  # Convert to lowercase
    data = ''.join([i if 32 < ord(i) < 128 else ' ' for i in data])  # Remove special characters
    data = re.sub('s+', 's', data)  # Remove extra whitespaces
    data = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', data)  # Remove punctuations
    return data

# Streamlit app
def main():
    # Custom CSS for background and styles
    st.markdown(
        """
        <style>
        body {
            background-image: url('.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            font-family: Arial, sans-serif;
        }
        .stApp {
            padding: 2rem;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 20px;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
            color: #333;
        }
        .stButton {
            background-color: #ff6347;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            transition: background-color 0.3s ease;
        }
        .stButton:hover {
            background-color: #ff4500;
        }
        </style>
        """
    )

    # Main app content
    st.title("Resume Category Prediction")
    st.markdown("---")
    st.write("Upload a resume (PDF) to predict its category.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read PDF file and extract text
        resume_text = extract_text_from_pdf(uploaded_file)
        cleaned_resume_text = clean_resume_text(resume_text)


        # Screen Resume button
        if st.button("Screen Resume"):
            # Process the cleaned resume text for prediction
            input_features = word_vectorizer.transform([cleaned_resume_text])

            # Make prediction using the loaded classifier
            prediction_id = clf.predict(input_features)[0]
            category_name = get_category_name(prediction_id)

            # Display predicted category with a styled message
            st.markdown("---")
            st.success(f"Predicted Category: **{category_name}**")

# Helper function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    num_pages = pdf_reader.numPages
    resume_text = ""
    for page_num in range(num_pages):
        page = pdf_reader.getPage(page_num)
        resume_text += page.extractText()
    return resume_text

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
