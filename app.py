import streamlit as st
import os
from werkzeug.utils import secure_filename
from uploads.rag_pipeline import PDFChatRAG

# Initialize the RAG Pipeline
rag_pipeline = PDFChatRAG()

# Set up the upload folder for storing temporary files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Streamlit UI: File upload and query input
st.title("PDF Chat Interface")

# File upload widget
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Text area for user to input a query
query = st.text_area("Enter your query about the PDF:")

if st.button("Get Response"):
    if pdf_file is not None and query.strip():
        try:
            # Save the uploaded PDF to the server
            filename = secure_filename(pdf_file.name)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            with open(filepath, "wb") as f:
                f.write(pdf_file.getbuffer())

            # Process the query using the RAG pipeline
            response = rag_pipeline.chat_with_pdf(filepath, query)

            # Clean up the temporary file
            os.remove(filepath)

            # Display the response
            st.write("Response:", response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload a PDF and enter a query.")
