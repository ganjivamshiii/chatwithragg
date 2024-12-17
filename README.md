# chatwithragg
# PDF Chat RAG (Retrieval-Augmented Generation)

## Overview

The **PDF Chat RAG** is a Python-based pipeline for querying and interacting with PDF documents using Retrieval-Augmented Generation (RAG). This tool combines text extraction, chunking, vector-based retrieval, and transformer-based response generation to provide detailed and contextually accurate answers to user queries about PDF content.

---

## Features

- **Text Extraction**: Extracts text content from PDF files using PyMuPDF (`fitz`).
- **Text Chunking**: Breaks down long texts into manageable chunks with optional overlap for contextual continuity.
- **Semantic Search**: Uses FAISS for efficient similarity search on text embeddings.
- **Response Generation**: Leverages pre-trained models like `google/flan-t5-base` to generate natural language responses based on retrieved text.
- **End-to-End Pipeline**: Enables interactive querying of PDFs with minimal setup.

---

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- Pip

### Required Libraries

Install dependencies using pip:

```bash
pip install numpy nltk fitz faiss-cpu sentence-transformers transformers torch

```

---

## Usage

### Running the Script

1. Place your PDF file in the working directory.
2. Update the `pdf_path` variable in the `main()` function with the path to your PDF file.
3. Run the script:
    
    ```bash
    python <script_name>.py
    
    ```
    

### Example Queries

You can query the document with questions like:

- "Analyze the visual trends in the document."
- "Summarize the key points about data visualization."
- "Extract insights about unemployment from page 2."

---

## Code Details

### Pipeline Steps

1. **Text Extraction**: Extracts text from the PDF using the `extract_text_from_pdf` method.
2. **Chunking**: Splits text into chunks of configurable size and overlap using `chunk_text`.
3. **Embedding and Indexing**: Creates vector embeddings for the chunks using Sentence Transformers and stores them in a FAISS index.
4. **Query Retrieval**: Finds the most relevant chunks for the user query using semantic similarity.
5. **Response Generation**: Generates a natural language response using the FLAN-T5 model.

### Main Classes and Methods

### `PDFChatRAG`

- **`__init__()`**: Initializes embedding and response generation models.
- **`extract_text_from_pdf(pdf_path)`**: Extracts text from the given PDF.
- **`chunk_text(text, chunk_size=300, overlap=50)`**: Splits the text into manageable chunks.
- **`create_vector_index(chunks)`**: Creates a FAISS index from text embeddings.
- **`retrieve_relevant_chunks(query, chunks, index, top_k=3)`**: Retrieves top-K relevant chunks for a query.
- **`generate_response(query, retrieved_chunks)`**: Generates a response based on the query and relevant chunks.
- **`chat_with_pdf(pdf_path, query)`**: Orchestrates the entire process to handle user queries.

---

## Example

```python
def main():
    pdf_path = "data_visualization_practice.pdf"
    rag_pipeline = PDFChatRAG()

    # Example queries
    queries = [
        "Analyze the visual trends in the document",
        "Summarize the key points about data visualization",
        "Extract insights about unemployment from page 2"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_pipeline.chat_with_pdf(pdf_path, query)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()

```

---

## Dependencies

- `numpy`: For numerical computations.
- `nltk`: For text tokenization.
- `fitz`: For PDF text extraction.
- `faiss-cpu`: For fast similarity search.
- `sentence-transformers`: For generating text embeddings.
- `transformers`: For working with transformer models.
- `torch`: For PyTorch-based model inference.

---

## Notes

- Ensure the PDF file is text-based. Scanned PDFs may require OCR for text extraction.
- Adjust `chunk_size` and `overlap` in the `chunk_text` method for optimal chunking based on the document's structure.

---

## Future Improvements

- Add OCR support for scanned PDFs.
- Improve response generation by fine-tuning the response model on domain-specific datasets.
- Extend support for multilingual PDFs.

---

## License

This project is licensed under the MIT License.

---

## Contact

For issues or suggestions, please open an issue on the project's repository.